# models/sd3_light.py
# Final, diff-ready LightSD3Pipeline
#
# Guarantees:
# - Saved directory is a valid diffusers StableDiffusion3Pipeline
# - model_index.json is always generated
# - Denoiser (SD3Transformer2DModel) is RANDOMLY initialized
# - VAE + 3 text encoders stay pretrained & frozen
# - Only denoiser is trainable
# - Future runs load ONLY from saved init dir (no SD3 dependency)

import math
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
from diffusers.models.modeling_utils import ModelMixin
import torch
from torch import nn
import torch.nn.functional as F
import diffusers

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE


KEEP_IN_HIGH_PRECISION = [
    "pos_embed",
    "time_text_embed",
    "context_embedder",
    "norm_out",
    "proj_out",
]


# -----------------------------------------------------------------------------
# Utilities (copied from SD3 pipeline)
# -----------------------------------------------------------------------------
def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b



def _has_meta_params(module) -> bool:
    for _, p in module.named_parameters(recurse=True):
        if getattr(p, "is_meta", False) or str(p.device) == "meta":
            return True
    return False

def _pipe_has_meta(pipe) -> bool:
    for name in ["vae", "text_encoder", "text_encoder_2", "text_encoder_3", "transformer"]:
        m = getattr(pipe, name, None)
        if m is not None and _has_meta_params(m):
            return True
    return False

def _extract_transformer_sd_from_ds(diffusers_sd: dict):
    """
    DS Saver 传进来的 diffusers_sd 是扁平 dict：{original_name: tensor}
    我们尽量只抽 transformer.*，并把前缀 strip 掉以匹配 transformer.load_state_dict
    """
    if not diffusers_sd:
        return None
    out = {}
    for k, v in diffusers_sd.items():
        if k.startswith("transformer."):
            out[k[len("transformer."):]] = v
    return out if len(out) > 0 else None



# -----------------------------------------------------------------------------
# LightSD3Pipeline
# -----------------------------------------------------------------------------
class LightSD3Pipeline(BasePipeline):
    """
    Light SD3 Pipeline (pretrain-ready)

    - Architecture identical to SD3
    - Denoiser (transformer) is RANDOMLY initialized
    - num_layers configurable
    - No LoRA support
    - After one-time bootstrap+save, all training loads from saved dir
    """

    name = "sd3"
    checkpointable_layers = ["TransformerLayer"]
    adapter_target_modules = []  # explicitly no LoRA

    # -------------------------------------------------------------------------
    # Bootstrap init
    # -------------------------------------------------------------------------
    def __init__(self, config: Dict[str, Any]):
        """
        Bootstrap mode only.
        Later training MUST use load_from_pretrained().
        """
        self.config = config
        self.model_config = config["model"]
        self._latest_loss_breakdown = None

        dtype = self.model_config["dtype"]
        diffusers_path = self.model_config.get("diffusers_path", None)
        if diffusers_path is None:
            raise ValueError(
                "LightSD3Pipeline bootstrap requires model.diffusers_path. "
                "For training, use LightSD3Pipeline.load_from_pretrained()."
            )

        # IMPORTANT:
        # Load FULL SD3 pipeline (including pretrained transformer)
        # We will overwrite transformer immediately after.
        self.diffusers_pipeline = diffusers.StableDiffusion3Pipeline.from_pretrained(
            diffusers_path,
            torch_dtype=dtype,
        )

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)
        
    # --- add into class LightSD3Pipeline ---

    def to(self, *args, **kwargs):
        # diffusers_pipeline.to(...) 会返回 StableDiffusion3Pipeline；我们这里强制返回 self
        self.diffusers_pipeline.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        self.diffusers_pipeline.cuda(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        self.diffusers_pipeline.cpu(*args, **kwargs)
        return self

    def half(self, *args, **kwargs):
        self.diffusers_pipeline.half(*args, **kwargs)
        return self

    def float(self, *args, **kwargs):
        self.diffusers_pipeline.float(*args, **kwargs)
        return self

    def _iter_pipeline_modules(self):
        if hasattr(self.diffusers_pipeline, "components"):
            for module in self.diffusers_pipeline.components.values():
                if isinstance(module, nn.Module):
                    yield module
            return

        for name in ["transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"]:
            module = getattr(self.diffusers_pipeline, name, None)
            if isinstance(module, nn.Module):
                yield module

    def eval(self):
        for module in self._iter_pipeline_modules():
            module.eval()
        return self

    def train(self, mode: bool = True):
        for module in self._iter_pipeline_modules():
            module.train(mode)
        return self


    def load_ema_shadow(self, ema_shadow_path: str, strict: bool = False):
        """
        ema_shadow_path: torch.save() 的 dict，key 可能是:
        - "transformer.xxx" (推荐)
        - 或者直接是 transformer state_dict 的 key（不带前缀）
        """
        import torch

        ema_sd = torch.load(ema_shadow_path, map_location="cpu")
        if (not isinstance(ema_sd, dict)) or len(ema_sd) == 0:
            raise ValueError(f"Invalid ema_shadow at {ema_shadow_path}: empty or not a dict")

        # normalize keys -> transformer.state_dict() keys
        if any(k.startswith("transformer.") for k in ema_sd.keys()):
            tr_sd = {k[len("transformer."):]: v for k, v in ema_sd.items() if k.startswith("transformer.")}
        else:
            tr_sd = ema_sd

        missing, unexpected = self.diffusers_pipeline.transformer.load_state_dict(tr_sd, strict=strict)
        return missing, unexpected

    def load_diffusion_model(self):
        dtype = self.model_config["dtype"]
        transformer_dtype = self.model_config.get("transformer_dtype", dtype)
        diffusers_path = self.model_config["diffusers_path"]

        # 1. 读取 base config（仅用于结构参数）
        base_transformer = diffusers.SD3Transformer2DModel.from_pretrained(
            diffusers_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        base_config = base_transformer.config
        del base_transformer

        # 2. 用 config 构建「完整」随机 transformer
        transformer = diffusers.SD3Transformer2DModel.from_config(base_config)

        # 3. 显式裁剪 transformer blocks
        num_layers = self.model_config.get("num_layers", None)
        if num_layers is not None:
            num_layers = int(num_layers)
            assert num_layers > 0, "num_layers must be > 0"

            total = len(transformer.transformer_blocks)
            if num_layers > total:
                raise ValueError(
                    f"num_layers={num_layers} > available blocks={total}"
                )

            transformer.transformer_blocks = nn.ModuleList(
                transformer.transformer_blocks[:num_layers]
            )

            # 同步修正 config（很重要，影响 save / reload / pipeline parallel）
            if hasattr(transformer.config, "num_layers"):
                transformer.config.num_layers = num_layers
            if hasattr(transformer.config, "num_transformer_blocks"):
                transformer.config.num_transformer_blocks = num_layers

            print(f"[SD3-Light] Transformer blocks truncated: {total} → {num_layers}")

        # 4. dtype cast（保持你原来的逻辑）
        for n, p in transformer.named_parameters():
            if not (any(k in n for k in KEEP_IN_HIGH_PRECISION) or p.ndim == 1):
                p.data = p.data.to(transformer_dtype)

        # 5. attach
        self.diffusers_pipeline.transformer = transformer
        
        print(f"[SD3-Light] Transformer blocks:{len(self.diffusers_pipeline.transformer.transformer_blocks)}")

        # 6. 冻结非 denoiser
        self._freeze_except_denoiser()


    def _freeze_except_denoiser(self):
        # Freeze VAE
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # Freeze text encoders
        for te in self.get_text_encoders():
            te.eval()
            for p in te.parameters():
                p.requires_grad_(False)

        # Train denoiser
        self.transformer.train()
        for p in self.transformer.parameters():
            p.requires_grad_(True)

    # -------------------------------------------------------------------------
    # One-time bootstrap
    # -------------------------------------------------------------------------
    def build_random_init_and_save(self, save_dir: Union[str, Path]):
        """
        Build random-init denoiser and save FULL diffusers pipeline.
        """
        self.load_diffusion_model()
        self.save_model(save_dir)



    @classmethod
    def load_from_pretrained(
        cls,
        model_dir,
        dtype,
        transformer_dtype=None,
        extra_model_config=None,
    ):
        model_dir = str(model_dir)

        # 1. 先正常构造一个 LightSD3Pipeline 实例（走 __init__）
        dummy_config = {
            "model": {
                "dtype": dtype,
                "transformer_dtype": transformer_dtype or dtype,
                "diffusers_path": model_dir,
            }
        }
        if extra_model_config:
            dummy_config["model"].update(extra_model_config)

        self = cls(dummy_config)   # ✅ 关键：真正构造 LightSD3Pipeline

        # 2. 用保存好的目录加载完整 diffusers pipeline
        pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            device_map=None,
        )

        self.diffusers_pipeline = pipe
        self.config = dummy_config
        self.model_config = dummy_config["model"]
        self._latest_loss_breakdown = None

        # 3. 🔥 裁剪 transformer（SD3-Light-15 的关键）
        num_layers = self.model_config.get("num_layers", None)
        if num_layers is not None:
            num_layers = int(num_layers)
            transformer = self.diffusers_pipeline.transformer

            total = len(transformer.transformer_blocks)
            if num_layers > total:
                raise ValueError(f"num_layers={num_layers} > loaded={total}")

            if total != num_layers:
                transformer.transformer_blocks = nn.ModuleList(
                    transformer.transformer_blocks[:num_layers]
                )

                if hasattr(transformer.config, "num_layers"):
                    transformer.config.num_layers = num_layers
                if hasattr(transformer.config, "num_transformer_blocks"):
                    transformer.config.num_transformer_blocks = num_layers

                print(
                    f"[SD3-Light] load_from_pretrained: "
                    f"transformer blocks truncated {total} → {num_layers}"
                )

        # 4. dtype cast（和 bootstrap 行为对齐）
        if transformer_dtype is not None:
            for n, p in self.transformer.named_parameters():
                if not (any(k in n for k in KEEP_IN_HIGH_PRECISION) or p.ndim == 1):
                    p.data = p.data.to(transformer_dtype)

        # 5. 冻结 / 解冻
        self._freeze_except_denoiser()

        return self




    def _pipe_has_meta(pipe) -> bool:
        comps = ["transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"]
        for c in comps:
            m = getattr(pipe, c, None)
            if m is None:
                continue
            for _, p in m.named_parameters(recurse=True):
                if getattr(p, "is_meta", False) or str(p.device) == "meta":
                    return True
        return False


    def save_model(self, save_dir, diffusers_sd=None):
        import os
        import diffusers
        import torch

        save_dir = str(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        dtype = self.model_config["dtype"]
        base_dir = self.model_config.get("diffusers_path", None)
        if base_dir is None:
            raise ValueError("model_config['diffusers_path'] is required for safe saving.")

        # 1) 确保 frozen 组件（VAE/TE）不是 meta：必要时从 base_dir 重新 load 一套完整 pipe
        if _pipe_has_meta(self.diffusers_pipeline):
            save_pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
                base_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                device_map=None,
            )
        else:
            save_pipe = self.diffusers_pipeline

        # 2) 用 diffusers_sd 覆盖 transformer（这是关键）
        if diffusers_sd is not None:
            # diffusers_sd 是 Saver 合并出来的 “全模型 key->tensor”
            # 我们只取 transformer 的部分，并把可能的 "transformer." 前缀去掉
            tr_sd = {}
            for k, v in diffusers_sd.items():
                if k.startswith("transformer."):
                    tr_sd[k[len("transformer."):]] = v.detach().to("cpu")
                else:
                    # 如果你的 original_name 本来就不带 transformer. 前缀，也允许直接尝试
                    tr_sd[k] = v.detach().to("cpu")

            missing, unexpected = save_pipe.transformer.load_state_dict(tr_sd, strict=False)
            if len(unexpected) > 0:
                print(f"[WARN] unexpected keys when loading transformer sd: {unexpected[:20]}")
            if len(missing) > 0:
                print(f"[WARN] missing keys when loading transformer sd: {missing[:20]}")

        # 3) 最终保存
        save_pipe.save_pretrained(save_dir, safe_serialization=True)

        if save_pipe is not self.diffusers_pipeline:
            del save_pipe



    # -------------------------------------------------------------------------
    # Training hooks
    # -------------------------------------------------------------------------
    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder, self.text_encoder_2, self.text_encoder_3]

    def save_adapter(self, *args, **kwargs):
        raise NotImplementedError("LightSD3Pipeline does not support LoRA.")

    # -------------------------------------------------------------------------
    # Data preparation (unchanged from SD3)
    # -------------------------------------------------------------------------

    def _normalize_prompt_batch(self, prompt_batch):
        if isinstance(prompt_batch, str):
            prompt_batch = [prompt_batch]
        else:
            prompt_batch = list(prompt_batch)
        return ["" if p is None else str(p) for p in prompt_batch]

    def _encode_clip_prompts(self, captions, text_encoder, tokenizer):
        captions = self._normalize_prompt_batch(captions)
        tokenized = tokenizer(
            captions,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(text_encoder.device)
        outputs = text_encoder(input_ids, output_hidden_states=True)
        pooled_prompt_embeds = outputs[0]
        prompt_embeds = outputs.hidden_states[-2].to(dtype=text_encoder.dtype, device=text_encoder.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=text_encoder.dtype, device=text_encoder.device)
        return prompt_embeds, pooled_prompt_embeds

    def _encode_t5_prompts(self, captions):
        captions = self._normalize_prompt_batch(captions)
        max_sequence_length = self.model_config.get('t5_max_length', 256)
        tokenized = self.tokenizer_3(
            captions,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.text_encoder_3.device)
        prompt_embeds = self.text_encoder_3(input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_3.dtype, device=self.text_encoder_3.device)
        return prompt_embeds


    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        if text_encoder == self.text_encoder:
            def fn(caption, is_video):
                assert not any(is_video)
                # prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                #     prompt=caption,
                #     device=text_encoder.device,
                #     clip_model_index=0,
                # )
                prompt_embed, pooled_prompt_embed = self._encode_clip_prompts(
                    captions=caption,
                    text_encoder=text_encoder,
                    tokenizer=self.tokenizer,
                )
                return {'prompt_embed': prompt_embed, 'pooled_prompt_embed': pooled_prompt_embed}
            return fn

        elif text_encoder == self.text_encoder_2:
            def fn(caption, is_video):
                assert not any(is_video)
                # prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                #     prompt=caption,
                #     device=text_encoder.device,
                #     clip_model_index=1,
                # )
                prompt_2_embed, pooled_prompt_2_embed = self._encode_clip_prompts(
                    captions=caption,
                    text_encoder=text_encoder,
                    tokenizer=self.tokenizer_2,
                )
                return {'prompt_2_embed': prompt_2_embed, 'pooled_prompt_2_embed': pooled_prompt_2_embed}
            return fn
        elif text_encoder == self.text_encoder_3:
            def fn(caption, is_video):
                assert not any(is_video)
                # return {'t5_prompt_embed': self._get_t5_prompt_embeds(prompt=caption, device=text_encoder.device)}
                return {'t5_prompt_embed': self._encode_t5_prompts(caption)}
            return fn
        else:
            raise RuntimeError(f'Text encoder {text_encoder.__class__} does not have a function to call it')

    
    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs["latents"]
        if isinstance(latents, dict):
            if "latents" not in latents:
                raise ValueError("Latents dict is missing 'latents' key.")
            latents = latents["latents"]
        latents = latents.float()
        prompt_embed = inputs["prompt_embed"]
        pooled_prompt_embed = inputs["pooled_prompt_embed"]
        prompt_2_embed = inputs["prompt_2_embed"]
        pooled_prompt_2_embed = inputs["pooled_prompt_2_embed"]
        t5_prompt_embed = inputs["t5_prompt_embed"]
        mask = inputs["mask"]

        uncond_dropout_enabled = self.model_config.get("enable_uncond_text_dropout", False)
        uncond_dropout_prob = self.model_config.get("uncond_text_dropout_prob", 0.464)

        if uncond_dropout_enabled and uncond_dropout_prob > 0:
            def _apply_dropout(embeds):
                if embeds is None:
                    return embeds
                if embeds.numel() == 0:
                    return embeds
                shape = (embeds.shape[0],) + (1,) * (embeds.ndim - 1)
                drop_mask = (torch.rand(shape, device=embeds.device) < uncond_dropout_prob)
                return embeds * (~drop_mask).to(embeds.dtype)

            prompt_embed = _apply_dropout(prompt_embed)
            pooled_prompt_embed = _apply_dropout(pooled_prompt_embed)

            prompt_2_embed = _apply_dropout(prompt_2_embed)
            pooled_prompt_2_embed = _apply_dropout(pooled_prompt_2_embed)

            t5_prompt_embed = _apply_dropout(t5_prompt_embed)

        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        bs, c, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode="nearest-exact")

        timestep_sample_method = self.model_config.get("timestep_sample_method", "logit_normal")

        if timestep_sample_method == "logit_normal":
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == "uniform":
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == "logit_normal":
            sigmoid_scale = self.model_config.get("sigmoid_scale", 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get("shift", None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get("flux_shift", False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        return (noisy_latents, t * 1000, prompt_embeds, pooled_prompt_embeds), (target, mask)

    # -------------------------
    # Pipeline parallel layers
    # -------------------------
    def to_layers(self):
        transformer = self.transformer
        text_recon_config = self._text_recon_config()
        layers = [InitialLayer(transformer, text_recon_config)]
        for idx, block in enumerate(transformer.transformer_blocks, start=1):
            layers.append(TransformerLayer(block, idx, text_recon_config))
        layers.append(FinalLayer(transformer, text_recon_config))
        return layers

    def _text_recon_config(self):
        gamma = float(self.model_config.get("text_recon_gamma", 0.0))
        target_layer = int(self.model_config.get("text_recon_target_layer", 0))
        return {
            "enabled": gamma > 0,
            "gamma": gamma,
            "target_layer": target_layer,
        }

    def _update_loss_breakdown(self, denoise_loss, text_loss, gamma: float):
        with torch.no_grad():
            total_loss = denoise_loss
            text_val = None
            if text_loss is not None:
                total_loss = total_loss + gamma * text_loss
                text_val = float(text_loss.detach())

            self._latest_loss_breakdown = {
                "denoise_loss": float(denoise_loss.detach()),
                "text_recon_loss": text_val,
                "total_loss": float(total_loss.detach()),
            }

    def get_loss_breakdown(self):
        return getattr(self, "_latest_loss_breakdown", None)

    def get_loss_fn(self):
        base_loss_fn = super().get_loss_fn()
        text_recon_config = self._text_recon_config()
        text_recon_enabled = text_recon_config["enabled"]
        gamma = text_recon_config["gamma"]

        def loss_fn(output, label):
            target, mask = label
            final_tokens = None
            target_tokens = None

            if text_recon_enabled and isinstance(output, tuple) and len(output) == 3:
                output, final_tokens, target_tokens = output

            denoise_loss = base_loss_fn(output, (target, mask))
            text_loss = None

            if (
                text_recon_enabled
                and final_tokens is not None
                and target_tokens is not None
                and target_tokens.numel() > 0
            ):
                with torch.autocast("cuda", enabled=False):
                    final_tokens = final_tokens.to(torch.float32)
                    target_tokens = target_tokens.to(final_tokens.device, torch.float32)
                    text_loss = F.mse_loss(final_tokens, target_tokens, reduction="mean")

            total_loss = denoise_loss if text_loss is None else denoise_loss + gamma * text_loss
            self._update_loss_breakdown(denoise_loss, text_loss, gamma)

            return total_loss

        return loss_fn


class InitialLayer(nn.Module):
    def __init__(self, model, text_recon_config):
        super().__init__()
        self.pos_embed = model.pos_embed
        self.time_text_embed = model.time_text_embed
        self.context_embedder = model.context_embedder
        self.model = [model]
        self.text_recon_enabled = text_recon_config["enabled"]
        self.text_recon_target_layer = text_recon_config["target_layer"]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast("cuda", dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        hidden_states, timestep, encoder_hidden_states, pooled_projections = inputs

        height, width = hidden_states.shape[-2:]
        latent_size = torch.tensor([height, width], device=hidden_states.device)

        hidden_states = self.pos_embed(hidden_states)  # adds positional embeddings
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        result = make_contiguous(hidden_states, temb, latent_size, encoder_hidden_states)

        if self.text_recon_enabled:
            target_tokens = (
                encoder_hidden_states.detach()
                if self.text_recon_target_layer == 0
                else torch.empty(0, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            )
            result += (target_tokens,)

        return result


class TransformerLayer(nn.Module):
    def __init__(self, block, layer_idx, text_recon_config):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.text_recon_enabled = text_recon_config["enabled"]
        self.text_recon_target_layer = text_recon_config["target_layer"]

    @torch.autocast("cuda", dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, temb, latent_size, *extra = inputs
        encoder_hidden_states = extra[0] if len(extra) > 0 else None
        target_tokens = None

        if self.text_recon_enabled:
            target_tokens = extra[1] if len(extra) > 1 else None

        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )

        result = make_contiguous(hidden_states, temb, latent_size)
        if encoder_hidden_states is not None:
            result += (encoder_hidden_states,)

        if self.text_recon_enabled:
            if target_tokens is None:
                target_tokens = torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)
            if self.layer_idx == self.text_recon_target_layer:
                target_tokens = encoder_hidden_states.detach()
            result += (target_tokens,)
        return result


class FinalLayer(nn.Module):
    def __init__(self, model, text_recon_config):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.model = [model]
        self.text_recon_enabled = text_recon_config["enabled"]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast("cuda", dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, temb, latent_size, *extra = inputs
        height = int(latent_size[0].item())
        width = int(latent_size[1].item())

        encoder_hidden_states = extra[0] if len(extra) > 0 else None
        target_tokens = extra[1] if len(extra) > 1 else None

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        # IMPORTANT: patch_size belongs to transformer config (diffusers), not pipeline config.
        patch_size = int(self.model[0].config.patch_size)

        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if self.text_recon_enabled and encoder_hidden_states is not None:
            if target_tokens is None:
                target_tokens = torch.empty(0, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            return output, encoder_hidden_states, target_tokens

        return output
