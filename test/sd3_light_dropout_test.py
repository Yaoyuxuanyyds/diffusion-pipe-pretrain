import importlib.machinery
import sys
import types
from pathlib import Path

import torch

peft_stub = types.SimpleNamespace()
peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)


class _DummyLoraConfig:
    @classmethod
    def from_pretrained(cls, *_, **__):
        return cls()


peft_stub.LoraConfig = _DummyLoraConfig
peft_stub.get_peft_model = lambda *_, **__: None
sys.modules.setdefault("peft", peft_stub)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sd3_light import LightSD3Pipeline


def _make_pipeline(config):
    pipe = LightSD3Pipeline.__new__(LightSD3Pipeline)
    pipe.model_config = config
    return pipe


def _build_inputs(batch_size=2):
    latents = torch.zeros(batch_size, 4, 8, 8)
    prompt_embed = torch.ones(batch_size, 3, 8)
    pooled_prompt_embed = torch.ones(batch_size, 16)

    prompt_2_embed = torch.ones(batch_size, 3, 8) * 2
    pooled_prompt_2_embed = torch.ones(batch_size, 16) * 2

    t5_prompt_embed = torch.ones(batch_size, 2, 24) * 3
    mask = None

    return {
        "latents": latents,
        "prompt_embed": prompt_embed,
        "pooled_prompt_embed": pooled_prompt_embed,
        "prompt_2_embed": prompt_2_embed,
        "pooled_prompt_2_embed": pooled_prompt_2_embed,
        "t5_prompt_embed": t5_prompt_embed,
        "mask": mask,
    }


def test_dropout_disabled_keeps_embeddings():
    config = {
        "timestep_sample_method": "uniform",
        "enable_uncond_text_dropout": False,
    }
    pipe = _make_pipeline(config)
    inputs = _build_inputs()

    ( _noisy_latents, _timesteps, prompt_embeds, pooled_prompt_embeds), _ = pipe.prepare_inputs(inputs)

    clip_prompt_embeds = torch.cat([inputs["prompt_embed"], inputs["prompt_2_embed"]], dim=-1)
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, inputs["t5_prompt_embed"].shape[-1] - clip_prompt_embeds.shape[-1])
    )
    expected_prompt_embeds = torch.cat([clip_prompt_embeds, inputs["t5_prompt_embed"]], dim=-2)
    expected_pooled = torch.cat([inputs["pooled_prompt_embed"], inputs["pooled_prompt_2_embed"]], dim=-1)

    assert torch.allclose(prompt_embeds, expected_prompt_embeds)
    assert torch.allclose(pooled_prompt_embeds, expected_pooled)


def test_dropout_zeroes_all_when_prob_one():
    config = {
        "timestep_sample_method": "uniform",
        "enable_uncond_text_dropout": True,
        "uncond_text_dropout_prob": 1.0,
    }
    pipe = _make_pipeline(config)
    inputs = _build_inputs()

    (_noisy_latents, _timesteps, prompt_embeds, pooled_prompt_embeds), _ = pipe.prepare_inputs(inputs)

    clip_seq_len = inputs["prompt_embed"].shape[1]
    t5_seq_len = inputs["t5_prompt_embed"].shape[1]

    clip_part = prompt_embeds[:, :clip_seq_len]
    t5_part = prompt_embeds[:, clip_seq_len:clip_seq_len + t5_seq_len]

    assert torch.count_nonzero(clip_part) == 0
    assert torch.count_nonzero(t5_part) == 0
    assert torch.count_nonzero(pooled_prompt_embeds) == 0


if __name__ == "__main__":
    test_dropout_disabled_keeps_embeddings()
    test_dropout_zeroes_all_when_prob_one()
    print("sd3_light dropout tests passed")
