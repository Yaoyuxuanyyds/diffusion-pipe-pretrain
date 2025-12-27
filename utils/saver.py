from pathlib import Path
import os
import shutil
import time
import sys

import torch
from deepspeed import comm as dist
from deepspeed.utils.logging import logger

from utils.common import is_main_process


def convert_state_dict_dtype(state_dict, dtype):
    for key, v in state_dict.items():
        state_dict[key] = v.to(device='cpu', dtype=dtype)


last_checkpoint_time = None
def need_to_checkpoint(config, epoch=None):
    global last_checkpoint_time

    if epoch is not None:
        if 'checkpoint_every_n_epochs' in config and epoch % config['checkpoint_every_n_epochs'] == 0:
            last_checkpoint_time = time.time()
            return True
        else:
            return False

    if 'checkpoint_every_n_minutes' not in config:
        return False

    checkpoint = False
    if is_main_process():
        current_time = time.time()
        if last_checkpoint_time is None:
            last_checkpoint_time = current_time
        elif (current_time - last_checkpoint_time) / 60 > config['checkpoint_every_n_minutes']:
            checkpoint = True
            last_checkpoint_time = current_time
    result = [checkpoint]
    torch.distributed.broadcast_object_list(result, src=0)
    return result[0]


class Saver:
    def __init__(self, args, config, is_adapter, save_root, model, train_dataloader, model_engine, pipeline_model, ema=None):
        self.ema = ema
        self.args = args
        self.config = config
        self.is_adapter = is_adapter
        self.save_root = Path(save_root)
        self.model = model
        self.train_dataloader = train_dataloader
        self.model_engine = model_engine
        self.pipeline_model = pipeline_model

    def _get_ema_tensor(self, p):
        # 兼容两种 EMA shadow 存法：key 是 param 对象 或 original_name
        if self.ema is None:
            return None
        sh = getattr(self.ema, "shadow", None)
        if sh is None:
            return None
        if p in sh:
            return sh[p]
        if hasattr(p, "original_name") and p.original_name in sh:
            return sh[p.original_name]
        return None

    def save_adapter(self, name):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        tmp_dir = save_dir / 'tmp'
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        dist.barrier()

        if dp_id == 0:
            partial_state_dict = {}
            for pname, p in self.pipeline_model.named_parameters():
                if p.requires_grad:
                    if not hasattr(p, 'original_name'):
                        logger.warning(f'WARNING: parameter {pname} requires_grad but does not have original_name. Not saving it.')
                        continue
                    partial_state_dict[p.original_name.replace('.default', '').replace('.modules_to_save', '')] = p.detach()
            if 'save_dtype' in self.config:
                convert_state_dict_dtype(partial_state_dict, self.config['save_dtype'])
            torch.save(partial_state_dict, tmp_dir / f'state_dict_{stage_id}.bin')

        dist.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in tmp_dir.glob('*.bin'):
                state_dict.update(torch.load(path, weights_only=True, map_location='cpu'))
            self.model.save_adapter(save_dir, state_dict)
            shutil.copy(self.args.config, save_dir)
            shutil.rmtree(tmp_dir)

    def save_full_model(self, name):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()

        save_dir = self.save_root / name
        tmp_dir = save_dir / "tmp"

        # stage0+dp0 创建临时目录（其他人等）
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        dist.barrier()

        # 每个 pipeline stage 的 dp0 负责写本 stage 的 state_dict 分片
        if dp_id == 0:
            partial_state_dict = {}
            total_params = 0
            skipped_params = 0
            for p in self.pipeline_model.parameters():
                total_params += 1
                if not hasattr(p, "original_name"):
                    skipped_params += 1
                    continue
                # detach 防止 optimizer/zero/bf16 optimizer 的 pickle 问题
                partial_state_dict[p.original_name] = p.detach().to("cpu")

            if "save_dtype" in self.config:
                convert_state_dict_dtype(partial_state_dict, self.config["save_dtype"])

            if len(partial_state_dict) == 0:
                logger.warning(
                    "save_full_model: no parameters saved on dp=%s stage=%s "
                    "(total=%s skipped=%s). Check original_name setup.",
                    dp_id,
                    stage_id,
                    total_params,
                    skipped_params,
                )
            else:
                logger.info(
                    "save_full_model: collected %s/%s params (skipped=%s) on dp=%s stage=%s",
                    len(partial_state_dict),
                    total_params,
                    skipped_params,
                    dp_id,
                    stage_id,
                )

            torch.save(partial_state_dict, tmp_dir / f"state_dict_{stage_id}.bin")

        dist.barrier()

        # stage0+dp0 汇总并调用 model.save_model（让 diffusers 保存）
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in tmp_dir.glob("state_dict_*.bin"):
                state_dict.update(torch.load(path, map_location="cpu", weights_only=True))

            if len(state_dict) == 0:
                raise RuntimeError(
                    "save_full_model: merged state_dict is empty. "
                    "No parameters were saved; check original_name setup."
                )
            logger.info(
                "save_full_model: merged %s params across pipeline stages.",
                len(state_dict),
            )

            # 关键：LightSD3Pipeline.save_model 内部会把 state_dict 写回 transformer，
            # 然后 diffusers_pipeline.save_pretrained(save_dir)
            self.model.save_model(save_dir, state_dict)

            # 复制 config + 清理 tmp
            shutil.copy(self.args.config, save_dir)
            shutil.rmtree(tmp_dir)

        dist.barrier()


    def save_model(self, name):
        if is_main_process():
            print(f"Saving model to directory {name}")

        if self.is_adapter:
            self.save_adapter(name)
            return

        # 1) normal weights
        self.save_full_model(name)

        # 2) EMA shadow saved into the SAME directory
        if self.ema is not None and is_main_process():
            save_dir = self.save_root / name
            torch.save(
                self.ema.shadow,
                save_dir / "ema_shadow.pt"
            )


    def save_checkpoint(self, step, examples):
        self.model_engine.save_checkpoint(
            self.save_root,
            client_state={
                'step': step,
                'examples': examples,
                'custom_loader': self.train_dataloader.state_dict(),
            },
            save_latest=True,
            exclude_frozen_parameters=True
        )

    def process_epoch(self, epoch, step, examples):
        checkpointed, saved = False, False
        if self.train_dataloader.epoch != epoch:
            if need_to_checkpoint(self.config, epoch):
                self.save_checkpoint(step, examples)
                checkpointed = True
            if 'save_every_n_epochs' in self.config and epoch % self.config['save_every_n_epochs'] == 0:
                self.save_model(f'epoch{epoch}')
                saved = True
            epoch = self.train_dataloader.epoch
            if epoch > self.config['epochs']:
                return None, checkpointed, saved
            if is_main_process():
                print(f'Started new epoch: {epoch}')
        return epoch, checkpointed, saved

    def process_step(self, step, examples):
        checkpointed, saved = False, False
        should_manually_save = False
        should_manually_quit = False
        save_signal_file = self.save_root / 'save'
        save_quit_signal_file = self.save_root / 'save_quit'

        if save_signal_file.exists() and save_signal_file.is_file():
            should_manually_save = True
            dist.barrier()
            if is_main_process():
                os.remove(save_signal_file)
        elif save_quit_signal_file.exists() and save_quit_signal_file.is_file():
            should_manually_save = True
            should_manually_quit = True
            dist.barrier()
            if is_main_process():
                os.remove(save_quit_signal_file)

        if 'save_every_n_steps' in self.config and step % self.config['save_every_n_steps'] == 0:
            self.save_model(f'step{step}')
            saved = True

        if need_to_checkpoint(self.config) or should_manually_save:
            self.save_checkpoint(step, examples)
            checkpointed = True

        if should_manually_quit:
            print('Manually quitting')
            sys.exit()

        return checkpointed, saved
