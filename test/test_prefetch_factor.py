import sys
import types

import torch
from torch.utils.data import Dataset

# Create lightweight stubs so importing utils.dataset does not require optional ComfyUI deps
comfy_stub = types.ModuleType("comfy")
comfy_model_mgmt_stub = types.ModuleType("comfy.model_management")
comfy_stub.model_management = comfy_model_mgmt_stub
sys.modules.setdefault("comfy", comfy_stub)
sys.modules.setdefault("comfy.model_management", comfy_model_mgmt_stub)

from utils import dataset as dataset_util


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"x": torch.zeros((1, 3, 8, 8)), "mask": None}


class _DummyGrid:
    def get_data_parallel_rank(self):
        return 0

    def get_data_parallel_world_size(self):
        return 1

    def get_model_parallel_rank(self):
        return 0

    def get_model_parallel_world_size(self):
        return 1

    def get_pipe_parallel_rank(self):
        return 0

    def get_pipe_parallel_world_size(self):
        return 1

    @property
    def pp_group(self):
        return [0]


class _DummyModel:
    name = "dummy"
    is_pipe_parallel = False

    @property
    def grid(self):
        return _DummyGrid()

    def gradient_accumulation_steps(self):
        return 1

    def prepare_inputs(self, batch, timestep_quantile=None):
        return batch, (torch.zeros(1), None)


class TestPrefetchFactor:
    def test_prefetch_factor_respects_constructor(self):
        dataset = _TinyDataset()
        model = _DummyModel()

        loader = dataset_util.PipelineDataLoader(
            dataset,
            model,
            model.gradient_accumulation_steps(),
            model,
            num_dataloader_workers=2,
            prefetch_batches_per_worker=1,
        )

        assert loader.dataloader.prefetch_factor == 1
