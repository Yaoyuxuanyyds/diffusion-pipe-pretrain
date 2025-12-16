import pytest
import torch

pytest.importorskip("torchvision")

from models.sd3_light import LightSD3Pipeline


def _build_pipe(gamma=0.0, target_layer=0):
    pipe = LightSD3Pipeline.__new__(LightSD3Pipeline)
    pipe.config = {}
    pipe.model_config = {"text_recon_gamma": gamma, "text_recon_target_layer": target_layer}
    return pipe


def test_text_recon_loss_combines_with_denoise_loss():
    pipe = _build_pipe(gamma=0.5, target_layer=0)
    loss_fn = pipe.get_loss_fn()

    pred = torch.tensor([1.0, 2.0])
    target = torch.zeros_like(pred)
    mask = torch.tensor([])

    final_tokens = torch.tensor([[1.0, 1.0]])
    target_tokens = torch.tensor([[0.0, 0.0]])

    loss = loss_fn((pred, final_tokens, target_tokens), (target, mask))

    base = torch.mean((pred - target) ** 2)
    text = torch.mean((final_tokens - target_tokens) ** 2)

    assert torch.allclose(loss, base + 0.5 * text)


def test_no_text_tokens_falls_back_to_base_loss():
    pipe = _build_pipe(gamma=0.75, target_layer=1)
    loss_fn = pipe.get_loss_fn()

    pred = torch.tensor([1.0])
    target = torch.tensor([0.0])
    mask = torch.tensor([])

    loss = loss_fn(pred, (target, mask))

    assert torch.allclose(loss, torch.mean((pred - target) ** 2))
