"""Shared N-MNIST architecture and portable, weights-only checkpoints."""
from pathlib import Path

import torch
from torch import nn
import sinabs.layers as sl
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

INPUT_SHAPE = (2, 34, 34)
ARCHITECTURE = "nmnist_cnn_v1"


def build_model(model_type, batch_size=1):
    if model_type not in ("ann", "snn"):
        raise ValueError(f"Unknown model type: {model_type}")

    def activation():
        if model_type == "ann":
            return nn.ReLU()
        return sl.IAFSqueeze(
            batch_size=batch_size, min_v_mem=-1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )

    # Keep layer indices identical to the original 04 training scripts.
    return nn.Sequential(
        nn.Conv2d(2, 8, 3, padding=1, bias=False),
        activation(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(8, 16, 3, padding=1, bias=False),
        activation(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(16, 16, 3, padding=1, stride=2, bias=False),
        activation(),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10, bias=False),
        activation(),
    )


def save_checkpoint(model, path, model_type, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not save per-batch neuron state: inference uses a different batch size.
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.endswith(".v_mem")
    }
    torch.save({
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "model_type": model_type,
        "epoch": epoch,
        "state_dict": state,
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or (
        checkpoint.get("format_version") != 1
        or checkpoint.get("architecture") != ARCHITECTURE
    ):
        raise ValueError("Unsupported checkpoint; use --output from an updated 04 script")
    model_type = checkpoint.get("model_type")
    model = build_model(model_type, batch_size=1)
    state = checkpoint.get("state_dict")
    expected = model.state_dict()
    expected_keys = {key for key in expected if not key.endswith(".v_mem")}
    if not isinstance(state, dict) or set(state) != expected_keys:
        raise ValueError("Checkpoint parameters do not match the N-MNIST architecture")
    if any(not torch.is_tensor(value) or not torch.isfinite(value).all()
           for value in state.values()):
        raise ValueError("Checkpoint contains invalid or non-finite parameters")
    expected.update(state)
    model.load_state_dict(expected, strict=True)
    model.eval()
    return model, model_type
