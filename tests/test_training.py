"""Offline smoke tests: python -m unittest discover -s tests -v."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import sinabs.layers as sl
from tonic.transforms import ToFrame
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "script" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticNMNIST(torch.utils.data.Dataset):
    sensor_size = (34, 34, 2)

    def __init__(self, save_to, train, transform):
        self.size = 4 if train else 3
        self.transform = transform
        assert isinstance(transform, ToFrame)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        events = np.zeros(
            8000, dtype=[("x", "i2"), ("y", "i2"), ("p", "i1"), ("t", "i8")]
        )
        events["x"] = rng.integers(0, 34, len(events))
        events["y"] = rng.integers(0, 34, len(events))
        events["p"] = rng.integers(0, 2, len(events))
        events["t"] = np.arange(len(events)) * 10
        return self.transform(events), index % 10


class TrainingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def run_training(self, filename):
        module = load_script(filename)
        steps = []
        resets = []
        test_case = self
        original_reset = sl.StatefulLayer.reset_states

        class CheckedSGD(torch.optim.SGD):
            def step(self, closure=None):
                gradients = [
                    p.grad for group in self.param_groups
                    for p in group["params"] if p.grad is not None
                ]
                test_case.assertTrue(gradients)
                test_case.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                steps.append(1)
                return super().step(closure)

        def checked_reset(layer, *args, **kwargs):
            original_reset(layer, *args, **kwargs)
            for _, state in layer.named_buffers():
                test_case.assertEqual(torch.count_nonzero(state).item(), 0)
                test_case.assertIsNone(state.grad_fn)
            resets.append(1)

        torch.manual_seed(0)
        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as checkpoint_dir,
            patch.object(module, "NMNIST", SyntheticNMNIST),
            patch.object(module, "SGD", CheckedSGD),
            patch.object(module, "tqdm", lambda data: tqdm(data, disable=True)),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(sl.StatefulLayer, "reset_states", checked_reset),
            patch("sys.argv", [filename, "--epochs", "2", "--batch-size", "2",
                               "--num-workers", "0", "--output",
                               str(Path(checkpoint_dir) / "model.pt")]),
            contextlib.redirect_stdout(output),
        ):
            module.main()
            from nmnist_model import load_checkpoint
            restored, model_type = load_checkpoint(Path(checkpoint_dir) / "model.pt")
            self.assertEqual(model_type, "snn" if "snn" in filename else "ann")
            self.assertFalse(restored.training)
        self.assertIn("Training device: cpu", output.getvalue())
        self.assertEqual(output.getvalue().count("accuracy:"), 2)
        self.assertEqual(len(steps), 4)
        if "snn" in filename:
            # Four stateful layers, two train batches and one test batch per epoch.
            self.assertGreaterEqual(len(resets), 4 * (2 + 1) * 2)

    def test_ann_cpu_training_and_evaluation(self):
        self.run_training("04_01_NMNIST_ann_train.py")

    def test_snn_cpu_training_and_evaluation(self):
        self.run_training("04_02_NMNIST_snn_train.py")

    def test_invalid_arguments_fail_before_dataset_download(self):
        for filename in ("04_01_NMNIST_ann_train.py", "04_02_NMNIST_snn_train.py"):
            for arguments in (["--device", "cuda:0"], ["--epochs", "0"],
                              ["--batch-size", "0"], ["--num-workers", "-1"]):
                with self.subTest(script=filename, arguments=arguments):
                    module = load_script(filename)
                    with (
                        patch.object(module, "NMNIST") as dataset,
                        patch.object(torch.cuda, "is_available", return_value=False),
                        patch("sys.argv", [filename, *arguments]),
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as error,
                    ):
                        module.main()
                    self.assertEqual(error.exception.code, 2)
                    dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
