"""Offline checks using real Sinabs/Samna conversion; no Speck connection."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import samna
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
from nmnist_model import build_model, load_checkpoint, save_checkpoint

spec = importlib.util.spec_from_file_location(
    "speck_inference", ROOT / "script" / "05_NMNIST_speck_inference.py"
)
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


def spike(core, digit):
    event = samna.speck2f.event.Spike()
    event.layer = core
    event.feature = digit
    return event


def sample_events():
    return np.array([(2, 3, 1, 120), (4, 5, 0, 100)],
                    dtype=[("x", "i2"), ("y", "i2"), ("p", "i1"), ("t", "i8")])


class InferenceTests(unittest.TestCase):
    def test_sample_selection_covers_remaining_dataset(self):
        self.assertEqual(inference.sample_indices(10, 0, 3), [0, 4, 9])
        self.assertEqual(inference.sample_indices(10, 7, 0), [7, 8, 9])
        self.assertEqual(inference.sample_indices(10, 7, 100), [7, 8, 9])
        self.assertEqual(inference.sample_indices(10, 4, 1), [4])
        with self.assertRaises(ValueError):
            inference.sample_indices(10, 10, 1)

    def test_checkpoint_roundtrip_and_offline_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model_type in ("ann", "snn"):
                with self.subTest(model_type=model_type):
                    original = build_model(model_type, batch_size=2)
                    if model_type == "snn":
                        # Initialized training state must not constrain inference batch size.
                        with torch.no_grad():
                            original(torch.zeros(2, 2, 34, 34))
                    path = Path(tmp) / f"{model_type}.pt"
                    save_checkpoint(original, path, model_type, epoch=1)
                    loaded, restored_type = load_checkpoint(path)
                    self.assertEqual(restored_type, model_type)
                    for name, value in original.named_parameters():
                        torch.testing.assert_close(dict(loaded.named_parameters())[name], value)
                    for name, value in loaded.named_buffers():
                        if name.endswith("v_mem"):
                            self.assertEqual(value.numel(), 0)
                    network = inference.build_hardware_network(path)
                    with patch.object(samna.device, "open_device") as open_device:
                        config = network.make_config(
                            device="speck2fdevkit:0", monitor_layers=[-1],
                            config_modifier=inference.disable_camera,
                        )
                        open_device.assert_not_called()
                    self.assertEqual(len(network.layer2core_map), 4)
                    self.assertFalse(config.dvs_layer.pass_sensor_events)
                    self.assertFalse(config.dvs_layer.monitor_enable)

    def test_dry_run_never_downloads_or_opens_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ann.pt"
            save_checkpoint(build_model("ann"), path, "ann", epoch=1)
            with (
                patch("sys.argv", ["05", "--checkpoint", str(path), "--dry-run"]),
                patch.object(inference, "NMNIST") as dataset,
                patch.object(samna.device, "open_device") as open_device,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                inference.main()
            dataset.assert_not_called()
            open_device.assert_not_called()
            self.assertIn("Dry run OK", output.getvalue())

    def test_bad_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.pt"
            torch.save({"state_dict": {}}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)
            save_checkpoint(build_model("ann"), path, "ann", epoch=1)
            state = torch.load(path, weights_only=True)
            state["state_dict"].pop("0.weight")
            torch.save(state, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_events_are_sorted_rebased_and_routed(self):
        factory = inference.ChipFactory("speck2fdevkit:0")
        events = sample_events()
        result = inference.to_spikes(events, factory, input_core=5)
        self.assertEqual([event.timestamp for event in result], [0, 20])
        self.assertEqual([event.layer for event in result], [5, 5])
        self.assertEqual([(event.x, event.y, event.feature) for event in result],
                         [(4, 5, 0), (2, 3, 1)])
        self.assertEqual(events["t"].tolist(), [120, 100])
        self.assertEqual(inference.to_spikes(events[:0], factory, 5), [])
        events["x"][0] = 34
        with self.assertRaises(ValueError):
            inference.to_spikes(events, factory, 5)

    def test_prediction_filters_output_core_and_handles_abstentions(self):
        event_type = samna.speck2f.event.Spike
        prediction, counts, status = inference.decode_prediction(
            [spike(7, 3), spike(7, 3), spike(7, 4), spike(2, 8), object()],
            7, event_type,
        )
        self.assertEqual((prediction, counts.sum(), status), (3, 3, "ok"))
        self.assertEqual(inference.decode_prediction([], 7, event_type)[0], -1)
        self.assertEqual(inference.decode_prediction(
            [spike(7, 0), spike(7, 1)], 7, event_type
        )[2], "tie")

    def test_evaluation_resets_and_counts_empty_sample_as_incorrect(self):
        network = Mock()
        network.layer2core_map = {0: 5, 1: 7}
        network.exit_layer_ids = [1]
        network.return_value = [spike(7, 3)]
        dataset = [(sample_events(), 3), (sample_events()[:0], 0)]
        output = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            result = inference.evaluate(
                network, dataset, range(2), inference.ChipFactory("speck2fdevkit:0"),
                csv.writer(output), output,
            )
        self.assertEqual(result, (1, 2, 1))
        self.assertEqual(network.reset_states.call_count, 2)
        network.assert_called_once()
        self.assertTrue(all(event.layer == 5 for event in network.call_args.args[0]))
        self.assertEqual(len(output.getvalue().splitlines()), 2)

    def test_failed_deployment_still_closes_device_and_graphs(self):
        network = Mock()
        network.to.side_effect = RuntimeError("deployment failed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ann.pt"
            save_checkpoint(build_model("ann"), path, "ann", epoch=1)
            with (
                patch("sys.argv", ["05", "--checkpoint", str(path)]),
                patch.object(inference, "build_hardware_network", return_value=network),
                patch.object(inference, "NMNIST", return_value=[(sample_events(), 3)]),
                patch.object(samna.device, "close_device") as close_device,
                self.assertRaisesRegex(RuntimeError, "deployment failed"),
            ):
                inference.main()
            network.device_input_graph.stop.assert_called_once()
            network.device_output_graph.stop.assert_called_once()
            close_device.assert_called_once_with(network.samna_device)


if __name__ == "__main__":
    unittest.main()
