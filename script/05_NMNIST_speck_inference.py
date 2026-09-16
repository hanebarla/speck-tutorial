"""Replay N-MNIST test events through a trained network on Speck."""
import argparse
import csv
from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np
import samna
from tonic.datasets.nmnist import NMNIST
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory

from nmnist_model import INPUT_SHAPE, load_checkpoint


def build_hardware_network(checkpoint_path):
    model, model_type = load_checkpoint(checkpoint_path)
    if model_type == "ann":
        model = from_model(
            model, input_shape=INPUT_SHAPE, batch_size=1,
            add_spiking_output=False,
        ).spiking_model
    return DynapcnnNetwork(
        snn=model.cpu().eval(), input_shape=INPUT_SHAPE,
        discretize=True, dvs_input=False,
    )


def disable_camera(config):
    # Only replayed N-MNIST spikes should enter the CNN cores.
    config.dvs_layer.pass_sensor_events = False
    config.dvs_layer.monitor_enable = False
    config.dvs_layer.raw_monitor_enable = False
    return config


def to_spikes(events, factory, input_core):
    if events.dtype.names is None or not {"x", "y", "p", "t"} <= set(events.dtype.names):
        raise ValueError("Expected structured N-MNIST events with x, y, p, t fields")
    if len(events) == 0:
        return []
    if (
        np.any((events["x"] < 0) | (events["x"] >= 34))
        or np.any((events["y"] < 0) | (events["y"] >= 34))
        or np.any((events["p"] < 0) | (events["p"] > 1))
        or np.any(events["t"] < 0)
    ):
        raise ValueError("N-MNIST event coordinates, polarity or timestamps are invalid")
    ordered = events[np.argsort(events["t"], kind="stable")]
    if int(ordered["t"][-1]) - int(ordered["t"][0]) >= 2**32:
        raise ValueError("Sample duration exceeds the 32-bit timestamp range")
    return factory.xytp_to_events(ordered, layer=input_core, reset_timestamps=True)


def decode_prediction(events, output_core, spike_type):
    counts = np.zeros(10, dtype=np.int64)
    for event in events:
        if (isinstance(event, spike_type) and event.layer == output_core
                and 0 <= event.feature < 10):
            counts[event.feature] += 1
    if not counts.any():
        return -1, counts, "no_spikes"
    winners = np.flatnonzero(counts == counts.max())
    if len(winners) != 1:
        return -1, counts, "tie"
    return int(winners[0]), counts, "ok"


def close_hardware(network):
    # Also handle a partially failed deployment without masking the original error.
    for name in ("device_input_graph", "device_output_graph"):
        graph = getattr(network, name, None)
        if graph is not None:
            try:
                graph.stop()
            except Exception as exc:
                print(f"Warning: could not stop {name}: {exc}", file=sys.stderr)
    device = getattr(network, "samna_device", None)
    if device is not None:
        try:
            device.get_stop_watch().stop()
        except Exception as exc:
            print(f"Warning: could not stop device timestamps: {exc}", file=sys.stderr)
        try:
            samna.device.close_device(device)
        except Exception as exc:
            print(f"Warning: could not close device: {exc}", file=sys.stderr)


def evaluate(network, dataset, indices, factory, writer=None, output_file=None):
    # Logical layer IDs are NOT necessarily physical core IDs.
    input_core = network.layer2core_map[0]
    output_core = network.layer2core_map[network.exit_layer_ids[0]]
    spike_type = factory.get_config_builder().get_samna_module().event.Spike
    correct = rejected = total = 0
    for index in indices:
        events, label = dataset[index]
        spikes = to_spikes(events, factory, input_core)
        network.reset_states()
        # Sinabs' hardware forward requires a non-empty list (uses max(timestamp)).
        output = network(spikes) if spikes else []
        prediction, counts, status = decode_prediction(output, output_core, spike_type)
        correct += int(prediction == int(label))
        rejected += int(prediction == -1)
        total += 1
        print(f"sample={index} label={label} prediction={prediction} "
              f"output_spikes={int(counts.sum())} status={status}", flush=True)
        if writer is not None:
            writer.writerow([index, int(label), prediction, status, int(prediction == label),
                             *counts.tolist()])
            output_file.flush()
    print(f"Speck accuracy: {correct}/{total} ({100 * correct / total:.2f}%) "
          f"undecided={rejected}", flush=True)
    return correct, total, rejected


def sample_indices(dataset_size, start_index, num_samples):
    if not 0 <= start_index < dataset_size or num_samples < 0:
        raise ValueError("Invalid dataset size, start index or sample count")
    remaining = dataset_size - start_index
    count = remaining if num_samples == 0 else min(num_samples, remaining)
    # Spread a small evaluation across the dataset, which may be class-ordered.
    return np.linspace(start_index, dataset_size - 1, num=count, dtype=int).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint from 04")
    parser.add_argument("--data", default="data", help="N-MNIST dataset directory")
    parser.add_argument("--device", default="speck2fdevkit:0", help="Speck device identifier")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="evenly spaced test samples; 0 means all remaining samples")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional CSV path; must not already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate mapping without opening a device or loading data")
    args = parser.parse_args()
    if args.num_samples < 0 or args.start_index < 0:
        parser.error("--num-samples and --start-index must be nonnegative")
    if not args.device.startswith("speck"):
        parser.error("--device must identify a Speck device, e.g. speck2fdevkit:0")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}; train with an updated 04 script")
    if args.output is not None and args.output.exists() and not args.dry_run:
        parser.error(f"Output already exists: {args.output}; choose another CSV path")

    factory = ChipFactory(args.device)
    network = build_hardware_network(args.checkpoint)
    options = dict(device=args.device, layer2core_map="auto",
                   monitor_layers=[-1], config_modifier=disable_camera)
    if args.dry_run:
        network.make_config(**options)
        print(f"Dry run OK: layer -> core = {network.layer2core_map}")
        return

    dataset = NMNIST(save_to=args.data, train=False)
    if args.start_index >= len(dataset):
        parser.error(f"--start-index must be less than the dataset size ({len(dataset)})")
    indices = sample_indices(len(dataset), args.start_index, args.num_samples)

    with ExitStack() as stack:
        output_file = writer = None
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_file = stack.enter_context(args.output.open("x", newline="", encoding="utf-8"))
            writer = csv.writer(output_file)
            writer.writerow(["index", "label", "prediction", "status", "correct",
                             *[f"spikes_{digit}" for digit in range(10)]])
        try:
            network.to(**options)
            print(f"Deployed: layer -> core = {network.layer2core_map}", flush=True)
            evaluate(network, dataset, indices, factory, writer, output_file)
        finally:
            close_hardware(network)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInference interrupted; device closed.", file=sys.stderr)
        raise SystemExit(130)
