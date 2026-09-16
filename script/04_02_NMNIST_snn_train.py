import argparse
from pathlib import Path

from nmnist_model import build_model, save_checkpoint

from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame

import sinabs.layers as sl

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import nn

from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data", help="path to the dataset")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("output/nmnist_snn.pt"),
                        help="checkpoint path (updated after each epoch)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch-size must be positive; num-workers must be nonnegative")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or a CUDA-enabled PyTorch environment")
    print(f"Training device: {device}")

    # Dataset
    n_time_steps = 100
    to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)

    snn_train_dataset = NMNIST(save_to=args.data, train=True, transform=to_raster)
    snn_test_dataset = NMNIST(save_to=args.data, train=False, transform=to_raster)

    # Train the model
    epochs = args.epochs
    lr = 1e-3
    batch_size = args.batch_size
    num_workers = args.num_workers

    snn_bptt = build_model("snn", batch_size=batch_size)

    # init the model weights
    for layer in snn_bptt.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(layer.weight.data)

    snn_train_dataloader = DataLoader(
        snn_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True,
    )
    snn_test_dataloader = DataLoader(
        snn_test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,
    )

    snn_bptt = snn_bptt.to(device=device)

    optimizer = SGD(params=snn_bptt.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    for e in range(epochs):

        # train
        snn_bptt.train()
        train_p_bar = tqdm(snn_train_dataloader)
        for data, label in train_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)
            # forward
            optimizer.zero_grad()
            # Each batch contains independent recordings, so clear neuron state.
            for layer in snn_bptt.modules():
                if isinstance(layer, sl.StatefulLayer):
                    layer.reset_states()
            output = snn_bptt(data)
            # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
            output = output.reshape(batch_size, n_time_steps, -1)
            # accumulate all time-steps output for final prediction
            output = output.sum(dim=1)
            loss = criterion(output, label)
            # backward
            loss.backward()
            optimizer.step()

            # set progressing bar
            train_p_bar.set_description(
                f"Epoch {e} - BPTT Training Loss: {round(loss.item(), 4)}"
            )

        # validate
        snn_bptt.eval()
        correct_predictions = []
        with torch.no_grad():
            test_p_bar = tqdm(snn_test_dataloader)
            for data, label in test_p_bar:
                # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
                data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
                label = label.to(dtype=torch.long, device=device)
                # forward
                # Each batch contains independent recordings, so clear neuron state.
                for layer in snn_bptt.modules():
                    if isinstance(layer, sl.StatefulLayer):
                        layer.reset_states()
                output = snn_bptt(data)
                # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
                output = output.reshape(batch_size, n_time_steps, -1)
                # accumulate all time-steps output for final prediction
                output = output.sum(dim=1)
                # calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                # compute the total correct predictions
                correct_predictions.append(pred.eq(label.view_as(pred)))
                # set progressing bar
                test_p_bar.set_description(f"Epoch {e} - BPTT Testing Model...")

            correct_predictions = torch.cat(correct_predictions)
            print(
                f"Epoch {e} - BPTT accuracy: {correct_predictions.sum().item()/(len(correct_predictions))*100}%"
            )

        save_checkpoint(snn_bptt, args.output, "snn", epoch=e + 1)
        print(f"Checkpoint saved: {args.output}")


if __name__ == "__main__":
    main()
