import argparse
from pathlib import Path

from nmnist_model import build_model, save_checkpoint

from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame

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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("output/nmnist_ann.pt"),
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

    # load the NMNIST dataset
    to_frame = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=1)
    train_dataset = NMNIST(args.data, train=True, transform=to_frame)
    test_dataset = NMNIST(args.data, train=False, transform=to_frame)

    ## Check the shape of the transformed array
    sample_data, label = train_dataset[0]
    print(
        f"The transformed array is in shape [Time-Step, Channel, Height, Width] --> {sample_data.shape}"
    )

    cnn = build_model("ann")

    # init the model weights
    for layer in cnn.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(layer.weight.data)

    # Train the model
    epochs = args.epochs
    lr = 1e-3
    batch_size = args.batch_size
    num_workers = args.num_workers
    shuffle = True

    cnn = cnn.to(device=device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    optimizer = SGD(params=cnn.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    for e in range(epochs):
        # train
        cnn.train()
        train_p_bar = tqdm(train_dataloader)
        for data, label in train_p_bar:
            # remove the time-step axis since we are training CNN
            # move the data to accelerator
            data = data.squeeze(dim=1).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)
            # forward
            optimizer.zero_grad()
            output = cnn(data)
            loss = criterion(output, label)
            # backward
            loss.backward()
            optimizer.step()
            # set progressing bar
            train_p_bar.set_description(
                f"Epoch {e} - Training Loss: {round(loss.item(), 4)}"
            )

        # validate
        cnn.eval()
        correct_predictions = []
        with torch.no_grad():
            test_p_bar = tqdm(test_dataloader)
            for data, label in test_p_bar:
                # remove the time-step axis since we are training CNN
                # move the data to accelerator
                data = data.squeeze(dim=1).to(dtype=torch.float, device=device)
                label = label.to(dtype=torch.long, device=device)
                # forward
                output = cnn(data)
                # calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                # compute the total correct predictions
                correct_predictions.append(pred.eq(label.view_as(pred)))
                # set progressing bar
                test_p_bar.set_description(f"Epoch {e} - Testing Model...")

            correct_predictions = torch.cat(correct_predictions)
            print(
                f"Epoch {e} - accuracy: {correct_predictions.sum().item()/(len(correct_predictions))*100}%"
            )

        save_checkpoint(cnn, args.output, "ann", epoch=e + 1)
        print(f"Checkpoint saved: {args.output}")


if __name__ == "__main__":
    main()
