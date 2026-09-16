import argparse

from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame, ToEvent

import sinabs.layers as sl
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import nn

from tqdm.notebook import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data", help="path to the dataset")
args = parser.parse_args()

# Dataset
n_time_steps = 100
to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)

snn_train_dataset = NMNIST(save_to=args.data, train=True, transform=to_raster)
snn_test_dataset = NMNIST(save_to=args.data, train=False, transform=to_raster)

# Train the model
epochs = 10
lr = 1e-3
batch_size = 4
num_workers = 4
device = "cuda:0"
shuffle = True

# just replace the ReLU layer with the sl.IAFSqueeze
snn_bptt = nn.Sequential(
    # [2, 34, 34] -> [8, 17, 17]
    nn.Conv2d(
        in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False
    ),
    sl.IAFSqueeze(
        batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()
    ),
    nn.AvgPool2d(2, 2),
    # [8, 17, 17] -> [16, 8, 8]
    nn.Conv2d(
        in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False
    ),
    sl.IAFSqueeze(
        batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()
    ),
    nn.AvgPool2d(2, 2),
    # [16 * 8 * 8] -> [16, 4, 4]
    nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(3, 3),
        padding=(1, 1),
        stride=(2, 2),
        bias=False,
    ),
    sl.IAFSqueeze(
        batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()
    ),
    # [16 * 4 * 4] -> [10]
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10, bias=False),
    sl.IAFSqueeze(
        batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()
    ),
)

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
    train_p_bar = tqdm(snn_train_dataloader)
    for data, label in train_p_bar:
        # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
        data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
        label = label.to(dtype=torch.long, device=device)
        # forward
        optimizer.zero_grad()
        output = snn_bptt(data)
        # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
        output = output.reshape(batch_size, n_time_steps, -1)
        # accumulate all time-steps output for final prediction
        output = output.sum(dim=1)
        loss = criterion(output, label)
        # backward
        loss.backward()
        optimizer.step()

        # detach the neuron states and activations from current computation graph(necessary)
        for layer in snn_bptt.modules():
            if isinstance(layer, sl.StatefulLayer):
                for name, buffer in layer.named_buffers():
                    buffer.detach_()

        # set progressing bar
        train_p_bar.set_description(
            f"Epoch {e} - BPTT Training Loss: {round(loss.item(), 4)}"
        )

    # validate
    correct_predictions = []
    with torch.no_grad():
        test_p_bar = tqdm(snn_test_dataloader)
        for data, label in test_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)
            # forward
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