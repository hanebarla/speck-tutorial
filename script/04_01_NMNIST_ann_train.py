import argparse

from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame, ToEvent

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import nn

from tqdm.notebook import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data", help="path to the dataset")

args = parser.parse_args()

# load the NMNIST dataset
to_frame = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bin=1)
train_dataset = NMNIST(args.data, train=True, transform=to_frame)
test_dataset = NMNIST(args.data, train=False, transform=to_frame)

## Check the shape of the transformed array
sample_data, label = train_dataset[0]
print(
    f"The transformed array is in shape [Time-Step, Channel, Height, Width] --> {sample_data.shape}"
)

# define a CNN model
cnn = nn.Sequential(
    # [2, 34, 34] -> [8, 17, 17] (イベントデータは正と負のイベントの2チャンネルで構成されるため、入力チャネル数は2)
    nn.Conv2d(
        in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False
    ),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    # [8, 17, 17] -> [16, 8, 8]
    nn.Conv2d(
        in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False
    ),
    nn.ReLU(),
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
    nn.ReLU(),
    # [16 * 4 * 4] -> [10]
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10, bias=False),
    nn.ReLU(),
)

# init the model weights
for layer in cnn.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)
        
# Train the model
epochs = 10
lr = 1e-3
batch_size = 4
num_workers = 4
device = "cuda:0"
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
    drop_last=True,
    shuffle=shuffle,
)

optimizer = SGD(params=cnn.parameters(), lr=lr)
criterion = CrossEntropyLoss()

for e in range(epochs):
    # train
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