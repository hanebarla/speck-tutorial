import torch
import torch.nn as nn

from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork


DEVICE = "speck2fdevkit:0"


def build_ann():
    # 28x28, 1 channel入力の簡単なCNN
    # 注意: Speckに載せることを優先し、bias=Falseにする
    return nn.Sequential(
        nn.Conv2d(1, 20, kernel_size=5, stride=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),

        nn.Conv2d(20, 32, kernel_size=5, stride=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),

        nn.Conv2d(32, 128, kernel_size=3, stride=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),

        nn.Flatten(),
        nn.Linear(128, 500, bias=False),
        nn.ReLU(),
        nn.Linear(500, 10, bias=False),
    )


def main():
    ann = build_ann()

    # PyTorch ANNをSinabs SNNへ変換
    sinabs_model = from_model(
        ann,
        batch_size=1,
        add_spiking_output=True,
    )

    # Speck/DYNAP-CNN用ネットワークへ変換
    hw_model = DynapcnnNetwork(
        snn=sinabs_model.spiking_model,
        input_shape=(1, 28, 28),
        discretize=True,
        dvs_input=False,
    )

    # Speckにデプロイ
    hw_model.to(device=DEVICE, chip_layers_ordering="auto")

    print("Deployment succeeded.")
    print(hw_model)


if __name__ == "__main__":
    main()