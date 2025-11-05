import torch
from torch import nn

if __name__ == "__main__"
    my_tensor = torch.ones(16,3,256,256)
    print(f"original : {my_tensor.shape}")

    conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=0, stride=1)
    out1 = conv(input_tensor)
    print(f"out1 : {out1.shape}")