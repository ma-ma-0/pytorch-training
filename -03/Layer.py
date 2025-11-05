import torch
from torch import nn

if __name__ == "__main__"
    my_tensor = torch.ones(16,3,256,256)
    print(f"out1 : {my_tensor.shape}")

    conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=0, stride=1)
    out2 = conv2(my_tensor)
    print(f"out2 : {out2.shape}")

    conv3 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1, stride=2)
    out3 = conv3(my_tensor)
    print(f"out3 : {out3.shape}")

    conv4 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1, stride=1)
    out4 = conv4(my_tensor)
    print(f"out4 : {out4.shape}")

    conv5 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, padding=2, stride=2)
    out5 = conv5(my_tensor)
    print(f"out5 : {out5.shape}")