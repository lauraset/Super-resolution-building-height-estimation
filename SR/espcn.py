"""
Code is from https://github.com/leftthomas/ESPCN
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class ESPC(nn.Module):
    def __init__(self, input_channel=4, upscale_factor=4):
        super(ESPC, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x, image_name=None):
        x = F.tanh(self.conv1(x))
        # np.save(f"./middle_feature/{image_name[0]}", x[0].detach().cpu().numpy())
        x = F.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

if __name__ == "__main__":
    model = ESPC(input_channel=4, upscale_factor=4).cuda()
    x = torch.randn((1, 4, 64, 64)).cuda()
    out = model(x)
    print(out.shape)
    # print(model)
    nparas = sum([p.numel() for p in model.parameters()])
    print('nparams: %.2f M'%(nparas/1e+6))
    # 0.04 M