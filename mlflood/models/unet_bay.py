
import torch
import torch.nn as nn
import torch.nn.functional as F

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# """ Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

## """ Full assembly of the parts to form the complete network """

class UNet_bay(nn.Module):      
        
    def __init__(self, args, n_classes=1, bilinear=True, border_size=0, timestep = 1, use_diff_dem=False, predict_ahead = 5, ts_out = 0):
        super(UNet_bay, self).__init__()

        self.timestep = catchment_kwargs["timestep"]
        self.use_diff_dem = catchment_kwargs["use_diff_dem"]
        self.border_size = catchment_kwargs["border_size"]
        self.predict_ahead = catchment_kwargs["predict_ahead"]
        self.ts_out = catchment_kwargs["ts_out"]
        self.n_channels =  self.timestep*3  + self.predict_ahead + (4 if self.use_diff_dem else 0)   #dem, rainfall*timestep, wd*timestep 
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.args = args

        if args.task == "wd_ts":
            self.n_channels =  self.timestep*3  + self.predict_ahead + (4 if self.use_diff_dem else 0)   #dem, rainfall*timestep, wd*timestep 
        else:
            self.n_channels = 26
        
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.args = args
        

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)


        #bayesian
        self.m_outc1 = nn.Conv2d(64, 16, kernel_size=1)
        if self.ts_out:
            self.m_outc2 = nn.Conv2d(16, self.ts_out, kernel_size=1)
        else:
            self.m_outc2 = nn.Conv2d(16, 1, kernel_size=1)

        self.v_outc1 = nn.Conv2d(64, 16, kernel_size=1)
        
        if self.ts_out:
            self.v_outc2 = nn.Conv2d(16, self.ts_out, kernel_size=1)
        else:
            self.v_outc2 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, inputs):
        if type(inputs) is dict:
            x = inputs['data']
            mask = inputs['mask']
        else:
            x = inputs
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
    
        mu = F.relu(self.m_outc1(x))
        sigma = F.relu(self.v_outc1(x))

        mu = self.m_outc2(mu)
        #mu = torch.squeeze(mu, 0)


        sigma = self.v_outc2(sigma)
        ## clamping values otherwise encountering nan loss
        #sigma = torch.clamp(sigma, min=-40, max=40)
        sigma = torch.exp(sigma)
        ## Nico adds 1e-7 after exp to the variance to avoid nan errors
        sigma = sigma + 1e-7
        sigma = torch.squeeze(sigma, 0)

        return {"y_pred": mu, "sigma": sigma}
