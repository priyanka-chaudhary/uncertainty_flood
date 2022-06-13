import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNrolling(nn.Module):
    def __init__(self, args, catchment_kwargs, k = 64, border_size=0):
        super(CNNrolling, self).__init__()
        
        self.timestep = catchment_kwargs["timestep"]
        self.use_diff_dem = catchment_kwargs["use_diff_dem"]
        self.predict_ahead = catchment_kwargs["predict_ahead"]
        self.input_channels = 27 #self.timestep *3  + self.predict_ahead + (4 if self.use_diff_dem else 0)   #dem, rainfall*timestep, wd*timestep

        
        
        self.conv1 = nn.Conv2d(self.input_channels, k, 5, padding=(2,2))
        self.conv2 = nn.Conv2d(k, k, 5, padding=(2,2))  
        self.conv3 = nn.Conv2d(k, 1, 5, padding=(2, 2))
        self.border_size = border_size

    def forward(self, inputs):
        if type(inputs) is dict:
            x = inputs['data']
            mask = inputs['mask']
        else:
            x = inputs
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)
        if self.border_size:
            x3 = x3[:,:,self.border_size:-self.border_size, self.border_size:-self.border_size]
            mask = mask[:,:,self.border_size:-self.border_size, self.border_size:-self.border_size]
        # x3 = torch.squeeze(x3, 0)
        #out = torch.cat([x3, mask], 1)
        
        return {"y_pred": x3}