from torch.nn import Module
import torch

class Baseline(Module):
    def __init__(self, border_size=0, normalize_output=False, use_diff_dem=True):
        super(Baseline, self).__init__()
        self.border_size = border_size
        self.normalize_output = normalize_output
        self.use_diff_dem = use_diff_dem

    def forward(self, x, mask, *args):
        batch, p, nx, ny = x.shape
        b = self.border_size
        if self.normalize_output:
            out =  torch.zeros([batch, 1, nx, ny])
        else:
            d = 4 if self.use_diff_dem else 0
            timestep = (p-d) // 3
            dim = p - timestep
            out = x[:,dim:dim+1]

        out = torch.cat([out, mask], 1)
        if b:
            out = out[:,:,b:-b,b:-b]
        return out

            