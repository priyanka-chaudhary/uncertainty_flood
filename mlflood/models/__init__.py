import torch
from models import Baseline
from models import CNNrolling
from models import unet
from models import TCN
from models import utae
from models import unet3d
from models import unet_bay

def get_model(args, catchment_kwargs):

    ############### setup network ###################
    if args.model == 'baseline':
        return Baseline.Baseline(args)
    if args.model == 'cnn':
        return CNNrolling.CNNrolling(args, catchment_kwargs)
    if args.model == 'unet' and args.if_bayesian == False:
        return unet.UNet(args, catchment_kwargs)
    if args.model == 'tcn':
        return TCN.TemporalConvNet(args)
    if args.model == 'utae':
        return utae.UTAE(args)
    if args.model == 'unet3d':
        return unet3d.UNet3D(args)
    if args.model == 'unet' and args.if_bayesian == True:
        torch.manual_seed(args.seed)
        return unet_bay.UNet_bay(args, catchment_kwargs)
    else:
        raise NotImplementedError

