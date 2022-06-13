import sys, argparse, os
import numpy as np
import random
from datetime import datetime as dt

import torch
import torch.nn.functional as F

import loss as L
import flag

from torch.utils.tensorboard import SummaryWriter
from utils.other import new_log
from utils.other import write_run_time
#from models import get_model
from models import  get_model
from dataloading import get_dataloaders
from utils.plot_utils import *

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description="training script")

#### general parameters #####################################################
parser.add_argument('--tag', default="max_wo_bay",type=str)
#parser.add_argument('--tag', default="__",type=str)
parser.add_argument("--device",default="cuda",type=str,choices=["cuda", "cpu"])
parser.add_argument("--save-dir", default="/scratch2/flood_sim/branch/save_dir/",help="Path to directory where models and logs should be saved saved !! this folder must already exist !!")
parser.add_argument("--runtime-dir", default="/scratch2/flood_sim/branch/save_dir/time/",help="Path to directory where runtime for the model is saved !! this folder must already exist !!")
parser.add_argument("--logstep-train", default=10,type=int,help="iterations step for training log")
parser.add_argument("--save-model", default="best",choices=['last','best','No'],help="which model to save")
parser.add_argument("--val-every-n-epochs", type=int, default=1,help="interval of training epochs to run the validation")
parser.add_argument('--resume', type=str, default=None,help='path to resume the model if needed')
parser.add_argument('--mode',default="only_train",type=str,choices=["None","train","test","train_and_test", "only_train"],help="mode to be run")

#### data parameters ##########################################################
parser.add_argument("--data",default="multi",type=str,choices=["toy", "709", "684", "multi"],help="dataset selection")
parser.add_argument("--task",default="max_depth",type=str,choices=["wd_ts", "max_depth"],help="select b/w task predicting water depth for next timesteps (wd_ts) or max depth for rainfall events")
#parser.add_argument("--timestep",default="one_ts",type=str,choices=["one_ts","two_ts"],help="select one or two timestep")
parser.add_argument("--timestep",default=1,type=int,choices=[1,2],help="select one or two timestep")
parser.add_argument("--datafolder",type=str,help="root directory of the dataset")
parser.add_argument("--workers", type=int, default=0,metavar="N",help="dataloader threads")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--normalize-patch",action='store_true',default=False,help="normalize patches")
parser.add_argument("--sample-type",default="single",type=str,choices=["single", "full"],help="select whether a sample is a single time step or the full event")
parser.add_argument('--predict_ahead', type=int, default=5)

#### optimizer parameters #####################################################
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--optimizer',default='adam', choices=['sgd','adam'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
#parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--w-decay', type=float, default=0)#1e-5)
parser.add_argument('--lr-scheduler', type=str, default='multi',choices=['no','step', 'smart', 'multi', 'cyclic'],help='lr scheduler mode')
parser.add_argument('--lr-step', type=int, default=50,help=' number of epochs between decreasing steps applies to lr-scheduler in [step, exp]')
parser.add_argument('--lr-gamma', type=float, default=0.1,help='decrease rate')

#### model parameters #####################################################
parser.add_argument("--model", default='unet', type=str,help="model to run: 'UResNet', 'tc_net', 'unet', 'unet_new', 'unet2', 'tc_net3', 'resnet', 'unet_ts_out', 'segnet")
parser.add_argument("--loss", default="l1_3", type=str, help="loss ['GNLL','MSE', 'L1', 'nll', 'gnll', 'gnll_m', 'stef_lnll', 'lnll', 'lnll2', 'lnll3', 'l1_1'] ")

#### bayesian training parameters #########################################
parser.add_argument("--if_bayesian",default=False,type=bool,choices=[True,False],help="training using bayesian predictive uncertainty")
parser.add_argument('--num_models', type=int, default=5)
parser.add_argument("--model_number",default="model1",type=str,help="ensemble model number training")
parser.add_argument('--seed', type=int, default=27,help="model1: 19, model2: 27, model3: 37, model4: 45, model5: 73")

## saving model on l1 loss if bayesian
parser.add_argument('--save_best_l1',default=False,type=bool,choices=[True,False],help="saving model based on best l1 loss")
## laplace prior
parser.add_argument('--l1_prior',default=False,type=bool,choices=[True,False],help="saving model based on best l1 loss")

## catchment
parser.add_argument("--fix_indexes", default=False, const=True, nargs='?', type=str2bool,help="select whether patches are generated sequentially (true) or randomly")
parser.add_argument("--tau", default=0.5,type=float,help="fraction of non-zero elements to accept a patch for training")
parser.add_argument("--dim_patch", default=256,type=int,help="patch dimensions")
parser.add_argument("--num_patch", default=10,type=int,help="number of patches to generate from a timestep")
parser.add_argument("--ts_out", default=3,type=int,help="number of timesteps to generate from model as output")
parser.add_argument("--rain_c", default=12,type=int,help="number of rainfall channels for input")

### fix indexes validation
parser.add_argument("--fix_indexes_val", default=False, const=True, nargs='?', type=str2bool,help="select whether patches are generated sequentially (true) or randomly")
### change value of tau during training
parser.add_argument("--change_tau",default=False,type=bool,choices=[True,False],help="training with different tau values during training")
## prior scale value
parser.add_argument("--prior_sc", default=0.05,type=float,help="fraction of non-zero elements to accept a patch for training")
parser.add_argument('--exp', type=str, default='exp4',choices=['exp1','exp2', 'exp3', 'exp4'],help='select which experiment to run')



class DevelopingSuite(object):

    def __init__(self, args):

        self.args = args
        
        self.dataloaders, self.data_stats =  get_dataloaders(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu")
        self.model = get_model(args,self.data_stats)
        self.i = 0
        self.model.to(self.device)
        if args.resume is not None:
            self.resume(path=args.resume)

        if "train" in args.mode:
            self.experiment_folder = new_log(args.save_dir,args.model + "_" + args.tag,args=args)
            self.writer = SummaryWriter(log_dir=self.experiment_folder)

            if args.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr,weight_decay=args.w_decay)
            elif args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=self.args.momentum,weight_decay=args.w_decay)

            if args.lr_scheduler == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
            elif args.lr_scheduler == 'smart':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=args.lr_step, factor=args.lr_gamma)
            elif args.lr_scheduler == 'multi':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,200,400], gamma=args.lr_gamma)
            elif args.lr_scheduler == 'cyclic':
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.00001, step_size_up=50, cycle_momentum=False)
            else:
                self.scheduler = None
            
        self.epoch = 0
        self.iter = 0

        self.val_stats = {}
        self.val_stats["best_optimization_loss"] = np.nan
        self.val_stats["optimization_loss"] = np.nan
        ##save best l1
        if args.if_bayesian == True and args.save_best_l1 == True:
            self.val_stats["best_l1_loss"] = np.nan

    def train_and_eval(self):

        if args.l1_prior == True:
            def create_weight(conv_layer, dist):
                t = dist.sample((conv_layer.weight.view(-1).size())).reshape(conv_layer.weight.size())
                with torch.no_grad():
                    conv_layer.weight.add_( t.to(self.device))

            laplacian_dist = torch.distributions.laplace.Laplace(loc=torch.tensor([0.]), scale=torch.tensor([args.prior_sc]))
            normal_dist = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.0]))

            conv_layers =[
                  self.model._modules['inc']._modules['double_conv']._modules['0'],
                  self.model._modules['inc']._modules['double_conv']._modules['3'],
                  self.model._modules['down1']._modules['maxpool_conv']._modules['1']._modules['double_conv']._modules['0'],
                  self.model._modules['down1']._modules['maxpool_conv']._modules['1']._modules['double_conv']._modules['3'],
                  self.model._modules['down2']._modules['maxpool_conv']._modules['1']._modules['double_conv']._modules['0'],
                  self.model._modules['down2']._modules['maxpool_conv']._modules['1']._modules['double_conv']._modules['3'],
                  self.model._modules['down3']._modules['maxpool_conv']._modules['1']._modules['double_conv']._modules['0'],
                  self.model._modules['down3']._modules['maxpool_conv']._modules['1']._modules['double_conv']._modules['3'],
                  self.model._modules['down4']._modules['maxpool_conv']._modules['1']._modules['double_conv']._modules['0'],
                  self.model._modules['down4']._modules['maxpool_conv']._modules['1']._modules['double_conv']._modules['3'],
                  self.model._modules['up1']._modules['conv']._modules['double_conv']._modules['0'],
                  self.model._modules['up1']._modules['conv']._modules['double_conv']._modules['3'],
                  self.model._modules['up2']._modules['conv']._modules['double_conv']._modules['0'],
                  self.model._modules['up2']._modules['conv']._modules['double_conv']._modules['3'],
                  self.model._modules['up3']._modules['conv']._modules['double_conv']._modules['0'],
                  self.model._modules['up3']._modules['conv']._modules['double_conv']._modules['3'],
                  self.model._modules['up4']._modules['conv']._modules['double_conv']._modules['0'],
                  self.model._modules['up4']._modules['conv']._modules['double_conv']._modules['3'],
                  self.model._modules['m_outc1'],
                  self.model._modules['m_outc2'],
                  self.model._modules['v_outc1'],
                  self.model._modules['v_outc2']
                        ] 
        
            for i in range(len(conv_layers)):
                create_weight(conv_layers[i],laplacian_dist)  


        with tqdm(range(0,self.args.epochs),leave=True) as tnr:
            tnr.set_postfix(training_loss= np.nan, validation_loss= np.nan,best_validation_loss = np.nan)
            for n in tnr:

                # to check the network weights
                #params = list(self.model.parameters())
                #print(params[0].data)

                # check epoch value for tau
                if self.args.change_tau == True:
                    flag.epoch = n

                self.training(tnr)
                
                if self.epoch % self.args.val_every_n_epochs == 0:
                    ##save best l1
                    if args.if_bayesian == True and args.save_best_l1 == True:
                        self.validate_bay()
                    else:
                        self.validate()

                if self.args.lr_scheduler == "step":
                    self.scheduler.step()
                    self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch )

                if self.args.lr_scheduler == "multi":
                    self.scheduler.step()
                    self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch )

                if self.args.lr_scheduler == "cyclic":
                    self.scheduler.step()
                    self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch )
                
                self.epoch += 1

                if self.args.save_model == "last":
                    self.save_model()

    def training(self,tnr=None):

        self.train_stats = None

        self.model.train()
        with tqdm(self.dataloaders["train"],leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss= np.nan)
            for en,sample in enumerate(inner_tnr):
                sample = self.to_device(sample)

                self.optimizer.zero_grad()
                output = self.model(sample)

                # multiply output with mask
                #output['y_pred'] = torch.mul(output['y_pred'], sample['mask'])
                #if args.if_bayesian == True:
                #    output['sigma'] = torch.mul(output['sigma'], sample['mask'])

                #loss,loss_dict = self.model.get_loss(sample,output)
                loss, loss_dict = L.get_loss(self, sample, output)

                if self.train_stats is None:
                    self.train_stats = loss_dict.copy()
                else:
                    for key in loss_dict:
                        self.train_stats[key] += loss_dict[key]
                
                loss.backward() #retain_graph=True)
                self.optimizer.step()

                self.iter += 1

                if (en+1) % self.args.logstep_train == 0:

                    for key in self.train_stats:
                        self.train_stats[key] = self.train_stats[key]  / self.args.logstep_train
                    
                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss= self.val_stats["optimization_loss"],
                                        best_validation_loss = self.val_stats["best_optimization_loss"])
                    
                    for key in self.train_stats:
                        self.writer.add_scalar('training/'+key, self.train_stats[key], self.iter )

                    self.train_stats = None

    def validate(self,tnr=None,save=True):

        for key in self.val_stats:
            if key != "best_optimization_loss":
                self.val_stats[key] = 0.

        with torch.no_grad():
            self.model.eval()
            for sample in tqdm(self.dataloaders["val"], leave=False):
                sample = self.to_device(sample)

                output = self.model(sample)
                
                #loss,loss_dict = self.model.get_loss(sample,output)
                loss, loss_dict = L.get_loss(self, sample, output)
                
                for key in loss_dict:
                    if key in self.val_stats:
                        self.val_stats[key] += loss_dict[key]
                    else:
                        self.val_stats[key] = loss_dict[key]
            
            # you can add patches to be visualozed in tensorboard that is very useful you just need to adapt
            # this function I think it is pretty straightforward
            #add_tensorboard_images(self.writer, sample, output, global_step=self.epoch,args=self.args)

            for key in self.val_stats:
                if key != "best_optimization_loss":
                    self.val_stats[key] = self.val_stats[key] / len(self.dataloaders["val"])

            if not self.val_stats["best_optimization_loss"] < self.val_stats["optimization_loss"]:
                self.val_stats["best_optimization_loss"] = self.val_stats["optimization_loss"]
                if save and self.args.save_model == "best":
                    self.save_model()

            for key in self.val_stats:
                self.writer.add_scalar('validation/'+key, self.val_stats[key], self.epoch)

    def validate_bay(self, tnr=None, save=True):
    ## validation function for saving model based on l1 loss during bayesian training

        for key in self.val_stats:
                if key not in ["best_optimization_loss", "best_l1_loss"]:
                    self.val_stats[key] = 0.

        with torch.no_grad():
            self.model.eval()
            for sample in tqdm(self.dataloaders["val"], leave=False):
                sample = self.to_device(sample)

                output = self.model(sample)

                #loss, loss_dict = self.model.get_loss(sample, output)
                loss, loss_dict = L.get_loss(self, sample, output)

                for key in loss_dict:
                    if key in self.val_stats:
                        self.val_stats[key] += loss_dict[key]
                    else:
                        self.val_stats[key] = loss_dict[key]

            # you can add patches to be visualozed in tensorboard that is very useful you just need to adapt
            # this function I think it is pretty straightforward
            # add_tensorboard_images(self.writer, sample, output, global_step=self.epoch,args=self.args)


            for key in self.val_stats:
                if key not in ["best_optimization_loss", "best_l1_loss"]:
                    self.val_stats[key] = self.val_stats[key] / len(self.dataloaders["val"])


            if not self.val_stats["best_optimization_loss"] < self.val_stats["optimization_loss"]:
                self.val_stats["best_optimization_loss"] = self.val_stats["optimization_loss"]

            if not self.val_stats["best_l1_loss"] < self.val_stats["l1_loss"]:
                self.val_stats["best_l1_loss"] = self.val_stats["l1_loss"]
                if save and self.args.save_model == "best":
                    self.save_model()

            for key in self.val_stats:
                self.writer.add_scalar('validation/' + key, self.val_stats[key], self.epoch)

    def simulate_full_rainfall_event(self,dataloader_key="test",index=0):

        assert (len(self.dataloaders[dataloader_key].dataset) > index)

        with torch.no_grad():
            self.model.eval()
            
            sample = self.dataloaders[dataloader_key].dataset[index]
                
            y_full_pred = torch.zeros_like(sample["gt"])
            current_x  = sample["data"][0:1]
            for t in tqdm(range(0,sample["data"].shape[0])):
                output = self.model({"data":current_x.to(self.device)})

                y_full_pred[t] = output["y_pred"][0].cpu()

                if t < sample["data"].shape[0]-1:
                    current_x = sample["data"][t+1:t+2]
                    current_x[0,1] = output["y_pred"][0,0].cpu()
            
        print("MAE loss for the full event simultation: {}".format(torch.mean(torch.abs(y_full_pred - sample["gt"]))))

        return y_full_pred,sample["gt"]
           
    def test(self):
        
        raise NotImplementedError

    def save_model(self):
        if args.if_bayesian == True:
            torch.save(self.model.state_dict(), os.path.join(self.experiment_folder, "ensemble_" + str(args.model_number) + ".pth.tar"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.experiment_folder,"model.pth.tar"))
        
    def resume(self,path):
        if not os.path.isfile(path):
            raise RuntimeError("=> no checkpoint found at '{}'".format(path))
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint)
        
        print("model loaded.")

        return
    
    def to_device(self, sample, device=None):
        if device is None:
            device = self.device
        sampleout = {}
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                sampleout[key] = val.to(device=device, dtype=torch.float)
            elif isinstance(val, list):
                new_val = []
                for e in val:
                    if (isinstance(e, torch.Tensor)):
                        new_val.append(e.to(device=device, dtype=torch.float))
                    else:
                        new_val.append(val)
                sampleout[key] = new_val
            else:
                sampleout[key] = val
        return sampleout


if __name__ == '__main__':

    args = parser.parse_args()
    #args.seed = random.randint(start, stop)
    print(args)

    start_time = dt.now()

    developingSuite = DevelopingSuite(args)
    developingSuite.train_and_eval()
    developingSuite.writer.close()

    running_time = (dt.now() - start_time).seconds
    write_run_time(args, running_time)

    print("Done")
