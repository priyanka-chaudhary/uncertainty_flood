import sys, argparse, os
import numpy as np
import random
import yaml

import torch
import torch.nn.functional as F

import loss as L

from torch.utils.tensorboard import SummaryWriter
# from dataset_utae import dataloader_args_utae, load_dataset_utae
# from dataset import load_dataset, dataloader_args
from dataloading import get_dataloaders
from utils import new_log
from models import get_model

from loss import l1_loss_weight, l2_loss, l1_loss, bay_loss, bay_loss_ts_out

from evaluation import predict_batch, predict_event, mae_event
from evaluation import plot_maes, multiboxplot, plot_answer_sample, boxplot_mae

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class DevelopingSuite(object):

    def __init__(self, args, catchment_kwargs):

        self.args = args
        
        self.dataloaders ={}
        self.dataloaders["train"], self.dataloaders["val"] = get_dataloaders(args, catchment_kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu")
        self.model = get_model(args, catchment_kwargs)
        self.i = 0
        self.model.to(self.device)
        if args.resume is not None:
            self.resume(path=args.resume)

        if "train" in args.mode:
            self.experiment_folder = new_log(catchment_kwargs, args.save_dir,args.model + "_" + args.tag,args=args)
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
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 53, 400], gamma=args.lr_gamma)
            elif args.lr_scheduler == 'cyclic':
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.00001, step_size_up=50, cycle_momentum=False)
            else:
                self.scheduler = None
            
        self.epoch = 0
        self.iter = 0

        self.val_stats = {}
        self.val_stats["best_optimization_loss"] = np.nan
        self.val_stats["optimization_loss"] = np.nan

    def train_and_eval(self):
        
        
        if self.args.l1_prior == True and self.args.if_bayesian == True:
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

                self.training(tnr)
                
                if self.epoch % self.args.val_every_n_epochs == 0:
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
#                         self.train_stats[key] = self.train_stats[key]  / self.args.logstep_train
                        self.train_stats[key] = self.train_stats[key] / len(self.dataloaders["train"])
                    
                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss= self.val_stats["optimization_loss"],
                                        best_validation_loss = self.val_stats["best_optimization_loss"])
                    
                    for key in self.train_stats:
#                         self.writer.add_scalar('training/'+key, self.train_stats[key], self.iter )
                        self.writer.add_scalar('training/'+key, self.train_stats[key], self.epoch )

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
        torch.save(self.model.state_dict(), os.path.join(self.experiment_folder,"model.pth.tar"))
        
    def resume(self,path):
        if not os.path.isfile(path):
            raise RuntimeError("=> no checkpoint found at '{}'".format(path))
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint)
        
        print("model loaded.")

        return
    
    def to_device_old(self, sample, device=None):
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

    def to_device(self, sample, device=None):
        if device is None:
            device = self.device
        sampleout = {}
        sampleout['gt'] = sample[1].to(device=device, dtype=torch.float)
        sampleout['mask'] = sample[0][1].to(device=device, dtype=torch.float)
        sampleout['data'] = sample[0][0].to(device=device, dtype=torch.float)

        return sampleout

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="training script")

    #### general parameters #####################################################
    parser.add_argument('--tag', default="temp",type=str)
    #parser.add_argument('--tag', default="__",type=str)
    parser.add_argument("--device",default="cuda",type=str,choices=["cuda", "cpu"])
    parser.add_argument("--save-dir",default="/scratch2/ml_flood/data/checkpoints/", 
                        help="Path to directory where models and logs should be saved saved !! this folder must already exist !!")
    parser.add_argument("--logstep-train", default=10,type=int,
                        help="iterations step for training log")
    parser.add_argument("--save-model", default="best",choices=['last','best','No'],help="which model to save")
    parser.add_argument("--val-every-n-epochs", type=int, default=1,help="interval of training epochs to run the validation")
    parser.add_argument('--resume', type=str, default=None,help='path to resume the model if needed')
    parser.add_argument('--mode',default="only_train",type=str,choices=["None","train","test","train_and_test", "only_train"],help="mode to be run")

    #### data parameters ##########################################################
    parser.add_argument("--data",default="709",type=str,choices=["toy", "709", "684", "multi"],help="dataset selection")
    parser.add_argument("--datafolder",type=str,help="root directory of the dataset")
    parser.add_argument("--workers", type=int, default=4,metavar="N",help="dataloader threads")
    parser.add_argument("--batch-size", type=int, default=8)

    #### optimizer parameters #####################################################
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--optimizer',default='adam', choices=['sgd','adam'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    #parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--w-decay', type=float, default=0)#1e-5)
    parser.add_argument('--lr-scheduler', type=str, default='multi',choices=['no','step', 'smart', 'multi', 'cyclic'],help='lr scheduler mode')
    parser.add_argument('--lr-step', type=int, default=350,help=' number of epochs between decreasing steps applies to lr-scheduler in [step, exp]')
    parser.add_argument('--lr-gamma', type=float, default=0.1,help='decrease rate')

    #### model parameters #####################################################	
    parser.add_argument("--model", default='unet3d', type=str,help="model to run: 'cnn', 'unet', 'utae', 'unet3d', 'unet_bay'")
    parser.add_argument("--loss", default="L1", type=str, help="loss ['MSE', 'L1', 'L1_upd', 'bay_loss', l1_loss_funct] ")
    parser.add_argument("--task",default="wd_ts",type=str,choices=["wd_ts", "max_depth"],help="select b/w task predicting water depth for next timesteps (wd_ts) or max depth for rainfall events")
    ## training files takes 2 random patches instead of 100 in the normal file. loc: /scratch2/flood_sim/data/709/one_alt/

    #### bayesian training parameters #########################################
    parser.add_argument("--if_bayesian",default=True,type=bool,choices=[True,False],help="training using bayesian predictive uncertainty") 
    parser.add_argument('--num_models', type=int, default=5)
    parser.add_argument("--model_number",default="model1",type=str,help="ensemble model number training")
    parser.add_argument('--seed', type=int, default=27,help="model1: 19, model2: 27, model3: 37, model4: 45, model5: 73")
    
    ## saving model on l1 loss if bayesian
    parser.add_argument('--save_best_l1',default=False,type=bool,choices=[True,False],help="saving model based on best l1 loss")    
    ## laplace prior
    parser.add_argument('--l1_prior',default=True,type=bool,choices=[True,False],help="saving model based on best l1 loss")
    ## prior scale value
    parser.add_argument("--prior_sc", default=0.05,type=float,help="fraction of non-zero elements to accept a patch for training")
    #parser.add_argument('--exp', type=str, default='exp4',choices=['exp1','exp2', 'exp3', 'exp4'],help='select which experiment to run')

    
    ### fix indexes validation
    parser.add_argument("--fix_indexes_val", default=False, const=True, nargs='?', type=str2bool,help="select whether patches are generated sequentially (true) or randomly")

    #### UTAE #####################################################
    parser.add_argument('--n_head', type=int, default=16)
    
    parser.add_argument("--catchment_kwargs", default='.exp_yml/default_catchment_kwargs.yml', type=str, 
                        help="path to catchment kwargs saved in yml file")
    
    
    args = parser.parse_args()
    print(args)
    
    with open(args.catchment_kwargs) as file:
        catchment_kwargs = yaml.full_load(file)
    
    developingSuite = DevelopingSuite(args, catchment_kwargs)

    developingSuite.train_and_eval()

    developingSuite.writer.close()
    print("Done")