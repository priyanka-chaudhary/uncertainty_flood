import torch
import math
import torch.nn.functional as F
import numpy as np

def l1_loss_weight(output, target):
    mask = output[:,1]
    predictions =  output[:,0]
    target = target[:,0]
    predictions = torch.mul(predictions, mask)
    loss = torch.abs(predictions  - target)
    idx = target > 0.2
    loss[idx] = loss[idx] *4
    loss = loss.sum()/(torch.sum(mask.float()))
    return loss   # comput the mean only considering elemnts in the mask


def l1_loss(output, target):
    mask = output[:,1]
    predictions =  output[:,0]
    target = target[:,0]
    predictions = torch.mul(predictions, mask)
#     loss = torch.mean(torch.abs(predictions  - target))    ####################### this is incorrect because you want to divide by the number of elements in the mask
    loss = torch.abs(predictions  - target)
    loss = loss.sum()/(torch.sum(mask.float()))
    return loss   # comput the mean only considering elemnts in the mask

def l2_loss(output, target):
    mask = output[:,1]
    predictions =  output[:,0]
    target = target[:,0]
    predictions = torch.mul(predictions, mask)
    loss = torch.mean(torch.square(predictions  - target))
    return loss   # comput the mean only considering elemnts in the mask

def l2_weighted(input, target, mask):
    
    assert input.shape == target.shape

    loss = torch.abs((input -target)) 
    loss = torch.square(loss)
    a = target > 0.2
    loss[a] = loss[a] * 40  # loss[a] = torch.square(loss[a])
    loss = torch.mul(loss, mask)  # compute masked loss
    loss = loss.sum() / (torch.sum(mask.float()))
    return loss 

def l1_loss_upd(input, target, mask):
    ## L1 loss with increase loss value
    ## for wd > 20 cms by factor of 4
    
    assert input.shape == target.shape

    loss = torch.abs((input-target))
    # increase loss for pixels > 20 cm
    a = target > 0.2
    loss[a] = loss[a] * 4  # loss[a] = torch.square(loss[a])
    loss = torch.mul(loss, mask)  # compute masked loss
    loss = loss.sum() / (torch.sum(mask.float()))
    return loss  # comput the mean only considering elemnts in the mask

def l1_loss_funct(input_data, target, mask):

    loss = torch.abs((input_data-target))*target*10
    loss = torch.mul(loss, mask)  # compute masked loss
    loss = loss.sum() / (torch.sum(mask.float()))
    return loss  # comput the mean only considering elemnts in the mask


def bay_loss(predictions, target, mask):
    mu, sigma = predictions['y_pred'], predictions['sigma']    
    sigma = torch.sqrt(sigma)
    dist = torch.distributions.normal.Normal(mu, sigma)   # create normal distribution with the parameters: mean and standard deviation of the distribution
    loss = -dist.log_prob(target)   # call the log_prob function of the true value    
    # increase loss for pixels > 20 cm
    a = target > 0.2
    loss[a] = loss[a] * 4    #     loss[a] = torch.square(loss[a])
    loss = torch.mul(loss, mask)   # compute masked loss
    loss = loss.sum()/(torch.sum(mask.float()))
    return loss   # comput the mean only considering elemnts in the mask


def bay_loss_ts_out(predictions, target, local_mask):
    '''
    Loss for ts_out
    '''
    mu, sigma = predictions['y_pred'], predictions['sigma'] 
    sigma = torch.sqrt(sigma)
    dist = torch.distributions.normal.Normal(mu, sigma)   # create normal distribution with the parameters: mean and standard deviation of the distribution
    loss = -dist.log_prob(target)   # call the log_prob function of the true value  
    a = target > 0.2
    print(target.shape)
    loss[a] = loss[a] * 4
    loss = loss.sum()/(torch.sum(local_mask.float()))/3
    return loss


def lnll(predictions, target, mask, eps=1e-8):
    ## laplacian negative log likelihood loss
    ## with mean with data elements from mask
    mu, sigma = predictions['y_pred'], predictions['sigma'] 

    assert mu.shape == sigma.shape == target.shape
    #Clamp for stability
    
    sigma = sigma.clone()
    with torch.no_grad():
        sigma.clamp_(min=eps)
    loss = torch.abs((mu -target))/sigma + torch.log(sigma)
    # aincrease loss for pixels > 20 cm
    a = target > 0.2
    loss[a] = loss[a] * 4  # loss[a] = torch.square(loss[a])
    loss = torch.mul(loss, mask)  # compute masked loss
    loss = loss.sum() / (torch.sum(mask.float()))

    return loss 

def exp_loss(input, target, mask):

    assert input.shape == target.shape

    l1 = torch.exp(target-1)
    l2 = torch.square(input - target)
    loss = l1*l2
    loss = torch.mul(loss, mask)   # compute masked loss
    loss = loss.sum()/ (torch.sum(mask.float()))

    return loss  # comput the mean only considering elemnts in the mask


def get_loss(self, sample, output):

    out =  output['y_pred'] * sample['mask']
    if self.args.loss == "MSE":
        loss = F.mse_loss(out, sample["gt"])
    elif self.args.loss == "l2_w":
        loss = l2_weighted(output["y_pred"], sample["gt"], sample["mask"])
    elif self.args.loss == "L1":
        loss = F.l1_loss(out, sample["gt"])
    elif self.args.loss == "L1_upd":
        loss = l1_loss_upd(output["y_pred"], sample["gt"], sample["mask"])
    elif self.args.loss == "bay_loss":
        loss = bay_loss(output, sample["gt"], sample["mask"]) 
    elif self.args.loss == "lnll":
        loss = lnll(output, sample["gt"], sample["mask"]) 
    elif self.args.loss == "exp_loss":
        loss = exp_loss(output["y_pred"], sample["gt"], sample["mask"])   
    elif self.args.loss == "bay_loss_ts_out":
        loss = bay_loss(output, sample["gt"], sample["mask"])  
    elif self.args.loss == "l1_loss_funct":
        loss = l1_loss_funct(output, sample["gt"], sample["mask"])  
    else:
        raise NotImplementedError

    # maybe you want to predict other losses just add them
    # the loss is the one that will be used to compute the gradient
    # the others are ignored
    with torch.no_grad():
        out =  output['y_pred'] * sample['mask']
        mse_loss = F.mse_loss(output["y_pred"], sample["gt"])

        l1_loss = F.l1_loss(output["y_pred"], sample["gt"])

    return loss, {"optimization_loss": loss.detach().item(),
                      "mse_loss": mse_loss.item(),
                      "l1_loss": l1_loss.item()}



