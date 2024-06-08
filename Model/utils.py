from __future__ import print_function, division
import json
import torch
from dataset import ngsimDataset
from model import highwayNet
from torch.utils.data import DataLoader



def rmse(predicted_values, true_values):

    difference = predicted_values - true_values
    squared_difference = difference**2
    mean_squared_difference = torch.mean(squared_difference)
    rmse = torch.sqrt(mean_squared_difference)

    return rmse


def load_args():
    with open(
        "Multimedia-forensics-Vehicle-Trajectory-Forecasting/Model/Config/net_arguments.json",
        "r",
    ) as read_file:
        args = json.load(read_file)
    return args


def load_dataset(t_h, t_f, batch_size=128):
    #  trSet = ngsimDataset('Multimedia-forensics-Vehicle-Trajectory-Forecasting/Data/TrainSet.mat', t_h=t_h )
    # TODO mocked for fast testing
    valSet = ngsimDataset(
        "Multimedia-forensics-Vehicle-Trajectory-Forecasting/Data/ValSet.mat", t_h=t_h, t_f=t_f
    )
    trSet = valSet

    trDataloader = DataLoader(
        valSet,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=valSet.collate_fn,
    )
    valDataloader = DataLoader(
        valSet,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=valSet.collate_fn,
    )
    return trDataloader, valDataloader

## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt):
    input_dim = y_pred.shape[2]
    if input_dim == 5:
        muX = y_pred[:,:,0]
        muY = y_pred[:,:,1]
        sigX = y_pred[:,:,2]
        sigY = y_pred[:,:,3]
        rho = y_pred[:,:,4]
        ohr = torch.pow(1-torch.pow(rho,2),-0.5)
        x = y_gt[:,:, 0]
        y = y_gt[:,:, 1]
        # If we represent likelihood in feet^(-1):
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        lossVal = out

    elif input_dim == 7:
        # FInd the NLL
        nll = compute_nll_mat_red(y_pred, y_gt)

        # nll_loss tensor filled with the loss value
        nll_loss = torch.zeros_like(mask)
        nll_loss[:, :, 0] = nll
        nll_loss[:, :, 1] = nll
        nll_loss[:, :, 2] = nll

        # mask the loss and find the mean value
        nll_loss = nll_loss * mask
        lossVal = torch.sum(nll_loss) / torch.sum(mask)

    return lossVal
# Initialize network
def init_model(args):
    model = highwayNet(args)
    model = model.cuda()
    return model
