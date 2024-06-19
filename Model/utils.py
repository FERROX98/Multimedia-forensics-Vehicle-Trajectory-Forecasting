from __future__ import print_function, division
import datetime
import json
import pickle
import torch
import sys

sys.path.insert(1, "Model/SGAN")
from models import highwayNetDiscriminator, highwayNetGenerator
from dataset import ngsimDataset
from model import highwayNet
from torch.utils.data import DataLoader
import os, shutil


def rmse(predicted_values, true_values):

    difference = predicted_values - true_values
    squared_difference = difference**2
    mean_squared_difference = torch.mean(squared_difference)
    rmse = torch.sqrt(mean_squared_difference)

    return rmse


def load_args():
    with open(
        "Model/Config/net_arguments.json",
        "r",
    ) as read_file:
        args = json.load(read_file)
    return args


def load_dataset(t_h, t_f, batch_size=128):
    #  trSet = ngsimDataset('Data/TrainSet.mat', t_h=t_h )
    # TODO mocked for fast testing
    # valSet = ngsimDataset(
    #     "Data/sample.csv","Data/sample_tracks.csv"
    # )
    samples = None
    with open("Data/val_processed.pkl", "rb") as f:
        samples = pickle.load(f)
    
    # fileter sample from nan value
    samples = [sample for sample in samples if not torch.isnan(sample[0]).any() and not torch.isnan(sample[1]).any() and not torch.isnan(sample[2]).any()]
    #70% of the data is used for training
    samples = samples[:int(0.7 * len(samples))]
    #10% of the data is used for validation
    val_samples = samples[int(0.7 * len(samples)) : int(0.8 * len(samples))]
    #20% of the data is used for testing
    test_samples = samples[int(0.8 * len(samples)) :]
    
    valSet = ngsimDataset(
        "Data/ValSet_samples.csv",
        "Data/ValSet_tracks.csv",
        samples=val_samples
        
    )

    trSet = ngsimDataset(
        "Data/TestSet_samples.csv",
        "Data/TestSet_tracks.csv",
        samples=test_samples
    )

    trDataloader = DataLoader(
        trSet,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=4,
        collate_fn=trSet.collate_fn,
        pin_memory=True,
    )
    valDataloader = DataLoader(
        valSet,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=4,
        collate_fn=valSet.collate_fn,
        pin_memory=True,
    )

    return trDataloader, valDataloader


## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt):
    input_dim = y_pred.shape[2]
    if input_dim == 5:
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        # If we represent likelihood in feet^(-1):
        out = (
            0.5
            * torch.pow(ohr, 2)
            * (
                torch.pow(sigX, 2) * torch.pow(x - muX, 2)
                + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
                - 2
                * rho
                * torch.pow(sigX, 1)
                * torch.pow(sigY, 1)
                * (x - muX)
                * (y - muY)
            )
            - torch.log(sigX * sigY * ohr)
            + 1.8379
        )
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


def clean_train_values(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


# Initialize network
def init_model(args):
    gen = highwayNetGenerator()

    dis = highwayNetDiscriminator()

    gen = gen.cuda()
    dis = dis.cuda()
    return gen, dis


def get_model_memory_usage_gen(model, input_size, input_size_2):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    activations = model(
        torch.randn(*input_size).cuda(), torch.randn(*input_size_2).cuda()
    )
    activations_memory = activations.element_size() * activations.nelement()
    total_memory = activations_memory + params * 4  # assuming 4 bytes per parameter
    return total_memory / (1024**2)  # convert to megabytes


def get_model_memory_usage(model, input_size):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    activations = model(torch.randn(*input_size))
    activations_memory = activations.element_size() * activations.nelement()
    total_memory = activations_memory + params * 4  # assuming 4 bytes per parameter
    return total_memory / (1024**2)  # convert to megabytes


def get_model_memory_usage_per_sample(model):
    # return the memory usage in MB
    return sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    ) / (1024**2)
