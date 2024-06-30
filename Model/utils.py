from __future__ import print_function, division
import datetime
import json
import pickle
import geopy
import torch
import sys

sys.path.insert(1, "Model/SGAN")
from models import highwayNetDiscriminator, highwayNetGenerator
from dataset import ngsimDataset
from torch.utils.data import DataLoader
import os, shutil
from torch import nn 

mse = nn.MSELoss()

def rmse_long(predicted_values, true_values):

    t1 = mse(predicted_values[:5, :, 0], true_values[:5, :, 0])
    t2 = mse(predicted_values[5:10, :, 0], true_values[5:10, :, 0])
    t3 = mse(predicted_values[10:15, :, 0], true_values[10:15, :, 0])
    t4 = mse(predicted_values[15:20, :, 0], true_values[15:20, :, 0])
    t5 = mse(predicted_values[20:25, :, 0], true_values[20:25, :, 0])

    return t1, t2, t3, t4, t5

def rmse_lat(predicted_values, true_values):
  
    t1 = mse(predicted_values[:5, :, 1], true_values[:5, :, 1])
    t2 = mse(predicted_values[5:10, :, 1], true_values[5:10, :, 1]) 
    t3 = mse(predicted_values[10:15, :, 1], true_values[10:15, :, 1])
    t4 = mse(predicted_values[15:20, :, 1], true_values[15:20, :, 1])
    t5 = mse(predicted_values[20:25, :, 1], true_values[20:25, :, 1])
    
    return t1, t2, t3, t4, t5

def rmse(predicted_values, true_values):
    # Convert coordinates from feet to meters   
    predicted_values = predicted_values * 0.3048
    true_values = true_values * 0.3048
    lat_t1, lat_t2, lat_t3, lat_t4, lat_t5 = rmse_lat(predicted_values, true_values)
    lon_t1, lon_t2, lon_t3, lon_t4, lon_t5  = rmse_long(predicted_values, true_values)
    tot_t1_loss = torch.mean((lat_t1+lon_t1))
    tot_t2_loss = torch.mean((lat_t2+lon_t2))
    tot_t3_loss = torch.mean((lat_t3+lon_t3))
    tot_t4_loss = torch.mean((lat_t4+lon_t4))
    tot_t5_loss = torch.mean((lat_t5+lon_t5))
 
    tot = torch.sqrt(tot_t1_loss+tot_t2_loss+tot_t3_loss+tot_t4_loss+tot_t5_loss)
   
    tot_t1 = torch.sqrt(tot_t1_loss)
    tot_t2 = torch.sqrt(tot_t2_loss)
    tot_t3 = torch.sqrt(tot_t3_loss)
    tot_t4 = torch.sqrt(tot_t4_loss)
    tot_t5 = torch.sqrt(tot_t5_loss)

    return tot_t1, tot_t2, tot_t3, tot_t4, tot_t5, tot


def rmse_old(predicted_values, true_values):
  
    t1 = torch.sqrt(torch.mean((predicted_values[:5,:,:] - true_values[:5,:,:] )**2))
    t2 = torch.sqrt(torch.mean((predicted_values[5:10,:,:] - true_values[5:10,:,:])**2))
    t3 = torch.sqrt(torch.mean((predicted_values[10:15:,:] - true_values[10:15:,:])**2))
    t4 = torch.sqrt(torch.mean((predicted_values[15:20,:,:] - true_values[15:20,:,:])**2))
    t5 = torch.sqrt(torch.mean((predicted_values[20:25,:,:] - true_values[20:25,:,:])**2))
    
    #Total RMSE
    difference = predicted_values - true_values
    squared_difference = difference**2
    mean_squared_difference = torch.mean(squared_difference)
    rmse = torch.sqrt(mean_squared_difference)

      
    return t1,t2,t3,t4,t5,rmse


def load_args():
    with open(
        "Model/Config/net_arguments.json",
        "r",
    ) as read_file:
        args = json.load(read_file)
    return args


def load_dataset(t_h, t_f, batch_size=128):
    
    #  trSet = ngsimDataset('Data/TrainSet.mat', t_h=t_h )
    # valSet = ngsimDataset(
    #     "Data/sample.csv","Data/sample_tracks.csv"
    # )
    samples = None
    with open("Data/val_processed.pkl", "rb") as f:
        samples = pickle.load(f)
    
    # fileter sample from nan value
    samples = [sample for sample in samples if not torch.isnan(sample[0]).any() and not torch.isnan(sample[1]).any() and not torch.isnan(sample[2]).any()]
    #70% of the data is used for training
    train_samples = samples[:int(0.7 * len(samples))]
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
        "Data/ValSet_samples.csv",
        "Data/ValSet_tracks.csv",
        samples=train_samples
    )
    
    tsSet = ngsimDataset(
        "Data/ValSet_samples.csv",
        "Data/ValSet_tracks.csv",
        samples=test_samples
    )

    valDataloader = DataLoader(
        valSet,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=6,
        collate_fn=valSet.collate_fn,
        pin_memory=True,
    )
    trDataloader = DataLoader(
        trSet,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=4,
        collate_fn=trSet.collate_fn,
        pin_memory=True,
    )
    tsDataloader = DataLoader(
        tsSet,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=4,
        collate_fn=tsSet.collate_fn,
        pin_memory=True,
    )


    return trDataloader, valDataloader, tsDataloader


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

# Initialize network
def load_model():
    gen = highwayNetGenerator()
    dis = highwayNetDiscriminator()
    
    gen.load_state_dict(torch.load("Model/Config/gen.tar"))
    dis.load_state_dict(torch.load("Model/Config/dis.tar"))

    gen = gen.cuda()
    dis = dis.cuda()
    return gen, dis


def get_model_memory_usage_gen(model, input_size, input_size_2,input_size_3,input_size_4):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    activations = model(
        torch.randn(*input_size).cuda(), torch.randn(*input_size_2).cuda(), torch.randn(*input_size_3).cuda(), torch.randn(*input_size_4).cuda()
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
