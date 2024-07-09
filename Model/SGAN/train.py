from __future__ import print_function
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from models import highwayNetDiscriminator, highwayNetGenerator
import sys
import geopy.distance

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.path.insert(1, "Model")

from utils import (
    clean_train_values,
    get_model_memory_usage,
    get_model_memory_usage_gen,
    init_model,
    load_args,
    load_dataset,
    rmse,
)
from torch.utils.data import DataLoader
import time
import math
import datetime
import logging

from torch import nan, nn
from tensorboardX import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)


def train(
    trDataloader: DataLoader,
    valDataloader: DataLoader,
    train_epochs: int,
    generator: highwayNetGenerator,
    discriminator: highwayNetDiscriminator,
):

    batch_size = trDataloader.batch_size

    g_loss_fn = nn.BCELoss()
    g_loss_fn_2 = rmse
    d_loss_fn = nn.BCELoss()

    opt_generator = torch.optim.Adam(generator.parameters(), lr=0.001)
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.005)
    
    scheduler_d = StepLR(opt_generator, step_size=30, gamma=0.1)
    scheduler_g = StepLR(opt_discriminator, step_size=35, gamma=0.1)

    for epoch_num in range(train_epochs):

        current_loss_g = 0
        current_loss_d = 0
        
        n_loss_count_g = 0
        n_loss_count_d = 0
                
        d_steps_left = 1
        g_steps_left = 2
       
        epoch_t1 = torch.zeros(1).cuda()
        epoch_t2 = torch.zeros(1).cuda()
        epoch_t3 = torch.zeros(1).cuda()
        epoch_t4 = torch.zeros(1).cuda()
        epoch_t5 = torch.zeros(1).cuda()
        
        for i, data in enumerate(tqdm(trDataloader)):
          

            if d_steps_left > 0:
                current_loss_d += discriminator_step(
                    data,
                    generator,
                    discriminator,
                    opt_discriminator,
                    d_loss_fn,
                ).item()
                n_loss_count_d += 1
                d_steps_left -= 1

            elif g_steps_left > 0:
                tmp_loss,t1,t2,t3,t4,t5 = generator_step(
                    data,
                    generator,
                    discriminator,
                    opt_generator,
                    g_loss_fn,
                    g_loss_fn_2,
                  
                )
                current_loss_g += tmp_loss.item()
                
                epoch_t1 += t1.item()
                epoch_t2 += t2.item()
                epoch_t3 += t3.item()
                epoch_t4 += t4.item()
                epoch_t5 += t5.item()
                
                n_loss_count_g += 1
                g_steps_left -= 1

            torch.cuda.synchronize()
        
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            d_steps_left = 1
            g_steps_left = 2

        epoch_loss_d = current_loss_d / n_loss_count_d
        epoch_loss_g = current_loss_g / n_loss_count_g
        
        epoch_t1 = epoch_t1 / n_loss_count_g
        epoch_t2 = epoch_t2 / n_loss_count_g
        epoch_t3 = epoch_t3 / n_loss_count_g
        epoch_t4 = epoch_t4 / n_loss_count_g
        epoch_t5 = epoch_t5 / n_loss_count_g
        
        writer.add_scalar("Data/loss_train_D", epoch_loss_d, epoch_num)
        writer.add_scalar("Data/loss_train_G", epoch_loss_g, epoch_num)

        writer.add_scalar("Data/RMSE_t1", epoch_t1.item(), epoch_num)
        writer.add_scalar("Data/RMSE_t2", epoch_t2.item(), epoch_num)
        writer.add_scalar("Data/RMSE_t3", epoch_t3.item(), epoch_num)
        writer.add_scalar("Data/RMSE_t4", epoch_t4.item(), epoch_num)
        writer.add_scalar("Data/RMSE_t5", epoch_t5.item(), epoch_num)

        print(
            "Epoch no:",
            epoch_num + 1,
            "| Epoch progress(%):",
            format(i / (len(trDataloader.dataset) / batch_size) * 100, "0.2f"),
            "| Avg train loss_D:",
            format(epoch_loss_d, "0.4f"),
            "| Avg train loss_G:",
            format(epoch_loss_g, "0.4f"),
            "| loss t1 :", epoch_t1.item(),
            "| loss t2 :", epoch_t2.item(),
            "| loss t3 :", epoch_t3.item(),
            "| loss t4 :", epoch_t4.item(),
            "| loss t5 :", epoch_t5.item(),
        )
        
        current_loss_d = 0
        current_loss_g = 0
        epoch_t1 = 0
        epoch_t2 = 0
        epoch_t3 = 0
        epoch_t4 = 0
        epoch_t5 = 0
        
        print("Calculating on validation set")
        validate(valDataloader, generator, discriminator, writer, epoch_num)
        scheduler_d.step()
        scheduler_g.step()
    writer.flush()
    writer.close()
    
    torch.save(
        generator.state_dict(),
        "Model/Config/gen.tar",
    )
    torch.save(
        discriminator.state_dict(),
        "Model/Config/dis.tar",
    )


def validate(
    valDataloader,
    generator,
    discriminator,
    writer,
    epoch_num,
    g_loss_fn=nn.BCELoss(),
    g_loss_fn2=rmse,
    d_loss_fn=nn.BCELoss(),
):

    d_steps_left = 1
    g_steps_left = 1
    
    generator.eval()
    discriminator.eval()
    
    loss_g = 0
    loss_d = 0
    
    val_t1 = 0
    val_t2 = 0
    val_t3 = 0
    val_t4 = 0
    val_t5 = 0
    
    n_loss_count_d = 0
    n_loss_count_g = 0
    
    acc = 0.0
    tot_traj = 0
    
    with torch.no_grad():
        for i, data in enumerate(valDataloader):
                
            history, nbrs, fut, _, _, _, vel, acc_vehi  = data
            
            history = history.cuda()
            nbrs = nbrs.cuda()
            fut = fut.cuda()
            vel = vel.cuda()
            acc_vehi = acc_vehi.cuda()

            if d_steps_left > 0:

                pred_traj_fake = generator(history, nbrs, vel, acc_vehi)

                traj_real = torch.cat([history[::,:,:2], fut[::,:,:]], dim=0)
                traj_fake = torch.cat([history[::,:,:2], pred_traj_fake[:, :, :2]], dim=0)

                y_pred_fake = discriminator(traj_fake)
                y_pred_real = discriminator(traj_real)

                loss_fake = d_loss_fn(y_pred_fake, torch.zeros_like(y_pred_fake))
                loss_real = d_loss_fn(y_pred_real, torch.ones_like(y_pred_real))

                tot_traj += y_pred_fake.shape[0] + y_pred_real.shape[0]
                acc += (
                    1.0
                    * (y_pred_fake.round() == torch.zeros_like(y_pred_fake))
                    .sum()
                    .item()
                    + (y_pred_real.round() == torch.ones_like(y_pred_real)).sum().item()
                )

                loss_d += loss_fake.item()
                loss_d += loss_real.item()
                
                n_loss_count_d += 1
                d_steps_left -= 1

            elif g_steps_left > 0:
                traj_fake = generator(history, nbrs, vel, acc_vehi)
                
                t1,t2,t3,t4,t5, tot = g_loss_fn2(traj_fake[:, :, :], fut[::,:,:])

                loss_g+= tot.item()
                
                val_t1 += t1.item()    
                val_t2 += t2.item()
                val_t3 += t3.item()
                val_t4 += t4.item()
                val_t5 += t5.item()
                
                traj_fake = torch.cat([history[::,:,:2], traj_fake[:, :, :2]], dim=0)

                scores_fake = discriminator(traj_fake)

                loss_g += g_loss_fn(scores_fake, torch.ones_like(scores_fake)).item()
                
                n_loss_count_g += 1
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue
            
            d_steps_left = 1
            g_steps_left = 2

        acc = 100 * (acc / tot_traj)
        
        loss_d = loss_d / n_loss_count_d
        loss_g = loss_g / n_loss_count_g
        
        val_t1 = val_t1 / n_loss_count_g
        val_t2 = val_t2 / n_loss_count_g
        val_t3 = val_t3 / n_loss_count_g
        val_t4 = val_t4 / n_loss_count_g
        val_t5 = val_t5 / n_loss_count_g
        
        print(
            "Accuracy_val_D: ",
            format(acc, "0.4f"),
            "| Avg val loss_D:",
            format(loss_d, "0.4f"),
            "| Avg val loss_G:",
            format(loss_g, "0.4f"),
        )
        
        writer.add_scalar("Data/accuracy_val_D", acc, epoch_num)
        writer.add_scalar("Data/loss_val_D", loss_d, epoch_num)
        writer.add_scalar("Data/loss_val_G", loss_g, epoch_num)
        
        writer.add_scalar("Data/RMSE_val_t1", val_t1, epoch_num)
        writer.add_scalar("Data/RMSE_val_t2", val_t2, epoch_num)
        writer.add_scalar("Data/RMSE_val_t3", val_t3, epoch_num)
        writer.add_scalar("Data/RMSE_val_t4", val_t4, epoch_num)
        writer.add_scalar("Data/RMSE_val_t5", val_t5, epoch_num)

        generator.train()
        discriminator.train()


def discriminator_step(
    data,
    generator,
    discriminator,
    optimizer_d,
    d_loss_fn,
):
    history, nbrs, fut, _, _, _, vel, acc = data
  
    history = history.cuda()
    nbrs = nbrs.cuda()
    fut = fut.cuda()
    vel = vel.cuda()
    acc = acc.cuda()

    loss = torch.zeros(1).cuda()

    pred_traj_fake = generator(history, nbrs, vel, acc)

    traj_real = torch.cat([history[::,:,:2], fut[::,:,:]], dim=0)
    traj_fake = torch.cat([history[::,:,:2], pred_traj_fake[:, :, :]], dim=0)

    y_pred_fake = discriminator(traj_fake)
    y_pred_real = discriminator(traj_real)

    loss_fake = d_loss_fn(y_pred_fake, torch.zeros_like(y_pred_fake))
    loss_real = d_loss_fn(y_pred_real, torch.ones_like(y_pred_real))
    loss += loss_fake
    loss += loss_real

    optimizer_d.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
    optimizer_d.step()

    return loss


def generator_step(
    data,
    generator,
    discriminator,
    optimizer_g,
    g_loss_fn=nn.BCELoss(),
    g_loss_fn2=rmse,
):

    loss = torch.zeros(1).cuda()

    history, nbrs, fut, _, _, _, vel, acc = data

    history = history.cuda()
    nbrs = nbrs.cuda()
    fut = fut.cuda()
    vel = vel.cuda()
    acc = acc.cuda()

    traj_fake = generator(history, nbrs, vel, acc)

    t1,t2,t3,t4,t5, tot = g_loss_fn2(traj_fake[:, :, :], fut[::,:,:])
    
    loss+=tot
    
    traj_fake = torch.cat([history[::,:,:2], traj_fake[:, :, :]], dim=0)

    scores_fake = discriminator(traj_fake)

    loss += 0.2 * g_loss_fn(scores_fake, torch.ones_like(scores_fake))

    optimizer_g.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(generator.parameters(), 5)
    optimizer_g.step()
    
    return loss,t1,t2,t3,t4,t5


if __name__ == "__main__":

    clean_train_values("SperimentalValue/Test")
    writer = SummaryWriter("SperimentalValue/Test")
   
    batch_size = 64
    
    # Model Arguments
    args = load_args()

    # Initialize network
    gen, dis = init_model(args)
    
    mem_usg_g = get_model_memory_usage_gen(gen, (15, batch_size, 4), (15, batch_size, 9, 2), ( batch_size, 1), ( batch_size, 1))
    print("Memory usage of generator: ", mem_usg_g)
 
    # Load dataset
    trDataloader, valDataloader, _ = load_dataset(
        30, 50, batch_size
    )  # historical step 3s prediction 5s
    
    start_time = datetime.datetime.now()
    
    # Train
    train(trDataloader, valDataloader, 30, gen, dis)
    
    end_time = datetime.datetime.now()
    
    print("Total training time: ", end_time - start_time)
