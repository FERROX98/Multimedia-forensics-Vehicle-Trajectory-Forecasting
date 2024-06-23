from __future__ import print_function
import torch
import sys
sys.path.insert(1, "Model")

from utils import (
    clean_train_values,
    load_args,
    load_dataset,
    load_model,
    ngsimDataset,
    rmse,
)
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np

from tensorboardX import SummaryWriter


def test():
    gen.eval()
    dis.eval()

    g_loss_fn = nn.BCELoss()
    g_loss_fn2 = rmse
    d_loss_fn = nn.BCELoss()


    vehid = []
    target_ID = []
    target_Loc = []
    pred_x = []
    pred_y = []
    T = []

    d_steps_left = 1
    g_steps_left = 1

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
    num_test = 0
    with torch.no_grad():
        for i, data in enumerate(tsDataloader):

            history, nbrs, fut,  t, locId, vehId, vel, acc_vehi = data

            vehid.append(vehId) 
            target_Loc.append(locId) 
            T.append(t)
            num_test += history.size()[1]
            history = history.cuda()
            nbrs = nbrs.cuda()
            fut = fut.cuda()
            vel = vel.cuda()
            acc_vehi = acc_vehi.cuda()

            if d_steps_left > 0:

                pred_traj_fake = gen(history, nbrs, vel, acc_vehi)

                traj_real = torch.cat([history, fut], dim=0)
                traj_fake = torch.cat([history, pred_traj_fake[:, :, :2]], dim=0)

                y_pred_fake = dis(traj_fake)
                y_pred_real = dis(traj_real)

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
                traj_fake = gen(history, nbrs, vel, acc_vehi)

                t1, t2, t3, t4, t5, tot = g_loss_fn2(traj_fake[:, :, :], fut)

                loss_g += tot.item()

                val_t1 += t1.item()
                val_t2 += t2.item()
                val_t3 += t3.item()
                val_t4 += t4.item()
                val_t5 += t5.item()

                traj_fake = torch.cat([history, traj_fake[:, :, :2]], dim=0)

                scores_fake = dis(traj_fake)

                #loss_g += g_loss_fn(scores_fake, torch.ones_like(scores_fake)).item()
                n_loss_count_g += 1
                
                g_steps_left -= 1
                    
                fut_pred_x = traj_fake[:,:,0].detach()
                fut_pred_x = fut_pred_x.cpu().numpy()
                #print (type(fut_pred_x)) # (25, 128)
                fut_pred_y = traj_fake[:,:,1].detach()
                fut_pred_y = fut_pred_y.cpu().numpy()
                pred_x.append(fut_pred_x)
                pred_y.append(fut_pred_y)

            if d_steps_left > 0 or g_steps_left > 0:
                continue
            
            d_steps_left = 1
            g_steps_left = 1

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

    writer.add_scalar("Data/accuracy_val_D", acc)
    writer.add_scalar("Data/loss_val_D", loss_d)
    writer.add_scalar("Data/loss_val_G", loss_g)

    writer.add_scalar("Data/RMSE_val_t1", val_t1)
    writer.add_scalar("Data/RMSE_val_t2", val_t2)
    writer.add_scalar("Data/RMSE_val_t3", val_t3)
    writer.add_scalar("Data/RMSE_val_t4", val_t4)
    writer.add_scalar("Data/RMSE_val_t5", val_t5)

    target_ID = sum(target_ID, [])
    target_ID = pd.DataFrame(target_ID)

    target_Loc = sum(target_Loc, [])
    target_Loc = pd.DataFrame(target_Loc)

    T = sum(T, [])
    T = pd.DataFrame(T)

    pred_x = np.concatenate(pred_x, axis=1)
    pred_x = pd.DataFrame(pred_x)

    pred_y = np.concatenate(pred_y, axis=1)
    pred_y = pd.DataFrame(pred_y)

    print("lossVal is:", loss_g)
    print("total test sample number:", num_test)


    print(
        "RMSE for each step is:", torch.pow(loss_g / n_loss_count_g, 0.5) * 0.3048
    )  # Calculate RMSE and convert from feet to meters

   # print("MAE for each step is:", mae / count_mae)
    print("Overall RMSE is:", torch.pow(sum(loss_g) / sum(n_loss_count_g), 0.5) * 0.3048)
  #  print("Overall MAE is:", sum(mae) / sum(count_mae))


if __name__ == "__main__":
    gen, dis = load_model()
    clean_train_values("SperimentalValue/TestValue")
    writer = SummaryWriter("SperimentalValue/TestValue")
    _, _, tsDataloader = load_dataset(batch_size=128)
