from __future__ import print_function
import torch
from tqdm import tqdm
from models import highwayNetDiscriminator, highwayNetGenerator
import sys
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
    maskedNLL,
    rmse,
)
from torch.utils.data import DataLoader
import time
import math
import datetime
import logging

FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)
from torch import nn
from tensorboardX import SummaryWriter

writer = SummaryWriter("SperimentalValue")


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
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    init_tr_tipe = datetime.datetime.now()
    for epoch_num in range(train_epochs):

        current_loss_g = 0
        current_loss_d = 0
        n_loss_count_g = 0
        n_loss_count_d = 0
        d_steps_left = 2
        g_steps_left = 1
        flg_init = False
        st_time_load_dataset = datetime.datetime.now()
        for i, data in enumerate(tqdm(trDataloader)):
            st_time = datetime.datetime.now()
            if not flg_init:
                print("Time to load dataset: ", st_time - st_time_load_dataset)
                flg_init = True
            if d_steps_left > 0:
                current_loss_d += discriminator_step(
                    epoch_num,
                    data,
                    generator,
                    discriminator,
                    opt_discriminator,
                    d_loss_fn,
                ).item()
                n_loss_count_g += 1
                d_steps_left -= 1

            elif g_steps_left > 0:
                current_loss_g += generator_step(
                    epoch_num,
                    data,
                    generator,
                    discriminator,
                    opt_generator,
                    g_loss_fn,
                    g_loss_fn_2,
                ).item()
                n_loss_count_d += 1
                g_steps_left -= 1

            torch.cuda.synchronize()
            print("Time 1: %s", datetime.datetime.now() - st_time)
            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue
            print("Time 2 : %s", datetime.datetime.now() - st_time)
            # reset steps
            d_steps_left = 1
            g_steps_left = 1

        batch_time = datetime.datetime.now() - st_time
        epoch_loss_d = current_loss_d / n_loss_count_d
        epoch_loss_g = current_loss_g / n_loss_count_g
        writer.add_scalar("Data/loss_train_D", epoch_loss_d, epoch_num)
        writer.add_scalar("Data/loss_train_G", epoch_loss_g, epoch_num)
        # TODO write the accuracy of discriminator
        print(
            "Epoch no:",
            epoch_num + 1,
            "| Epoch progress(%):",
            format(i / (len(trDataloader.dataset) / batch_size) * 100, "0.2f"),
            "| Avg train loss_D:",
            format(epoch_loss_d, "0.4f"),
            "| Avg train loss_G:",
            format(epoch_loss_g, "0.4f"),
            "| Epoch Time(s):",
            batch_time,
        )
        current_loss_d = 0
        current_loss_g = 0

        print("Calculating on validation set")
        validate(valDataloader, generator, discriminator, writer, epoch_num)

    writer.flush()
    writer.close()
    
    print("total training time: ", datetime.datetime.now() - init_tr_tipe)
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
    n_loss_count_d = 0
    n_loss_count_g = 0
    
    acc = 0.0
    tot_traj = 0
    
    flg_init = False
    st_time_load_dataset = datetime.datetime.now()
    
    with torch.no_grad():
        for i, data in enumerate(valDataloader):
            
            if not flg_init:
                print(
                    "Time to load Val dataset: ",
                    datetime.datetime.now() - st_time_load_dataset,
                )
                flg_init = True
                
            history, nbrs, fut, _, _, _ = data
            history = history.cuda()
            nbrs = nbrs.cuda()
            fut = fut.cuda()

            if d_steps_left > 0:

                pred_traj_fake = generator(history, nbrs)

                traj_real = torch.cat([history, fut], dim=0)
                traj_fake = torch.cat([history, pred_traj_fake[:, :, :2]], dim=0)

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
                traj_fake = generator(history, nbrs)
                loss_g += g_loss_fn2(traj_fake[:, :, :2], fut).item()

                traj_fake = torch.cat([history, traj_fake[:, :, :2]], dim=0)

                scores_fake = discriminator(traj_fake)

                loss_g += g_loss_fn(scores_fake, torch.ones_like(scores_fake)).item()
                n_loss_count_g += 1
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue

        acc = 100 * (acc / tot_traj)
        loss_d = loss_d / n_loss_count_d
        loss_g = loss_g / n_loss_count_g
        
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

        generator.train()
        discriminator.train()


def discriminator_step(
    epoch_num,
    data,
    generator,
    discriminator,
    optimizer_d,
    d_loss_fn,
):
    history, nbrs, fut, _, _, _ = data

    history = history.cuda()
    nbrs = nbrs.cuda()
    fut = fut.cuda()

    loss = torch.zeros(1).cuda()

    pred_traj_fake = generator(history, nbrs)

    traj_real = torch.cat([history, fut], dim=0)
    traj_fake = torch.cat([history, pred_traj_fake[:, :, :2]], dim=0)

    y_pred_fake = discriminator(traj_fake)
    y_pred_real = discriminator(traj_real)

    if (
        y_pred_fake.max(dim=0)[0] > 1
        or y_pred_fake.min(dim=0)[0] < 0
        or y_pred_real.max(dim=0)[0] > 1
        or y_pred_real.min(dim=0)[0] < 0
    ):
        print("y_pred_fake", y_pred_fake)
        print("y_pred_real", y_pred_real)

    loss_fake = d_loss_fn(y_pred_fake, torch.zeros_like(y_pred_fake))
    loss_real = d_loss_fn(y_pred_real, torch.ones_like(y_pred_real))
    loss += loss_fake
    loss += loss_real

    optimizer_d.zero_grad()
    loss.backward()
    optimizer_d.step()

    return loss


def generator_step(
    epoch_num,
    data,
    generator,
    discriminator,
    optimizer_g,
    g_loss_fn=nn.BCELoss(),
    g_loss_fn2=rmse,
):

    loss = torch.zeros(1).cuda()

    history, nbrs, fut, _, _, _ = data

    history = history.cuda()
    nbrs = nbrs.cuda()
    fut = fut.cuda()

    traj_fake = generator(history, nbrs)

    # TODO vedere se cambiare output a solo 2 dimensioni
    loss += g_loss_fn2(traj_fake[:, :, :2], fut)

    traj_fake = torch.cat([history, traj_fake[:, :, :2]], dim=0)

    scores_fake = discriminator(traj_fake)

    loss += g_loss_fn(scores_fake, torch.ones_like(scores_fake))

    optimizer_g.zero_grad()
    loss.backward()
    optimizer_g.step()

    return loss


if __name__ == "__main__":

    clean_train_values("SperimentalValue")
    start_time = datetime.datetime.now()

    # Model Arguments
    args = load_args()

    # Initialize network
    gen, dis = init_model(args)
    input_size = ((15, 24, 2), (15, 24, 9, 2))
    mem_usg_g = get_model_memory_usage_gen(gen, (15, 24, 2), (15, 24, 9, 2))
    # mem_usg_d= get_model_memory_usage(dis,input_size)
    print("Memory usage of generator: ", mem_usg_g)
    # print("Memory usage of discriminator: ", mem_usg_d)
    
    # Load dataset
    trDataloader, valDataloader = load_dataset(
        30, 50
    )  # historical step 3s prediction 5s

    # Train
    train(trDataloader, valDataloader, 10, gen, dis)

    end_time = datetime.datetime.now()
    print("Total training time: ", end_time - start_time)
