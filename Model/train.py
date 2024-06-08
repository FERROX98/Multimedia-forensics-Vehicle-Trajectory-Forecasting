from __future__ import print_function
import torch
from model import highwayNet
from utils import init_model, load_args, load_dataset, maskedNLL, rmse
from torch.utils.data import DataLoader
import time
import math
import datetime

from torch import nn
from tensorboardX import SummaryWriter


def train(
    trDataloader: DataLoader,
    valDataloader: DataLoader,
    train_epochs: int,
    model: highwayNet,
):

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = trDataloader.batch_size

    writer = SummaryWriter("SperimentalValue")

    prev_val_loss = math.inf
    model.debug = False
    
    mse_loss = nn.MSELoss()
    nll_loss = nn.NLLLoss()
    
    for epoch_num in range(train_epochs):

        avg_tr_loss = 0
        avg_tr_time = 0

        for i, data in enumerate(trDataloader):

            st_time = time.time()
            history, nbrs, fut, _, _, _ = data
            if model.debug:
                print("Fut: ", fut.shape)
                
            # Move to GPU
            history = history.cuda()
            nbrs = nbrs.cuda()
            fut = fut.cuda()

            fut_pred = model(history, nbrs)
            
            # Huber Loss 
            loss = mse_loss(fut_pred, fut)

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Calculate accuracy and time
            batch_time = time.time() - st_time
            avg_tr_loss += loss.item()
            avg_tr_time += batch_time
            accuracy = rmse(fut_pred, fut)

            if (i+1) % batch_size == 0:
                eta = avg_tr_time / 100 * (len(trDataloader.dataset) / batch_size - i)
                print(
                    "Epoch no:",
                    epoch_num + 1,
                    "| Epoch progress(%):",
                    format(i / (len(trDataloader.dataset) / batch_size) * 100, "0.2f"),
                    "| Avg train loss:",
                    format(avg_tr_loss / 100, "0.4f"),
                    "| Acc:",
                    format(accuracy, "0.4f"),
                    "| Validation loss prev epoch",
                    format(prev_val_loss, "0.4f"),
                    "| ETA(s):",
                    int(eta),
                )
                avg_tr_loss = 0
                avg_tr_time = 0

        model.train_flag = False

        print("Epoch", epoch_num + 1, "complete. Calculating validation loss...")

        writer.add_scalar("Multimedia-forensics-Vehicle-Trajectory-Forecasting/Data/accuracy_train", accuracy, epoch_num)
        writer.add_scalar("Multimedia-forensics-Vehicle-Trajectory-Forecasting/Data/loss_train", loss, epoch_num)

        prev_val_loss = validate(valDataloader, model, writer, epoch_num)

        end_time = datetime.datetime.now()
        print("Total training time: ", end_time - start_time)

        torch.save(
            model.state_dict(),
            "Multimedia-forensics-Vehicle-Trajectory-Forecasting/Model/Config/custom.tar",
        )

        # TODO plot loss, accuracy, validation


def validate(valDataloader, model, writer, epoch_num):

    avg_val_loss = 0
    val_batch_count = 0
    
    mse_loss = nn.MSELoss()
    nll_loss = nn.NLLLoss()
    
    for i, data in enumerate(valDataloader):
        history, nbrs, fut, _, _, _ = data

        # Move to GPU
        history = history.cuda()
        nbrs = nbrs.cuda()
        fut = fut.cuda()

        fut_pred = model(history, nbrs)

        loss = mse_loss(fut_pred, fut)

        # Calculate validation loss
        avg_val_loss += loss.item()
        val_batch_count += 1
        accuracy = rmse(fut_pred, fut)

    print(
        "Accuracy: ",
        format(accuracy, "0.4f"),
        "| Validation loss :",
        format(avg_val_loss / val_batch_count, "0.4f"),
    )
    writer.add_scalar("Multimedia-forensics-Vehicle-Trajectory-Forecasting/Data/validation_train", accuracy, epoch_num)
    return avg_val_loss / val_batch_count


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    # Model Arguments
    args = load_args()

    # Initialize network
    model = init_model(args)

    # Load dataset
    trDataloader, valDataloader = load_dataset(30, 50)  # historical step 3s prediction 5s
    
    # Train
    train(trDataloader, valDataloader, 10,  model)
    
    end_time = datetime.datetime.now()
    print("Total training time: ", end_time - start_time)
