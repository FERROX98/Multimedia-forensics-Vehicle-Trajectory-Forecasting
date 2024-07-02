from __future__ import print_function, division
import datetime
import sys
from tqdm import tqdm

sys.path.insert(1, "Script/pre_processing")

from natsort import natsorted
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
import hdf5storage
from header_enum import DatasetFields, HeaderReduced, get_header_type

from numpy import genfromtxt
from torch.utils.data import DataLoader


import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)
headers_reduced = [x.value for x in HeaderReduced]

headers = get_header_type()


class ngsimDataset(Dataset):

    def __init__(
        self,
        samples_path,
        tracks_path,
        t_h=30,
        t_f=50,
        d_s=2,
        enc_size=64,
        grid_size=(3, 3),
        load_from_csv=False,
        samples=[]
    ):
        self.samples_csv = pd.read_csv(
                samples_path, engine="c", dtype=headers, na_values=["-1"]
            ).to_numpy()
        print(samples_path, tracks_path)
        
        if load_from_csv:
            self.samples = pd.read_csv(
                samples_path, engine="c", dtype=headers, na_values=["-1"]
            ).to_numpy()
            self.samples = self.samples[
                self.samples[:, DatasetFields.FRAME_ID.value].argsort()
            ]
            print(self.samples.shape)
            self.tracks = pd.read_csv(
                tracks_path, engine="c", dtype=headers, na_values=["-1"]
            ).to_numpy()
            self.tracks = self.tracks[
                self.tracks[:, DatasetFields.FRAME_ID.value].argsort()
            ]
            print(self.tracks.shape)
      
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid

        self.processed_samples = samples


    def extract_samples(self):
        st_time_init = datetime.datetime.now()

        for locId in tqdm(np.unique(self.samples[:, DatasetFields.LOCATION.value])):
            tracks = self.tracks[self.tracks[:, DatasetFields.LOCATION.value] == locId]
            samples = self.samples[
                self.samples[:, DatasetFields.LOCATION.value] == locId
            ]

            for idx in tqdm(range(len(samples))):

                locId = samples[idx, DatasetFields.LOCATION.value]
                vehId = samples[idx, DatasetFields.VEHICLE_ID.value]
                t = samples[idx, DatasetFields.FRAME_ID.value]
                all_tracks_filter_by_location = tracks
                all_tracks_filter_by_target = all_tracks_filter_by_location[
                    all_tracks_filter_by_location[:, DatasetFields.VEHICLE_ID.value]
                    == vehId
                ]

                global_coord_x, global_coord_y = (
                    samples[idx, DatasetFields.GLOBAL_X.value],
                    samples[idx, DatasetFields.GLOBAL_Y.value],
                )
        
    
                hist = torch.from_numpy(
                    self.getHistory(
                        locId, vehId, t, all_tracks_filter_by_target
                    ).astype(np.float32)
                ).type(torch.float)
        
                
                fut = torch.from_numpy(
                    self.getFuture(locId, vehId, t, all_tracks_filter_by_target).astype(
                        np.float32
                    )
                ).type(
                    torch.float
                )  # future of the target vehicle
               
                # 4 meters in feet
                width_cell = 4 * 3.28084
                height_cell = 4 * 3.28084

                neighbors = torch.from_numpy(
                    np.asarray(
                        self.get_neighbors(
                            locId,
                            vehId,
                            t,
                            global_coord_x,
                            global_coord_y,
                            width_cell,
                            height_cell,
                        )
                    ).astype(np.float32)
                ).type(torch.float)

                self.processed_samples.append((hist, neighbors, fut, t, locId, vehId))
                if idx % 1000 == 0:
                    print(
                        "Processed time for ",
                        idx,
                        " samples",
                        datetime.datetime.now() - st_time_init,
                    )
        print(
            "Total time taken to process data", datetime.datetime.now() - st_time_init
        )

    def save_samples(self):
        with open("Data/val_processed.pkl", "wb") as f:
            pickle.dump(self.processed_samples, f)

    def get_neighbors(
        self, locId, vehId, t, global_coord_x, global_coord_y, width_cell, height_cell
    ):
        # Create the grid
        min_y = (
            global_coord_y - (height_cell // 2) - height_cell * self.grid_size[1] // 2
        )
        max_y = (
            global_coord_y + (height_cell // 2) + height_cell * self.grid_size[1] // 2
        )
        min_x = global_coord_x - (width_cell // 2) - width_cell * self.grid_size[0] // 2
        max_x = global_coord_x + (width_cell // 2) + width_cell * self.grid_size[0] // 2

        columns = np.linspace(min_y, max_y, self.grid_size[1])
        rows = np.linspace(min_x, max_x, self.grid_size[0])

        # Filter tracks based on location and frame
        base_condition = (
            (self.tracks[:, DatasetFields.LOCATION.value] == locId)
            & (self.tracks[:, DatasetFields.FRAME_ID.value] == t)
            & (self.tracks[:, DatasetFields.VEHICLE_ID.value] != vehId)
        )

        nbr_filtered = self.tracks[base_condition]
        if nbr_filtered.shape[0] == 0:
            return [np.zeros((15, 2))] * (len(rows) * len(columns))

        neighbors = []
        for x_row in rows:
            x_mask = (
                (nbr_filtered[:, DatasetFields.GLOBAL_X.value] > x_row)
                & (nbr_filtered[:, DatasetFields.GLOBAL_X.value] < x_row + width_cell)
            ) | (
                (nbr_filtered[:, DatasetFields.GLOBAL_X.value] < x_row)
                & (nbr_filtered[:, DatasetFields.GLOBAL_X.value] > x_row - width_cell)
            )

            nbr_filtered_spatial = nbr_filtered[x_mask]
            for y_column in columns:
                # Combine conditions for spatial filtering

                y_mask = (
                    (nbr_filtered_spatial[:, DatasetFields.GLOBAL_Y.value] > y_column)
                    & (
                        nbr_filtered_spatial[:, DatasetFields.GLOBAL_Y.value]
                        < y_column + height_cell
                    )
                ) | (
                    (nbr_filtered_spatial[:, DatasetFields.GLOBAL_Y.value] < y_column)
                    & (
                        nbr_filtered_spatial[:, DatasetFields.GLOBAL_Y.value]
                        > y_column - height_cell
                    )
                )

                nbr_filtered_spatial = nbr_filtered_spatial[y_mask]

                if nbr_filtered_spatial.shape[0] == 0:
                    neighbors.append(np.zeros((15, 2)))
                else:
                 
                    found_valid_history = False
                    for i in range(nbr_filtered_spatial.shape[0]):
                        vehId_nbrs = nbr_filtered_spatial[
                            i, DatasetFields.VEHICLE_ID.value
                        ]
                        all_tracks_nbrs = self.tracks[
                            (self.tracks[:, DatasetFields.LOCATION.value] == locId)
                            & (
                                self.tracks[:, DatasetFields.VEHICLE_ID.value]
                                == vehId_nbrs
                            )
                        ]
                        all_tracks_nbrs = all_tracks_nbrs[
                            np.argsort(all_tracks_nbrs[:, DatasetFields.FRAME_ID.value])
                        ]
                        hist_nbrs = self.getHistory(
                            locId, vehId_nbrs, t, all_tracks_nbrs
                        )

                        if hist_nbrs.shape[0] != 0:
                            hist_nbrs = np.pad(
                                hist_nbrs,
                                ((15 - hist_nbrs.shape[0], 0), (0, 0)),
                                "constant",
                            )
                            neighbors.append(hist_nbrs)
                            found_valid_history = True
                            break

                    if not found_valid_history:
                        neighbors.append(np.zeros((15, 2)))

        return neighbors

    def __len__(self):
        if len(self.processed_samples)>0:
            return len(self.processed_samples)
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        #hist, neighbors, fut, t, locId, vehId))
        if len(self.processed_samples)>0:
            all_field = self.samples_csv[((self.samples_csv[:, DatasetFields.LOCATION.value] == self.processed_samples[idx][4]) &
                                        (self.samples_csv[:, DatasetFields.VEHICLE_ID.value] == self.processed_samples[idx][5]) &
                                        (self.samples_csv[:, DatasetFields.FRAME_ID.value] == self.processed_samples[idx][3])), :]
            
            all_field_past = self.samples_csv[((self.samples_csv[:, DatasetFields.LOCATION.value] == self.processed_samples[idx][4]) &
                                        (self.samples_csv[:, DatasetFields.VEHICLE_ID.value] == self.processed_samples[idx][5])), :]
            
            if (len(all_field) == 0):
                return (*self.processed_samples[idx], torch.tensor(10), torch.tensor(0.5))
            vel = all_field[0, DatasetFields.V_VEL.value] if all_field[0, DatasetFields.V_VEL.value] == all_field[0, DatasetFields.V_VEL.value] else np.asarray(1)
            acc = all_field[0, DatasetFields.V_ACC.value] if all_field[0, DatasetFields.V_ACC.value] == all_field[0, DatasetFields.V_ACC.value]  else np.asarray(1)
            
        
            frame_history = all_field_past[all_field_past[:, DatasetFields.FRAME_ID.value] <= self.processed_samples[idx][3]]
            vel_past = [ x if x==x else 1 for x in frame_history[-30::2, DatasetFields.V_VEL.value] ]
            acc_past = [ x if x==x else 1 for x in frame_history[-30::2, DatasetFields.V_ACC.value] ]
            if len(vel_past) < 15:
                vel_past = np.pad(vel_past, (15-len(vel_past), 0), 'constant', constant_values=(1))
            if len(acc_past) < 15:
                acc_past = np.pad(acc_past, (15-len(acc_past), 0), 'constant', constant_values=(1))

            self.processed_samples[idx] = (np.concatenate((self.processed_samples[idx][0], np.asarray(vel_past).reshape(15,1),np.asarray(acc_past).reshape(15,1)),1), self.processed_samples[idx][1],self.processed_samples[idx][2],self.processed_samples[idx][3],self.processed_samples[idx][4],self.processed_samples[idx][5])
           # direction = all_field[0, DatasetFields.DIRECTION.value]
            return (*self.processed_samples[idx], vel, acc)
        
        locId = self.samples[idx, DatasetFields.LOCATION.value]
        vehId = self.samples[idx, DatasetFields.VEHICLE_ID.value]
        t = self.samples[idx, DatasetFields.FRAME_ID.value]
        all_tracks_filter_by_location = self.tracks[
            ((self.tracks[:, DatasetFields.LOCATION.value] == locId))
        ]
        all_tracks_filter_by_target = all_tracks_filter_by_location[
            all_tracks_filter_by_location[:, DatasetFields.VEHICLE_ID.value] == vehId
        ]

        global_coord_x, global_coord_y = (
            self.samples[idx, DatasetFields.GLOBAL_X.value],
            self.samples[idx, DatasetFields.GLOBAL_Y.value],
        )

        # shape of hist: (30//downsampling, 2)
        hist = self.getHistory(locId, vehId, t, all_tracks_filter_by_target)

 
        # shape of fut: (50//downsampling, 2)
        fut = self.getFuture(
            locId, vehId, t, all_tracks_filter_by_target
        )  # future of the target vehicle

        # 4 meters in feet
        width_cell = 4 * 3.28084
        height_cell = 4 * 3.28084
        neighbors = self.get_neighbors(
            locId, vehId, t, global_coord_x, global_coord_y, width_cell, height_cell
        )

        return hist, neighbors, fut, t, locId, vehId

    ## Helper function to get track history
    def getHistory(self, locId, vehId, t, all_tracks):

        # Gen only the history of the target vehicle position
        frame_before_t = all_tracks[all_tracks[:, DatasetFields.FRAME_ID.value] <= t]
        hist = frame_before_t[
            -30 :: self.d_s,
            DatasetFields.LOCAL_X.value : DatasetFields.LOCAL_Y.value + 1,
        ]

        return hist

    ## Helper function to get track future
    def getFuture(self, locId, vehId, t, all_tracks):
        # Gen only the history of the target vehicle position
        frame_after_t = all_tracks[all_tracks[:, DatasetFields.FRAME_ID.value] > t]
        #  frame_after_t = frame_after_t[frame_after_t[:, DatasetFields.FRAME_ID.value].argsort()]
        fut = frame_after_t[
            : 50 : self.d_s,
            DatasetFields.LOCAL_X.value : DatasetFields.LOCAL_Y.value + 1,
        ]
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):
        nbrs_batch = torch.zeros(
            30 // self.d_s, len(samples), self.grid_size[0] * self.grid_size[1], 2
        )
        veh_ID_batch = torch.zeros(len(samples), 1)
        vel_batch = torch.zeros(len(samples), 1)
        acc_batch = torch.zeros(len(samples), 1)

        time_batch = torch.zeros(len(samples), 1)
        dsID_batch = []
        hist_batch = torch.zeros(30 // self.d_s, len(samples), 4)
        fut_batch = torch.zeros(50 // self.d_s, len(samples), 2)

        for sampleId, (hist, neighbors, fut, t, locId, vehId, vel, acc ) in enumerate(samples):

            veh_ID_batch[sampleId, :] = torch.tensor(vehId).type(torch.int64)
            vel_batch[sampleId, :] = torch.tensor(vel).type(torch.float)
            acc_batch[sampleId, :] = torch.tensor(acc).type(torch.float)
                        
            time_batch[sampleId, :] = torch.tensor(t).type(torch.int64)
            dsID_batch.append(locId)

            hist_batch[0 : len(hist), sampleId, :] =  torch.tensor(hist[:, :])
            
            fut_batch[0 : len(fut), sampleId, :] = fut[:, :]
        
            nbrs_batch[:, sampleId, :, :] = (neighbors.permute(1, 0, 2)
            )
        return (
            hist_batch,
            nbrs_batch,
            fut_batch,
            time_batch,
            dsID_batch,
            veh_ID_batch,
            vel_batch,
            acc_batch,
        )


if __name__ == "__main__":
    
    st_time = datetime.datetime.now()
    samples = []
    with open("Data/val_processed.pkl", "rb") as f:
        samples = pickle.load(f)
    
    tr_set = ngsimDataset("Data/ValSet_tracks.csv", "Data/ValSet_tracks.csv",samples=samples)
    
    # tr_set= ngsimDataset(
    #     "Data/TrainSet_samples.csv","Data/TrainSet_tracks.csv"
    # )
    # tr_set= ngsimDataset(
    #     "Data/sample.csv","Data/sample_tracks.csv"
    # )
    #   tr_set.__getitem__(0)
    # samples = [ tr_set.__getitem__(x) for x in range(2)]
    # tr_set.collate_fn(samples)

    trDataloader = DataLoader(
        tr_set,
        batch_size=48,
        # shuffle=True,
        num_workers=0,
        collate_fn=tr_set.collate_fn,
        pin_memory=True,
    )

    for i, data in enumerate(tqdm(trDataloader)):
        history, nbrs, fut, _, _, _ = data
