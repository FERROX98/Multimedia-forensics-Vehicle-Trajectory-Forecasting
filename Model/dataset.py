from __future__ import print_function, division
import datetime
import sys
from tqdm import tqdm
sys.path.insert(1, 'Script/pre_processing')

from natsort import natsorted
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
import hdf5storage
from header_enum import DatasetFields, HeaderReduced, get_header_type
# import matplotlib.pyplot as plt
from numpy import genfromtxt
from torch.utils.data import DataLoader

headers_reduced = [
   x.value for x in HeaderReduced
]

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
    ):

        print(samples_path, tracks_path)
        
        self.samples =  pd.read_csv(samples_path, engine="c", dtype=headers, na_values=["-1"]).to_numpy()
        print(self.samples.shape)

        self.tracks = pd.read_csv(tracks_path, engine="c", dtype=headers, na_values=["-1"]).to_numpy()
        print(self.tracks.shape)

        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        locId = self.samples[idx, DatasetFields.LOCATION.value]
        vehId = self.samples[idx, DatasetFields.VEHICLE_ID.value]
        t = self.samples[idx, DatasetFields.FRAME_ID.value]
        all_tracks = self.tracks[
            (
                (self.tracks[:, DatasetFields.LOCATION.value] == locId)
                & (self.tracks[:, DatasetFields.VEHICLE_ID.value] == vehId)
            )
        ]
        all_tracks = all_tracks[all_tracks[:, DatasetFields.FRAME_ID.value].argsort()]

        global_coord_x, global_coord_y = (
             self.samples[idx, DatasetFields.GLOBAL_X.value],
             self.samples[idx, DatasetFields.GLOBAL_Y.value],
        )

        # shape of hist: (30//downsampling, 2)
        hist = self.getHistory(locId, vehId, t, all_tracks)

        # shape of fut: (50//downsampling, 2)
        fut = self.getFuture(
            locId, vehId, t, all_tracks
        )  # future of the target vehicle

        # 4 meters in feet
        width_cell = 4 * 3.28084
        height_cell = 4 * 3.28084
        neighbors = []
        
        min_y = global_coord_y - (height_cell // 2) - height_cell * self.grid_size[1] // 2
        max_y = global_coord_y + (height_cell // 2) + height_cell *  self.grid_size[1] // 2
        min_x = global_coord_x - (width_cell // 2) - width_cell *  self.grid_size[0] // 2
        max_x = global_coord_x + (width_cell // 2) + width_cell *  self.grid_size[0] // 2

        column = np.linspace(min_y, max_y, self.grid_size[1])
        rows = np.linspace(min_x, max_x,  self.grid_size[0])


        for x_row in rows:
            for y_column in column:
                nbr_filtered = self.tracks[
                    (
                        (self.tracks[:, DatasetFields.LOCATION.value] == locId)
                        & self.tracks[:, DatasetFields.FRAME_ID.value]
                        == t &  (self.tracks[:, DatasetFields.VEHICLE_ID.value] != vehId)
                    )]
                nbr_filtered = nbr_filtered[(
                        (nbr_filtered[:, DatasetFields.GLOBAL_X.value] > x_row)
                        & (
                            nbr_filtered[:, DatasetFields.GLOBAL_X.value]
                            < x_row + width_cell
                        )
                        | (nbr_filtered[:, DatasetFields.GLOBAL_X.value] < x_row)
                        & (
                            nbr_filtered[:, DatasetFields.GLOBAL_X.value]
                            > x_row - width_cell
                        )
                    )
                    & (
                        (nbr_filtered[:, DatasetFields.GLOBAL_Y.value] > y_column)
                        & (
                            nbr_filtered[:, DatasetFields.GLOBAL_Y.value]
                            < y_column + height_cell
                        )
                        | (nbr_filtered[:, DatasetFields.GLOBAL_Y.value] < y_column)
                        & (
                            nbr_filtered[:, DatasetFields.GLOBAL_Y.value]
                            > y_column - height_cell
                        )
                    )
                ]

                if nbr_filtered.shape[0] == 0:
                    neighbors.append( np.zeros((15,2)))
                else:
                    i = 0
                    while i<nbr_filtered.shape[0]:
                        
                        vehId_nbrs = nbr_filtered[i, DatasetFields.VEHICLE_ID.value]
                        all_tracks_nbrs = self.tracks[
                            (
                                (self.tracks[:, DatasetFields.LOCATION.value] == locId)
                                & (self.tracks[:, DatasetFields.VEHICLE_ID.value] == vehId_nbrs)
                            )
                        ]
                        all_tracks_nbrs = all_tracks_nbrs[all_tracks_nbrs[:, DatasetFields.FRAME_ID.value].argsort()]
                        hist_nbrs = self.getHistory(locId, vehId_nbrs, t, all_tracks_nbrs)
                        if hist_nbrs.shape[0] != 0:
                            for i in range(hist_nbrs.shape[0], 15):
                                hist_nbrs = np.vstack((np.zeros((1,2)), hist_nbrs ))
                            break
                        i+=1
                        hist_nbrs = np.zeros((15,2))
                    neighbors.append(hist_nbrs)

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

        fut = frame_after_t[
            : 50: self.d_s,
            DatasetFields.LOCAL_X.value: DatasetFields.LOCAL_Y.value + 1,
        ]
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):

        nbrs_batch =  torch.zeros(30//self.d_s, len(samples), self.grid_size[0] * self.grid_size[1], 2)
        veh_ID_batch  = torch.zeros(len(samples), 1)
        time_batch  = torch.zeros(len(samples), 1)
        dsID_batch = []
        hist_batch = torch.zeros(30//self.d_s, len(samples), 2)
        fut_batch = torch.zeros(50//self.d_s, len(samples), 2)

        for sampleId, (hist, neighbors, fut, t, locId, vehId) in enumerate(samples):
          
            veh_ID_batch[sampleId, :]= torch.tensor(vehId).type(torch.int64)
            time_batch[sampleId, :]= torch.tensor(t).type(torch.int64)
            dsID_batch.append(locId)
       
            hist_batch[0:len(hist), sampleId, :] = torch.from_numpy(hist[:, :].astype(np.float32)).type(torch.float)
            fut_batch[0:len(fut), sampleId, :] = torch.from_numpy(fut[:, :].astype(np.float32)).type(torch.float)
            
            nbrs_batch[:,sampleId, :,:] = torch.from_numpy(np.asarray(neighbors).astype(np.float32)).type(torch.float).permute(1,0,2) 
    
        return (
            hist_batch,
            nbrs_batch,
            fut_batch,
            time_batch,
            dsID_batch,
            veh_ID_batch,
        )
        


if __name__ == '__main__':
    st_time = datetime.datetime.now()
    tr_set= ngsimDataset(
        "Data/TestSet_samples.csv","Data/TestSet_tracks.csv"
    )
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
        batch_size=12,
        shuffle=True,
        num_workers=2,
        collate_fn=tr_set.collate_fn,
        pin_memory=False,
    )
   
    for i, data in enumerate(tqdm(trDataloader)):
            history, nbrs, fut, _, _, _ = data
            end_time = datetime.datetime.now()
            print("time taken to load data", end_time-st_time)
            #break
