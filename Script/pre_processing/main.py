import datetime
import multiprocessing
import os
import time
from natsort import natsorted
import numpy as np
import pandas as pd
from scipy.io import savemat
from collections import defaultdict
import hdf5storage
import h5py
# from multiprocessing import Process


import csv

from tqdm import tqdm

start_time = datetime.datetime.now()
# Dataset Fields before preprocessing

(
    location,
    vehicle_id,
    frame_id,
    global_time,
    local_x,
    local_y,
    global_x,
    global_y,
    v_length,
    v_width,
    v_class,
    v_vel,
    v_acc,
    lane_id,
    direction,
    preceding,
    following,
    space_headway,
) = range(18)

headers = {
    "Vehicle_ID": "Int64",
    "Frame_ID": "Int64",
    "Total_Frames": "Int64",
    "Global_Time": "Int64",
    "Local_X": float,
    "Local_Y": float,
    "Global_X": float,
    "Global_Y": float,
    "v_length": float,
    "v_Width": float,
    "v_Class": "Int64",
    "v_Vel": float,
    "v_Acc": float,
    "Lane_ID": "Int64",
    "O_Zone": str,
    "D_Zone": str,
    "Int_ID": str,
    "Section_ID": str,
    "Direction": str,
    "Movement": str,
    "Preceding": "Int64",
    "Following": "Int64",
    "Space_Headway": float,
    "Time_Headway": "float",
    " Location": str,
}

headers_reduced = [
    "Location",
    "Vehicle_ID",
    "Frame_ID",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "Global_X",
    "Global_Y",
    "v_length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Direction",
    "Preceding",
    "Following",
    "Space_Headway",
]


def append_par(subset, return_dict, _type):
    return_dict[_type] = subset
    return_dict[_type + "_s"] = filter_edge_cases_loc(subset)


def split_data(
    subset,
    train_ratio,
    val_ratio,
    traj_tr,
    traj_val,
    traj_ts,
    sample_tr,
    sample_val,
    sample_ts
):

    print("Splitting data", subset.shape)
    ul1 = int(train_ratio * subset.shape[0])
    ul2 = int((train_ratio + val_ratio) * subset.shape[0])

    traj_tr.extend(subset[:ul1])
    traj_val.extend(subset[ul1:ul2])
    traj_ts.extend(subset[ul2:])
    sample_tr.extend(filter_edge_cases_loc(subset[:ul1]))
    sample_val.extend(filter_edge_cases_loc(subset[ul1:ul2]))
    sample_ts.extend(filter_edge_cases_loc(subset[ul2:]))

def preprocess_ngsim():
    
  
    # Path
    path_csv = "Data/cutted.csv"
    path_csv = "Data/NGSIM_20240603.csv"

    # Load data and add dataset ids
    traj = []
    data = pd.read_csv(path_csv, engine="c", dtype=headers, na_values=["-1"])
    data = data.drop_duplicates(subset=["Vehicle_ID", "Frame_ID", "Location"])

    # location
    locations = data["Location"].unique()

    traj_tr = []
    traj_val = []
    traj_ts = []
    sample_tr = []
    sample_val = []
    sample_ts = []

    for i, loc in enumerate(locations):
        traj = data[data["Location"] == loc][headers_reduced]
        split_data(
            traj.to_numpy(),
            0.70,
            0.1,
            traj_tr,
            traj_val,
            traj_ts,
            sample_tr,
            sample_val,
            sample_ts
        )

    traj_tr, traj_val, traj_ts = np.asarray(traj_tr), np.asarray(traj_val), np.asarray(traj_ts)
    sample_tr, sample_val, sample_ts = (
        np.asarray(sample_tr),
        np.asarray(sample_val),
        np.asarray(sample_ts),
    )

    print("Trajectories: ", traj_tr.shape, traj_val.shape, traj_ts.shape)
    
    if sample_tr.shape[0] == 0:
        print("No data to save for training")
    else:
        save_mat("Data/TrainSet", traj_tr, sample_tr)

    if sample_val.shape[0] == 0:
        print("No data to save for validation")
    else:
        save_mat("Data/ValSet", traj_val, sample_val)

    if sample_ts.shape[0] == 0:
        print("No data to save for testing")
    else:
        save_mat("Data/TestSet", traj_ts, sample_ts)


def filter_edge_cases_loc(traj):
    inds = []
    track_location = traj
    only_vehic_id = track_location[:, vehicle_id]
    car_ids = np.unique(only_vehic_id)
    for vid in tqdm(car_ids, desc="Filtering edge cases (loc)"):

        filter_vehicle = only_vehic_id == vid
        sorted_frames = natsorted(track_location[filter_vehicle])
        total_frames = len(sorted_frames)

        if total_frames >= 30 + 50:  # Need at least 30 + 50  frames
            k = sorted_frames[29:-50]
            if len(k) > 0:
                inds.extend(k)

    if len(inds) == 0:
        return np.array([])
    return inds


def save_mat(filename, tracks, samples):
    print(filename)

    df = pd.DataFrame(samples)
    df.to_csv(filename + "_samples.csv", header=headers_reduced, index=False)

    df = pd.DataFrame(tracks)
    df.to_csv(filename + "_tracks.csv", header=headers_reduced, index=False)

    print("Saved samples: ", samples.shape, " to ", filename)
    print("Saved n_tracks: ", len(tracks), " to ", filename)


if __name__ == "__main__":
    
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()    
    preprocess_ngsim()
    end_time = datetime.datetime.now()
    print("Time preprocessing: ", end_time - start_time)
    
