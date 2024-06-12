import os
import time
from natsort import natsorted
import numpy as np
import pandas as pd
from scipy.io import savemat
from collections import defaultdict

start_time = time.time()
# Dataset Fields before preprocessing
ds_id = 0
(
    vehicle_id,
    frame_id,
    total_frames,
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
    o_zone,
    d_zone,
    
    int_id,
    section_id,
    direction,
    movement,
    
    preceding,
    following,
    space_headway,
    time_headway,
    
    location,
) = range(25)

def split_data(traj_all, train_ratio, val_ratio, subset):
    traj_tr = []
    traj_val = []
    traj_ts = []

    for k in range(1, subset + 1):
        # filter by dataset id
        subset = traj_all[traj_all[:, ds_id] == k]
        ul1 = int(train_ratio * subset.shape[0])
        ul2 = int((train_ratio + val_ratio) * subset.shape[0])

        traj_tr.append(subset[:ul1])
        traj_val.append(subset[ul1:ul2])
        traj_ts.append(subset[ul2:])

    return np.vstack(traj_tr), np.vstack(traj_val), np.vstack(traj_ts)


def preprocess_ngsim():

    # Path
    path_csv = "Data/cutted.csv"
    #path_csv = "Data/NGSIM_20240603.csv"
    # Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_length,
    # v_Width,v_Class,v_Vel,v_Acc,Lane_ID,O_Zone,D_Zone,Int_ID,Section_ID,Direction,Movement,
    # Preceding,Following,Space_Headway,Time_Headway,Location

    # Load data and add dataset ids
    traj = []
    data = pd.read_csv(path_csv, engine="c")
    data = data.drop_duplicates(subset=["Vehicle_ID", "Global_Time", "Location"])

    global ds_id, vehicle_id, frame_id, total_frames, global_time, local_x, local_y, global_x, global_y, v_length, v_width, v_class, v_vel, v_acc, lane_id, o_zone, d_zone, int_id, section_id, direction, movement, preceding, following, space_headway, time_headway, location
    
    # location
    locations = data["Location"].unique()

    for i, loc in enumerate(locations):
        dataset_id = (i + 1) * np.ones((data[data["Location"] == loc].shape[0], 1))
        traj.append(np.hstack((dataset_id, data[data["Location"] == loc])))

    (
        ds_id,
        vehicle_id,
        frame_id,
        total_frames,
        
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
        o_zone,
        
        d_zone,
        int_id,
        section_id,
        direction,
        
        movement,
        preceding,
        following,
        space_headway,
        
        time_headway,
        location,
        
    ) = range(26)

    vehTrajs = [defaultdict(list) for _ in range(len(locations))]
    vehTimes = [defaultdict(list) for _ in range(len(locations))]

    for loc in range(len(locations)):
        # get only the columns we need
        traj[loc] = traj[loc][
            :,
            [   
                ds_id,
                vehicle_id,
                frame_id,
                global_time,
                local_x,
                
                local_y,
                global_x,
                global_y,
                v_length,
                
                v_width,
                v_vel,
                v_acc,
                lane_id,
                
                section_id,
                direction,
                movement,
                preceding,
                
                following,
            ],
        ]

        for vehicle_data in traj[loc]:
            # vehicle_id state for vehicle_index from the header of the csv file
            veh_id = int(vehicle_data[vehicle_id])
            frame_num = int(vehicle_data[frame_id])
            vehTrajs[loc][veh_id].append(vehicle_data)
            vehTimes[loc][frame_num].append(vehicle_data)
            # vehTrajs = {loc: {veh_id: [vehicle_data, vehicle_data, ...]}}
            # vehTimes = {frame_num: [vehicle_data, vehicle_data, ...]}
    (
        ds_id,
        vehicle_id,
        frame_id,
        global_time,
        local_x,
        
        local_y,
        global_x,
        global_y,
        v_length,
        
        v_width,
        v_vel,
        v_acc,
        lane_id,
        
        section_id,
        direction,
        movement,
        preceding,
        
        following
    ) = range(18)
 

    
    # the resulting list is in the form of [ [location_id, vehicle_id, frame_id, ...], ...]
    traj_all = np.vstack(traj)
    traj_tr, traj_val, traj_ts = split_data(traj_all, 0.75, 0.1, len(locations))
    
    # just reorganizing the data
    tracks_tr = create_tracks(traj_tr, len(locations))
    tracks_val = create_tracks(traj_val, len(locations))
    tracks_ts = create_tracks(traj_ts, len(locations))

    # keep only the trajectories that have at least 80 frames
    traj_tr = filter_edge_cases(traj_tr, tracks_tr)
    traj_val = filter_edge_cases(traj_val, tracks_val)
    traj_ts = filter_edge_cases(traj_ts, tracks_ts)
    
    # use trajectory as samples (target vehicle) and tracks for get the neighbors of the samples

    save_mat("Data/TrainSet_new.mat", traj_tr, tracks_tr)
    save_mat("Data/ValSet_new.mat", traj_val, tracks_val)
    save_mat("Data/TestSet_new.mat", traj_ts, tracks_ts)


def create_tracks(traj, subset):
    tracks = {}
    for k in range(1, subset + 1):
        
        # filter by location
        traj_set = traj[traj[:, 0] == k]
        car_ids = np.unique(traj_set[:, vehicle_id])
        tracks[k] = {}
        for car_id in car_ids:
            vehtrack = traj_set[traj_set[:, vehicle_id] == car_id][
                :,
                [   ds_id,
                    vehicle_id,
                    frame_id,
                    global_time,
                    local_x,
                    local_y,
                    global_x,
                    global_y,
                    v_length,
                    v_width,
                    v_vel,
                    v_acc,
                    lane_id,
                    section_id,
                    direction,
                    movement,
                    preceding,
                    following,
                ],
            ].T
            tracks[k][int(car_id)] = vehtrack

    return tracks


def filter_edge_cases(traj, tracks):
    inds = []
    for k, data in enumerate(traj):
        loc = int(data[ds_id])  # dataset ID (Location)
        vid = int(data[vehicle_id])  # Vehicle ID
        t = int(data[frame_id])  # frame

        if vid in tracks[loc]:
            track_data = tracks[loc][vid]
            total_frames = track_data.shape[1]
            if total_frames >= 30 + 50:  # Need at least 33 frames (30 + 3) + 50 future frames
                
                # TODO check that are consecutive frames
                sorted_frames = natsorted(track_data[frame_id, : ]) 
                first_frame = sorted_frames[0]
                last_frame = sorted_frames[-1]
                if first_frame <= t - 27 and last_frame > t + 50:
                    inds.append(k)
    return traj[inds]


def save_mat(filename, traj, tracks):
    savemat(filename, {"traj": traj, "tracks": tracks})


if __name__ == "__main__":
    preprocess_ngsim()
    end_time = time.time()
    print("Time preprocessing: ", end_time - start_time)
