from enum import Enum


class DatasetFields(Enum):
    LOCATION = 0
    VEHICLE_ID = 1
    FRAME_ID = 2
    GLOBAL_TIME = 3
    LOCAL_X = 4
    LOCAL_Y = 5
    GLOBAL_X = 6
    GLOBAL_Y = 7
    V_LENGTH = 8
    V_WIDTH = 9
    V_CLASS = 10
    V_VEL = 11
    V_ACC = 12
    LANE_ID = 13
    DIRECTION = 14
    PRECEDING = 15
    FOLLOWING = 16
    SPACE_HEADWAY = 17
    
    
class HeaderReduced(Enum):
    LOCATION = "Location"
    VEHICLE_ID = "Vehicle_ID"
    FRAME_ID = "Frame_ID"
    GLOBAL_TIME = "Global_Time"
    LOCAL_X = "Local_X"
    LOCAL_Y = "Local_Y"
    GLOBAL_X = "Global_X"
    GLOBAL_Y = "Global_Y"
    V_LENGTH = "v_length"
    V_WIDTH = "v_Width"
    V_CLASS = "v_Class"
    V_VEL = "v_Vel"
    V_ACC = "v_Acc"
    LANE_ID = "Lane_ID"
    DIRECTION = "Direction"
    PRECEDING = "Preceding"
    FOLLOWING = "Following"
    SPACE_HEADWAY = "Space_Headway"

def get_header_type():
    header_type = {
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
    return header_type

if __name__ == '__main__':
    # print(HeaderType.to_dict())
    print()