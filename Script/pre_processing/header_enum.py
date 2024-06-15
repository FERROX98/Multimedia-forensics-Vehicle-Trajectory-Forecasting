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
    
    
