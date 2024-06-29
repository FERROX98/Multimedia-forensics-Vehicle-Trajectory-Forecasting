import cv2
import torch
import pandas as pd
from pathlib import Path

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to calculate velocity
def calculate_velocity(prev_pos, current_pos, fps):
    distance = ((current_pos[0] - prev_pos[0]) ** 2 + (current_pos[1] - prev_pos[1]) ** 2) ** 0.5
    velocity = distance * fps
    return velocity

# Initialize variables
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
vehicle_data = []
prev_positions = {}

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    results = model(frame)
    
    # Process detections
    for det in results.xyxy[0]:
        xmin, ymin, xmax, ymax, confidence, cls = det
        
        # Only consider vehicle classes (YOLOv5 class IDs for vehicles: 2 - car, 3 - motorcycle, 5 - bus, 7 - truck)
        if int(cls) in [2, 3, 5, 7]:
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            vehicle_id = f'{frame_count}_{int(cls)}_{int(xmin)}_{int(ymin)}'
            
            if vehicle_id in prev_positions:
                prev_pos = prev_positions[vehicle_id]
                velocity = calculate_velocity(prev_pos, (center_x, center_y), fps)
            else:
                velocity = 0
            
            prev_positions[vehicle_id] = (center_x, center_y)
            
            vehicle_data.append({
                'Frame': frame_count,
                'Vehicle_ID': vehicle_id,
                'Class': int(cls),
                'Confidence': float(confidence),
                'Xmin': int(xmin),
                'Ymin': int(ymin),
                'Xmax': int(xmax),
                'Ymax': int(ymax),
                'Center_X': center_x,
                'Center_Y': center_y,
                'Velocity': velocity
            })

cap.release()

# Save data to CSV
df = pd.DataFrame(vehicle_data)
df.to_csv('vehicle_data.csv', index=False)