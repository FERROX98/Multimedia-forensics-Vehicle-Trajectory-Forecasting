# Multimedia-Forensics-Vehicle-Trajectory-Forecasting

This repository contains the implementation of a vehicle trajectory forecasting model inspired by the work presented in the [conv-social-pooling repository](https://github.com/nachiket92/conv-social-pooling/blob/master) and the article "[Attention Based Vehicle Trajectory Prediction](https://inria.hal.science/hal-02543967/document)". The primary goal of this project is to predict future trajectories of vehicles based on their historical movements and the interactions with neighboring vehicles.

## Overview

Vehicle trajectory forecasting is a critical component in autonomous driving systems, enabling vehicles to anticipate the movements of others on the road. This project leverages deep learning techniques to model the complex interactions between vehicles and predict their future positions.


## Crucial Steps and Goals
1. **Data Preprocessing**: The preprocessing steps involve loading trajectory data, selecting relevant subsets, and preparing the data for training and evaluation.
2. **Model Architecture**: The model architecture is inspired by the conv-social-pooling repository, focusing on capturing the interactions between vehicles using LSTM and attention mechanisms.
3. **Training**: The training loop includes loss computation, backpropagation, and optimization steps. Monitoring the training progress with ETA calculations helps manage long training processes.
4. **Evaluation**: Evaluating the model on a test dataset helps in understanding its performance and identifying areas for improvement.
5. **Goal**: The ultimate goal is to accurately predict future trajectories of vehicles based on their past movements and interactions with neighboring vehicles, enhancing the safety and reliability of autonomous driving systems.


## Repository Structure

### Root Directory
- **Data/**: This directory contains datasets related to the project.

- **Model/**: This directory contains various scripts related to the model and its functionalities.
    - `dataset.py`: Handles loading and preprocessing of datasets.
    - `evaluate.py`: Contains code for evaluating the model's performance.
    - `model.py`: Defines the architecture of the model.
    - `train.py`: Handles the training process of the model.
    - `utils.py`: Contains utility functions used throughout the project.

 files.
  - **Config/**: This subdirectory contains params and argument of the model.

- **Report/images/**: This subdirectory under Report contains images used in reports or documentation.
  - `architecture.drawio`: A diagram representing the architecture of the model or system.

- **Script/**: This directory contains additional scripts or tools related to the project.
    - `main.py`: Pre processing script.


- **SperimentalValue**: Contanis the data relative to training.


## Pre Processing

In this project, we are using a CSV file associated with the NGSIM dataset. The NGSIM dataset contains pre-extracted data about agents (vehicles) on the road, which includes various attributes such as vehicle ID, frame ID, position coordinates, velocity, acceleration, and more. The primary objective of this project is not to extract information from raw data but to predict the future trajectories of target vehicles using the provided dataset.

The preprocessing steps are as follows:

1. Load Data: The data is loaded from the CSV file and preprocessed to remove duplicates and irrelevant entries. Each entry in the dataset contains information about a vehicle at a specific time frame.

2. Filter and Structure Data: The data is filtered and structured into appropriate formats to facilitate further processing and analysis. This involves organizing the data by location and vehicle ID filtering it by vehicle ID, sorting the data by frame ID, and then extracting sequences between specified frame ranges (from frame 30 to frame -50). This ensures that each extracted sequence meets the required criteria for further analysis.

3. Split Data: The dataset is split into training, validation, and test sets based on the specified ratios. This ensures that the model can be trained, validated, and tested on different subsets of the data.

4. Create Tracks: Vehicle tracks are created for each location. A track consists of the trajectory of a vehicle over time, capturing its movement and attributes frame by frame.

5. Filter Edge Cases: Trajectories with insufficient frames are filtered out to ensure the model has enough data to make accurate predictions.

## Extraction From Video Sequence
For completeness, we have also considered a model based on YOLO for extracting information from video sequences. YOLO is a state-of-the-art, real-time object detection system that can identify and locate multiple objects in video frames with high accuracy.

# Running Python Script for Training and Evaluation


## Train
```bash
python train.py
```

## Evaluate
```bash
python evaluate.py
```


## HighwayNet Model Architecture

HighwayNet is a deep learning model designed for vehicle trajectory forecasting. It comprises two primary components: the **Generator** and the **Discriminator**. These components collaborate to predict future vehicle trajectories from historical data and to discern between real and generated trajectories.

### Generator Architecture

The Generator generates future vehicle trajectories based on historical and neighboring vehicle data inputs.

- **Input Embedding Layers**: 
  - **Target Embedding Layer**: Embeds historical trajectory data.
  - **Neighbor Embedding Layer**: Embeds neighboring vehicle data.

- **Leaky ReLU Activation**: Applies a non-linear activation function with a small negative slope.

- **Encoder LSTMs**: 
  - **Target Encoder LSTM**: Encodes historical trajectory data.
  - **Neighbor Encoder LSTM**: Encodes neighboring vehicle data.

- **Decoder LSTM**: Combines encoded states to predict future trajectories.

- **Output Layer**: Linear layer mapping LSTM outputs to predicted trajectory points.

#### Workflow

1. **Input Processing**: Embeds historical and neighboring vehicle data.
2. **Encoding**: LSTM encodes embedded data.
3. **Decoding**: LSTM combines encoded states to predict trajectories.
4. **Prediction**: LSTM outputs are refined using Gaussian bivariate distribution.

### Discriminator Architecture

The Discriminator distinguishes between real and generated trajectories using LSTM and MLP networks.

- **Spatial Embedding**: Linear layer embedding 2D trajectory coordinates.

- **Encoder LSTM**: Encodes trajectory data into hidden states.

- **ReLU Activation**: Applies a non-linear ReLU activation function.

- **Real Classifier**: 
  - **Dropout Layer**: Randomly sets fraction of input units to zero during training.
  - **Linear Layer**: Maps LSTM hidden state to a single dimension.
  - **Sigmoid Activation**: Outputs probability (0 to 1) indicating trajectory authenticity.

#### Workflow

1. **Input Processing**: Embeds trajectory using spatial embedding.
2. **Encoding**: LSTM encodes embedded trajectory data.
3. **Classification**: MLP classifies trajectory authenticity using LSTM hidden states.

### Summary

TODO add some information about accuracy, F1 score, ecc and comparison with the orginal one 