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

- **SperimentalValue**: Contanis the data relative to training.


# Running Python Script for Training and Evaluation

In this guide, we'll walk through the steps to run a Python script that trains and evaluates a machine learning model. Below are the necessary steps:

## Step 1: Install Dependencies

You'll need to install the required libraries for your machine learning project. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py
```

## Evaluate
```bash
python evaluate.py
```

