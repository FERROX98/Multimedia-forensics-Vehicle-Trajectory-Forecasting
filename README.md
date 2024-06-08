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
