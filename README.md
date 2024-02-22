# Human Activity Recognition Data Science Project
## Overview
This repository is dedicated to the Human Activity Recognition (HAR) project, which aims to classify different human activities based on the data from wearable sensors. The project focuses on developing models that can accurately recognize and classify activities like walking, reading book, using phone, writing etc.
## Files
### CNN 
*  CNN.py - CNN feature extactor model
*  cnn_utils.py - util functions for CNN feature extractor
### LSTM
*  lstm_autoencoder.py - LSTM autoencoder model
*  lstm_autoencoders_utils.py - util functionss for LSTM autoencoder
### main_models - contains different experiments conducted
*  cnn_to_rf.ipynb - 3D CNN feature extractor to Random Forest model
*  cnn_to_xgb.ipynb - 3D CNN feature extractor to XGBoost model
*  embedding_nn.ipynb - LSTM autoencoder to Neural Network model
*  embedding_rf.ipynb - LSTM autoencoder to Random Forest model
*  lstm+cnn_rf.ipynb - LSTM+CNN feature extractor to Random Forest model
*  lstm_secret_data.ipynb - 3D CNN feature extractor on extended train data with filled gaps into LSTM model
*  only_1Dcnn.ipynb - only 1D CNN feature extractor model
*  only_cnn.ipynb - only 3D CNN feature extractor model
*  only_rf.ipynb - only Random Forest model
*  only_xgboost.ipynb - only XGBoost model
*  simple_prob.ipynb - probability according to classes distribution in train
### main_utils - containts utils for main
* fill_ranges_script.ipynb - fills ranges in train_data.csv to extedn train data
* generate_graphs.ipynb - generate graphs from values from saved logs
* get_secret_results.ipynb - use output of lstm_secret_data.ipynb to generate submission file
* merge_lstm_results.ipynb - use ensemble method on 5 LSTM models from lstm_secret_data.ipynb
### models_utils
* Datasets.py - contains all datasets for PyTorch models
* GLOBALS.py - contains all global variables used in experiemnts
* utils.py - contains util functions
### NN
*  NeuralNetwork.py - NN model
*  nn_utils.py - util functions for NN model
### RF_XGB
*  RandomForest.py - Random Forest model
*  XGBoost.py - XGBoost model

## Dataset
[Link to Kaggle Dataset](https://www.kaggle.com/competitions/bgu-i-know-what-you-did-last-measurement-time/leaderboard)
### Structure
The dataset is composed of 2 types of files:
*  Acceleration data from Smartwatch sensor
*  X,Y,Z data from Vicon


## Results - [Link to Leaderboard](https://www.kaggle.com/competitions/bgu-i-know-what-you-did-last-measurement-time/leaderboard)
![image](https://github.com/Qehbr/Human-Activity-Recognition-DS-Project/assets/49615282/ca2db049-db66-4b35-add5-5c67e5b2f1b5)




