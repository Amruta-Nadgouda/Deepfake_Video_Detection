# Deepfake_Video_Detection

Deepfake (derived from “deep learning” and “fake”) media refers to fictional images, videos, and audios synthesized by manipulating original media. The detection of such falsified content is imperative for authenticity verification and risk mitigation.
Our research explored the usage of machine learning/deep learning techniques to detect whether a video is a deepfake or real.
The study adopted a two-pronged approach for the video classification task: analysis of temporal and visual inconsistencies across video frames.
The former method was achieved by implementing an ensemble of Convolutional Neural Network (CNN) such as ResNet and its variants and Recurrent Neural Network such as Long Short Term Memory (LSTM) model, while the latter was addressed by employing MesoNet model architectures Meso-4, MesoInception-4 and their ensemble. The models were trained, evaluated, and tested on the Deepfake Detection Challenge dataset (DFDC). The performance metrics such as accuracy, recall, precision, and F-1 scores were calculated to assess the efficacy of the models in detecting the deepfake videos.

### Ensemble of CNN and RNN models

This directory has the CNN models such as ResNet50, ResNet152, ResNeXt50 and ResNeXt101, each ensembled with LSTM model.

ResXX_ModelTrain.ipynb files shows how to train the model and respective ResXX_Predict.ipynb files show how to use checkpoints to make predictions on the test data.

The checkpoints of the models can be found here: https://drive.google.com/drive/folders/16OANq0C_USZ66C5TX4HQcZsDFPk5bKs1?usp=sharing
