# Deepfake_Video_Detection

Deepfake (derived from “deep learning” and “fake”) media refers to fictional images, videos, and audios synthesized by manipulating original media. The detection of such falsified content is imperative for authenticity verification and risk mitigation. Our research explored the usage of machine learning/deep learning techniques to detect whether a video is a deepfake or real.

The study adopted a two-pronged approach for the video classification task: analysis of temporal and visual inconsistencies across video frames.
For Temporal Features: We implement an ensemble of Convolutional Neural Network (CNN) such as ResNet and its variants and Recurrent Neural Network such as Long Short Term Memory (LSTM) model.

For Visual Inconsistencies: We employ MesoNet model architectures Meso-4, MesoInception-4 and their ensemble.

The models were trained, evaluated, and tested on the Deepfake Detection Challenge dataset (DFDC). The performance metrics such as accuracy, recall, precision, and F-1 scores were calculated to assess the efficacy of the models in detecting the deepfake videos.

### Dataset
We used DFDC dataset for training, evaluation and testing the models.

The link for the entire dfdc dataset is https://ai.facebook.com/datasets/dfdc/

For download, an aws account with IAM role is needed. 

The preprocessed videos can be found in hpc in the path 

/scratch/cmpe295-guzun/deepfake/dfdc/train/face_only_16k -> preprocessed  16k videos for (CNN+RNN and MesoNet) (split into train and validation)

/scratch/cmpe295-guzun/deepfake/dfdc/test/new_face_test  --> preprocessed testing videos (CNN+RNN and MesoNet)

The captured frames for training, validation and testing can be accessed under the following folders in HPC:

/scratch/cmpe295-guzun/deepfake/dfdc/train/save_train_frames  --> for training(MesoNet)

/scratch/cmpe295-guzun/deepfake/dfdc/train/save_valid_frames  --> for validation(MesoNet)

/scratch/cmpe295-guzun/deepfake/dfdc/test/save_test_frames2 ->  for testing (MesoNet)

### Ensemble of CNN and RNN models

This directory has the CNN models such as ResNet50, ResNet152, ResNeXt50 and ResNeXt101, each ensembled with LSTM model.

ResXX_ModelTrain.ipynb files shows how to train the model and respective ResXX_Predict.ipynb files show how to use checkpoints to make predictions on the test data.

Model checkpoints can be found here: https://drive.google.com/drive/folders/16OANq0C_USZ66C5TX4HQcZsDFPk5bKs1?usp=sharing

### MesoNet models and ensemble

All files related to MesoNet can be found in the folder MesoNet & Ensemble. The MesoFinal.ipnyb file shows how to preprocess, generate tensor images and train Meso4 and MesoInception4 models and make predictions on the DFDC test dataset. 

The file Ensemble.ipnyb has the code for stacked ensemble of Meso4 and MesoInception4 models.

Trained MesoNet models can be found in the saved folder of the HPC here: 

/scratch/cmpe295-guzun/deepfake/dfdc/train/model/Meso4_12ktrained-model-val2 (Meso4 trained model)

/scratch/cmpe295-guzun/deepfake/dfdc/train/model/MesoI_12ktrained-model-val (MesoI trained model)

