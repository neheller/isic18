# ISIC 2018
The code associated with our submission to the 2018 ISIC challenge

## General Structure
  * *dataset_prep* the location of all scripts used for preparing the data for
 training.
    * See README in that directory for more information
  * *task3* the location of all scripts used for training the models
    * _keras_model_utilities.py_ - This module managed the keras models on disk
    and held all reused code for model training
    * _\[model\_name\]\_k.py_ - The main scripts for each model training
    * _isic_data.py_ - The module which served data to the classifiers
    * _run_predictions.py_ - The main script for performing
    testing and full validation.
