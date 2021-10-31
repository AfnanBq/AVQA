# Arabic Visual Question Answering 
This Project aims to create VQA system based on Standard Arabic dataset and MSCOCO dataset. 
## Table of contents
* [Introduction](#introduction)
* [Results](#results)
* [Technologies](#technologies)
* [Setup](#setup)
* [Contact Us](#contact us)

## Introduction
## Results
## Technologies
Project is created with:
* Python 3.6.8 
* gensim 3.8.1
* Keras  2.4.3
* matplotlib 2.2.2
* numpy 1.19.5
* pandas 0.23.0
* tensorflow 2.5.0
* 
## Setup
To run this project, install it locally:
* git clone https://github.com/AfnanBq/AVQA.git
* cd ../AVQA 
* install all the required libraries [Technologies](#technologies)
```
* run the following command put your values for the arguments.
Python train.py --type (train or test) --epoch (number of epoch) --batch_size (number of batch size) --train_file_name (json file name) --model_name (to save the model with proposed destination and name) --test_file_name (json file name for testing) 
* Example 
Python train.py --type train --epoch 5 --batch_size 16 --train_file_name yes_no_train.json --model_name .. /AVQA/models/yes_no_resnet_withdropout.hdf5 --test_file_name yes_no_test.json 
```
## Contact Us
