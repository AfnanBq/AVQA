# Arabic Visual Question Answering 
This Project aims to create VQA system based on Standard Arabic dataset. 
## Table of contents
* [Introduction](#introduction)
* [Scope of functionalities](#scope-of-functionalities)
* [Results](#results)
* [Technologies](#technologies)
* [Setup](#setup)
* [Contact us](#contact-us)
* [References](#references)

## Introduction

## Scope of functionalities
The proposed system focuses on two types of questions in Standard Arabic(هل هذه ..؟، ماذا يوجد..؟)for natural language processing side. For example, Is this a cat? "هل هذه قطة" and What is in
the picture? "ماذا يوجد في الصورة". For the computer vision side we chose two general catogries: animals and transportation with 18 different classes.
![alt text](https://github.com/AfnanBq/AVQA/blob/master/dataset.png?raw=true)

## Results
![alt text](https://github.com/AfnanBq/AVQA/blob/master/results.png?raw=true)
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
## Contact us
## References
