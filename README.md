# Skeletons & Animals

![image](https://user-images.githubusercontent.com/59373737/120549838-c3422900-c3fc-11eb-852e-345b29f54a38.png)


Software that can generate a photo of an animal based on its skeleton and a photo of an animal's skeleton based on its animal


## Developers students: 

Fedosenko Maxim 

Vasilenko Vladislav

## [Project presentation](https://docs.google.com/presentation/d/1u2XB4frWrKLC3IF9PyfLpp70SFf2xA6u/edit#slide=id.p1)

## [Used dataset (1000+ paired images)](https://drive.google.com/drive/u/0/folders/1I8upXHQT9WA9wT2T8k0IO6OiQajv_AB-)

### This repository contains two neural networks:

pix2pix generates an animal based on the skeleton

pix2pix_rev generates a skeleton based on an animal

### Explanation of the files in the folder:

crt_ftable- script for creating a training and validation table (*starts first)

data_utils- functions for generating training pairs and augmenting them

model- the model structure is described

tf_losses- the loss functions are described

train_model- the main script for training (*starts second)

*To increase the learning power, you need to replace the parameter GLOBAL_M = 32  with a larger value, for example 64


eval- the script demonstrates the operation of the neurowork

*For the network to work, you need to download the dataset to the folder


### software

Especially for these neural networks, a small application was made on a PC

## [link](https://drive.google.com/drive/u/0/folders/1PXJF2LKQo4GgF8Ba_iF4qwESzwcyU7i8)   

The program already contains the necessary weights to work with this version of the neural network

To launch the application, you need to open the file prog.exe and upload images. 

The "преобразовать в скелет" button inverts the neural network and allows you to generate an animal skeleton based on its image.

*If you have trained your neural network and want to use our program, you need to replace the weight files in prog/pix2pix

### Demonstration of the neural network operation:

![image](https://user-images.githubusercontent.com/59373737/120548940-a0634500-c3fb-11eb-9d91-dcd2c7beaeee.png)


![image](https://user-images.githubusercontent.com/59373737/120548989-abb67080-c3fb-11eb-8011-e9f825502320.png)


As a result of the training, the network has learned to generate images, but rather blurry and inaccurate. You can get similar results on some images:

![image](https://user-images.githubusercontent.com/59373737/120549128-d1dc1080-c3fb-11eb-85ed-253af139a973.png)

We believe that if you increase the power for training the neural network and increase the size of the dataset, you can improve the network performance and similar results will be generated less

<br><br><br>
------------------------------------------------------------------------------

## This project was developed as a course work on the subject "Machine learning and intelligent systems" by students of the 3rd year of the group IDB-18-09

