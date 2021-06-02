# Skeletons & Animals

Software that can generate a photo of an animal based on its skeleton and a photo of an animal's skeleton based on its animal

# Developers students: 
Fedosenko Maxim

Vasilenko Vladislav

# Project presentation: https://docs.google.com/presentation/d/1u2XB4frWrKLC3IF9PyfLpp70SFf2xA6u/edit#slide=id.p1
# Used dataset (1000+ paired images): https://drive.google.com/drive/u/0/folders/1I8upXHQT9WA9wT2T8k0IO6OiQajv_AB-

# This repository contains two neural networks:
pix2pix generates an animal based on the skeleton

pix2pix_rev generates a skeleton based on an animal

# Explanation of the files in the folder:
crt_ftable- script for creating a training and validation table (*starts first)
data_utils- functions for generating training pairs and augmenting them
model- the model structure is described
tf_losses- the loss functions are described
train_model- the main script for training (*starts second)
eval- the script demonstrates the operation of the neurowork

*For the network to work, you need to download the dataset to the folder
