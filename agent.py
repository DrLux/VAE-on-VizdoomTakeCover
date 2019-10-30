import numpy as np
import tensorflow as tf
tfd = tf.contrib.distributions
import dataset #my file
import vae #my file

#Hyperparameters

#Dataset
env_id = 'Freeway-v0'
dataset_size = 1000
frame_shape = [64,64]
batch_size = 10

#VAE
latent_size = 8
learning_rate = 0.001
train_epochs = 10

num_batches = int(dataset_size/batch_size)
print("*************num_batches: ", num_batches)

dataset = dataset.Dataset(env_id,dataset_size,frame_shape,batch_size)
vae = vae.VAE(latent_size,learning_rate,train_epochs,frame_shape,batch_size,num_batches)
vae.train_vae(dataset)
#vae.sample_random_image()
