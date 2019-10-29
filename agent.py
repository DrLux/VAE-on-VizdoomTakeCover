import numpy as np
import tensorflow as tf
tfd = tf.contrib.distributions
import dataset #my file
import vae #my file

latent_size = 8
learning_rate = 0.001
train_epochs = 100


dataset = dataset.Dataset('Freeway-v0',10000,[64,64],10)
vae = vae.VAE(latent_size,learning_rate,train_epochs)

#dataset.create_dataset()
#dataset.test_dataset()

vae.train_vae(dataset)

