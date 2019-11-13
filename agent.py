import numpy as np
import gym
import dataset #my file
import vae #my file
import matplotlib.pyplot as plt
import vizdoomgym


'''
before run install vizdoom:
    git clone https://github.com/simontudo/vizdoomgym.git
    cd vizdoomgym
    pip install -e .
'''

#HyperParameters
#Dataset
dataset_size = 1000
frame_shape = [64,64]
batch_size = 100

#VAE
latent_size = 64
learning_rate = 0.00005
train_epochs = 3


# Init
env = gym.make('VizdoomTakeCover-v0')
dataset = dataset.Dataset(env,dataset_size,frame_shape,batch_size)

#dataset.create_new_dataset()
dataset.load_dataset()

vae = vae.VAE(latent_size, learning_rate,train_epochs,frame_shape,dataset)

vae.load_json()
vae.train_vae(checkpoint = False)
vae.save_json()


choosed_img = dataset.dataset[850]

imgplot = plt.imshow(choosed_img)
plt.show()

imgplot = plt.imshow(vae.synthesize_image(choosed_img))
plt.show()
