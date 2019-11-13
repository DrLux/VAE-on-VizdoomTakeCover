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

# Init
env = gym.make('VizdoomTakeCover-v0')
dataset = dataset.Dataset(env,frame_shape)

#dataset.create_new_dataset()
dataset.load_dataset()

vae = vae.VAE(frame_shape,dataset)

vae.load_json()
vae.train_vae(checkpoint = False)
vae.save_json()


choosed_img = dataset.dataset[850]

imgplot = plt.imshow(choosed_img)
plt.show()

imgplot = plt.imshow(vae.synthesize_image(choosed_img))
plt.show()
