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

FRAME_SHAPE = [64,64]

# Init
env = gym.make('VizdoomTakeCover-v0')
dataset = dataset.Dataset(env,FRAME_SHAPE)

#dataset.create_new_dataset(temporary=False)
dataset.load_dataset()

vae = vae.VAE(FRAME_SHAPE,dataset)

vae.load_json()
vae.train_vae(checkpoint = True)
vae.save_json()


choosed_img = dataset.dataset[850]

imgplot = plt.imshow(choosed_img)
plt.show()

imgplot = plt.imshow(vae.synthesize_image(choosed_img))
plt.show()
