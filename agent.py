import numpy as np
import gym
import dataset #my file
import vae #my file
import matplotlib.pyplot as plt
import vizdoomgym

'''
before run:
git clone https://github.com/simontudo/vizdoomgym.git
cd vizdoomgym
pip install -e .
'''

#Dataset
#env_id = 'Freeway-v0'
dataset_size = 100000
frame_shape = [64,64]
batch_size = 100

#VAE
latent_size = 64
learning_rate = 0.0001
train_epochs = 3

#Plot
img_to_plot = 5


# Instanziazione
env = gym.make('VizdoomTakeCover-v0')
dataset = dataset.Dataset(env,dataset_size,frame_shape,batch_size)
vae = vae.VAE(latent_size, learning_rate,train_epochs,frame_shape,dataset)
vae.train_vae()
print("fine addestramento vae")
#vae.save_json()

#test vae
vae.load_json()

choosed_img = dataset.dataset[5]
latent_v = vae.get_encoded_vec(choosed_img)
recontructed_img = vae.decode_latent_vec(latent_v)
recontructed_img = np.round(recontructed_img * 255.).astype(np.uint8)
recontructed_img = recontructed_img.reshape(64, 64, 3)

imgplot = plt.imshow(choosed_img)
plt.show()

imgplot = plt.imshow(recontructed_img)
plt.show()
