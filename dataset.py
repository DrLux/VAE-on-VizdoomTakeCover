#https://towardsdatascience.com/vaes-generating-images-with-tensorflow-61de08e82f1f

import numpy as np
import tensorflow as tf
import gym
from PIL import Image as PilImage
tfd = tf.contrib.distributions
import matplotlib.pyplot as plt 

class Dataset(object):
    def __init__(self, env_id,db_size,frame_shape,batch_size):
        self.env_id = env_id # Name env
        self.size_database = db_size # number of frames to collect
        self.batch_size = batch_size # batch size
        self.frame_shape = frame_shape  # 64,64

        self.env = gym.make('Freeway-v0')
        self.frames = []

        self.iterator = None
        self.input_batch = None

        self.create_dataset()
        
    def create_dataset(self):
        obs = self.env.reset()
        for itr in range(self.size_database):
            obs,rew,done,_ = self.env.step(self.env.action_space.sample()) # take a random action
            obs = obs[25:195, 10:, :].astype(np.float)/255.0
            obs = ((1.0 - obs) * 255).round().astype(np.uint8)
            obs = np.array(PilImage.fromarray(obs,'RGB').resize((self.frame_shape[0],self.frame_shape[1])))
            self.frames.append(obs)
        self.env.close()
        self.frames = np.stack(self.frames)
        print("######################## DATABASE CREATO ########################")

        with tf.variable_scope("DataPipe"):
            dataset = tf.data.Dataset.from_tensor_slices(self.frames)
            dataset = dataset.map(lambda x: tf.image.convert_image_dtype([x], dtype=tf.float32))
            dataset = dataset.batch(batch_size=self.batch_size).prefetch(self.batch_size)

            self.iterator = dataset.make_initializable_iterator()
            self.input_batch = self.iterator.get_next()
            self.input_batch = tf.reshape(self.input_batch, shape=[-1, self.frame_shape[0], self.frame_shape[1], 3])
    


    def test_dataset(self):
        init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
        with tf.Session() as sess:
            sess.run([init_vars, self.iterator.initializer]) # Initialize variables and the iterator

            while 1:    # Iterate until we get out of range error!
                try:
                    batch = sess.run(self.input_batch)
                    print(batch.shape)  # Get batch dimensions
                    plt.imshow(batch[0,:,:,0])
                    plt.show()
                except tf.errors.OutOfRangeError:  # This exception is triggered when all batches are iterated
                    print('All batches have been iterated!')
                    break


    def get_single_batch(self):
        init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
        with tf.Session() as sess:
            sess.run([init_vars, self.iterator.initializer]) # Initialize variables and the iterator
            try:
                batch = sess.run(self.input_batch)
            except tf.errors.OutOfRangeError:  # This exception is triggered when all batches are iterated
                print('All batches have been iterated!')
        return batch

    