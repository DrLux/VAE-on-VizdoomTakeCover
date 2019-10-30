#https://towardsdatascience.com/vaes-generating-images-with-tensorflow-61de08e82f1f

import numpy as np
import tensorflow as tf
import gym
from PIL import Image as PilImage
tfd = tf.contrib.distributions
import matplotlib.pyplot as plt 

class Dataset(object):
    
    def __init__(self,env_id,db_size,frame_shape,batch_size):
        self.env_id = env_id # Name env on open ai gym
        self.size_dataset = db_size # number of frames to collect
        self.batch_size = batch_size # batch size
        self.frame_shape = frame_shape  # final shape for each frame [64,64]

        self.env = gym.make(env_id)
        self.frames = [] # list of frames to collect

        self.iterator = None
        self.input_batch = None

        self.create_dataset()
        
    
    def create_dataset(self):
        obs = self.env.reset()
        for itr in range(self.size_dataset):
            obs,rew,done,_ = self.env.step(self.env.action_space.sample()) # take a random action
            obs = obs[25:195, 10:, :]# crop image
            obs = np.array(PilImage.fromarray(obs,'RGB').resize((self.frame_shape[0],self.frame_shape[1]))) #resize img
            self.frames.append(obs)
        self.env.close()
        self.frames = np.stack(self.frames)
       
        #turn list of frames to tensorflow dataset type
        with tf.variable_scope("DataPipe"):
            #create a dataset whose elements are the frames collected
            dataset = tf.data.Dataset.from_tensor_slices(self.frames)
            #Convert each image of dataset to dtype
            dataset = dataset.map(lambda x: tf.image.convert_image_dtype([x], dtype=tf.float32))
            #split dataset in batches
            dataset = dataset.batch(batch_size=self.batch_size).prefetch(self.batch_size)
            
            # create an iterator over the dataset
            self.iterator = dataset.make_initializable_iterator()
            #get a batch of data
            self.input_batch = self.iterator.get_next()
            # reshape from (?, 1, frame_shape[0], frame_shape[1], 3) to (?, , frame_shape[0], frame_shape[1], 3)
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