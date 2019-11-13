import numpy as np
import tensorflow as tf
import gym
from PIL import Image as PilImage
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
import pickle
import os
from scipy.misc import imresize as resize


class Dataset(object):
    
    def __init__(self,env,dataset_size,frame_shape,batch_size):
        self.env = env
        self.dataset_size = dataset_size
        self.frame_shape = frame_shape
        self.batch_size = batch_size
        self.dataset = None

        self.make_dataset()
        self.num_batchs = int(np.floor(dataset_size/batch_size))
        #imgplot = plt.imshow(self.dataset[3])
        #plt.show()
    
    def make_dataset(self):
        pickle_dump_name = os.listdir('dataset')
        
        #check if a pickle dump already exists. If not create it
        if pickle_dump_name:
            dataset_collection = []    
            for dset_name in pickle_dump_name:
                with open("dataset/"+dset_name, 'rb') as data:
                    temp_dataset = pickle.load(data)
                    dataset_collection.append(temp_dataset)
                    
                    print("Dataset ", dset_name," loaded")
                    print("Dataset ", dset_name, " size: ",temp_dataset.shape)
            
            self.dataset = dataset_collection[0]
            for i in range(1,len(dataset_collection)):
                self.dataset = np.append(self.dataset,dataset_collection[i],axis=0)
            print("Final Dataset size: ",self.dataset.shape)
        else:                
            frames = []
            obs = self.env.reset()
            for itr in range(self.dataset_size):
                self.env.render()
                obs,rew,done,_ = self.env.step(self.env.action_space.sample()) # take a random action
                
                obs = np.array(obs[0:400, :, :]).astype(np.float)/255.0
                obs = np.array(resize(obs, (self.frame_shape[0], self.frame_shape[1])))
                obs = ((1.0 - obs) * 255).round().astype(np.uint8)
                
                frames.append(obs)
                if done:
                    self.env.reset()
                if itr % 1000 == 0:
                    print("Dumped ", itr, " frames!")
            self.env.close() #non ho finito con l' env

            #from len = 1000 to shape = (1000, 64, 64, 3)
            self.dataset = np.stack(frames)

            #with open('dataset/dump_frames.pickle', 'wb') as output:
            #    pickle.dump(self.dataset, output)

            print("Dataset size: ",self.dataset.shape)

    # split dataset into batches
    def get_batchs(self):
        batches = []
        np.random.shuffle(self.dataset)
        for idx in range(self.num_batchs):
            data = self.dataset[idx*self.batch_size:(idx+1)*self.batch_size]
            batches.append(data.astype(np.float)/255.0)
        return batches


        
