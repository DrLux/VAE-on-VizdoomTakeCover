import numpy as np
import tensorflow as tf
import gym
from PIL import Image as PilImage
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
import pickle
import os
from scipy.misc import imresize as resize

#Dataset hyperparameters
DATASET_SIZE = 1000
BATCH_SIZE = 100


class Dataset(object):
    
    def __init__(self,env,frame_shape):
        self.env = env
        self.dataset_size = DATASET_SIZE
        self.frame_shape = frame_shape
        self.batch_size = BATCH_SIZE
        
        self.dataset = None
        self.num_batches = 0

    #Crop image before store to dataset 
    #same preprocessing from orginal world_model source
    def preprocess_frame(self,obs):
        obs = np.array(obs[0:400, :, :]).astype(np.float)/255.0
        obs = np.array(resize(obs, (self.frame_shape[0], self.frame_shape[1])))
        obs = ((1.0 - obs) * 255).round().astype(np.uint8)
        return obs
        

    def create_new_dataset(self, temporary = True):
        frames = []
        obs = self.env.reset()
        for itr in range(self.dataset_size):
            self.env.render()
            obs,rew,done,_ = self.env.step(self.env.action_space.sample()) # take a random action 
            frames.append(self.preprocess_frame(obs))
            if done:
                self.env.reset()
            
            if itr % 1000 == 0:
                print("Dumped ", itr, " frames!")
        
        #REMEMBER: not finish yet with env. This is a temporary line
        self.env.close() 

        #from len = 1000 to shape = (1000, 64, 64, 3)
        self.dataset = np.stack(frames)

        #dump created dataset into picle file
        if not temporary:
            with open('dataset/new_dataset.pickle', 'wb') as output:
                pickle.dump(self.dataset, output)

        print("Dataset size: ",self.dataset.shape)
        self.num_batches = int(np.floor(self.dataset.shape[0]/self.batch_size))

        
    def load_dataset(self):
        pickle_dump_name = os.listdir('dataset')
        
        #check if a pickle dump is present
        assert len(pickle_dump_name) > 0, "Dataset folder is empty!"
        dataset_collection = []    

        # Load all dataset in folder
        for dset_name in pickle_dump_name:
            with open("dataset/"+dset_name, 'rb') as data:
                temp_dataset = pickle.load(data)
                dataset_collection.append(temp_dataset)
                
                print("Dataset ", dset_name," loaded")
                print("Dataset ", dset_name, " size: ",temp_dataset.shape)
        
        self.dataset = dataset_collection[0]
        
        #combine together all loaded datasets
        for i in range(1,len(dataset_collection)):
            self.dataset = np.append(self.dataset,dataset_collection[i],axis=0)
        print("Final Dataset size: ",self.dataset.shape)
        self.num_batches = int(np.floor(self.dataset.shape[0]/self.batch_size))
        
            

    # split dataset into batches
    # return list of batches
    def get_batches(self):
        batches = []
        np.random.shuffle(self.dataset)
        for idx in range(self.num_batches):
            data = self.dataset[idx*self.batch_size:(idx+1)*self.batch_size]
            batches.append(data.astype(np.float)/255.0)
        return batches


        
