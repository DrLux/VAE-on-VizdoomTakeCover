#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import json

class VAE(object):
    def __init__(self, latent_size, learning_rate,train_epochs,frame_shape,dataset):
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.frame_shape = frame_shape
        self.dataset = dataset
        self.build_computational_graph()
        
        

    def build_computational_graph(self):
        self.input_batch = tf.placeholder(tf.float32, shape=[None, self.frame_shape[0], self.frame_shape[1], 3])
        self.latent_mean,self.latent_std_dev = self.encoder(self.input_batch) 
        self.latent_vec = self.reparametrization_trick(self.latent_mean,self.latent_std_dev)
        self.output_batch = self.decoder(self.latent_vec)
        self.img_loss,self.kl_loss,self.loss,self.optimizer = self.define_optimizer(self.input_batch,self.output_batch,self.latent_mean,self.latent_std_dev)

        #operazione per realizzare il restore del modello
        t_vars = tf.trainable_variables() #restituisce la lista delle variabili interessate dall' addestramento
        self.assign_ops = {} #dizionario
        for var in t_vars:
            pshape = var.get_shape() #leggo la shape della variabile
            pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder') #creo un nuovo placeholder con la stessa shape della variabile
            assign_op = var.assign(pl) #mi memorizzo in una variabile l' operazione di assegnamento del placeholder alla variabile (in un secondo momento mi bastera fare il feed del placeholder per assegnare il valore alla variabile)
            self.assign_ops[var] = (assign_op, pl) #associo col dizionario la var all' op di assegnamento e al relativo placeholder 

        #Init Session
        init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(init_vars)

    # data shape:  (batch_size, self.train_epochs, frame_shape, 3)
    def encoder(self,data):
        with tf.variable_scope("Encoder"):
            input_layer = tf.layers.conv2d(inputs=data, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv1", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv1/Relu:0", shape=(batch_size, 31, 31, 32), dtype=float32)

            hidden_layer = tf.layers.conv2d(input_layer, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv2", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv2/Relu:0", shape=(batch_size, 14, 14, 64), dtype=float32)

            hidden_layer = tf.layers.conv2d(hidden_layer, filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv3", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv3/Relu:0", shape=(batch_size, 6, 6, 128), dtype=float32)

            hidden_layer = tf.layers.conv2d(hidden_layer, filters=256, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv4", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv4/Relu:0", shape=(1batch_size, 2, 2, 256), dtype=float32)

            last_layer = tf.layers.flatten(hidden_layer) # same effect of tf.reshape(hidden_layer, [-1, 2*2*256])        
            #Tensor("Encoder/flatten/Reshape:0", shape=(batch_size, 1024), dtype=float32)

            
            # Latent Vector: mean and st_dev
            latent_mean = tf.layers.dense(last_layer, units=self.latent_size, name='mean', reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)

            
            # std_dev mus be always > 0, so we use softplus to force it
            #latent_std_dev = tf.nn.softplus(tf.layers.dense(last_layer, units=self.latent_size, reuse=tf.AUTO_REUSE), name='std_dev',)  
            #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)

            #PROVO A RIFARE LA st_dev
            latent_std_dev = tf.layers.dense(last_layer, units=self.latent_size, reuse=tf.AUTO_REUSE)
            sigma = tf.exp(latent_std_dev / 2.0)
            latent_std_dev = sigma

        return latent_mean,latent_std_dev


    def reparametrization_trick(self,latent_mean,latent_std_dev):
        # Reparametrization trick
        # allocate a vector of random epsilon for rep.trick
        epsilon = tf.random_normal(tf.shape(latent_std_dev), name='epsilon')
        #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)

        # Sample a vector z by mean + (str_dev * epsilon)
        z = latent_mean + tf.multiply(epsilon, latent_std_dev)
        #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)

        return z

    def decoder(self,latent_vec):
        with tf.variable_scope("Decoder"):
            input_layer = tf.layers.dense(latent_vec, 4*256, name="dec_input_fullycon",reuse=tf.AUTO_REUSE)
            #Tensor("Decoder/dec_input_fullycon/BiasAdd:0", shape=(10, 1024), dtype=float32)
            
            input_layer = tf.reshape(input_layer, [-1, 1, 1, 4*256])
            #Tensor("Decoder/Reshape:0", shape=(10, 1, 1, 1024), dtype=float32)

            hidden_layer = tf.layers.conv2d_transpose(input_layer, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1",reuse=tf.AUTO_REUSE)
            #Tensor("Decoder/dec_deconv1/Relu:0", shape=(10, 5, 5, 128), dtype=float32)    

            hidden_layer = tf.layers.conv2d_transpose(hidden_layer, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2",reuse=tf.AUTO_REUSE)
            #Tensor("Decoder/dec_deconv2/Relu:0", shape=(10, 13, 13, 64), dtype=float32)

            hidden_layer = tf.layers.conv2d_transpose(hidden_layer, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3",reuse=tf.AUTO_REUSE)
            #Tensor("Decoder/dec_deconv3/Relu:0", shape=(10, 30, 30, 32), dtype=float32)

            # NB: we use a sigmoid function so the output values will be between 0 and 1 
            output_batch = tf.layers.conv2d_transpose(hidden_layer, 3, 6, strides=2, activation=tf.nn.sigmoid, name="reconstructor",reuse=tf.AUTO_REUSE)  
            #output_batch:  Tensor("Decoder/reconstructor/Sigmoid:0", shape=(batch_size, 64, 64, 3), dtype=float32)

        return output_batch
   
    def define_optimizer(self,input_batch,output_batch,latent_mean,latent_std_dev):
        with tf.name_scope('loss'):
            # reconstruction loss (MEAN SQUARE ERROR between 2 images):  
            img_loss = tf.reduce_mean(tf.reduce_sum(tf.square(input_batch - output_batch),reduction_indices = [1,2,3]))
                
            # kl loss for two gaussian
            # formula: https://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
            #kl_loss = - 0.5 * tf.reduce_sum((1 + tf.log(tf.square(latent_std_dev)) - tf.square(latent_mean) - tf.square(latent_std_dev)), reduction_indices = 1)

            ###kl-loss from hardmaru "world_model" repo
            kl_loss = - 0.5 * tf.reduce_sum((1 + latent_std_dev - tf.square(latent_mean) - tf.exp(latent_std_dev)),reduction_indices = 1)          
            kl_loss = tf.reduce_mean(kl_loss)

            loss = img_loss + kl_loss
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
        return img_loss,kl_loss,loss,optimizer

    def train_vae(self):
        for epoch in range(self.train_epochs):
            print("\n Epoch: ", epoch)
            batchs = self.dataset.get_batchs()
            for b in batchs:
                #loss_value,kl_loss_value, img_loss_value,_ = sess.run([loss, kl_loss, img_loss, optimizer], {input_batch: b})
                img_loss_value,kl_loss_value,loss_value,_ = self.sess.run([self.img_loss,self.kl_loss,self.loss,self.optimizer], {self.input_batch: b})
                
                print("loss_value: ", loss_value)
                #print("kl_loss_value: ", kl_loss_value)
                #print("img_loss_value: ", img_loss_value)

    def get_encoded_vec(self,img):
        input_frame = img.reshape(1, 64, 64, 3)
        norm_frame_input = np.float32(input_frame.astype(np.float)/255.0)
        
        placeholder_input = tf.placeholder(tf.float32, shape=[None, self.frame_shape[0], self.frame_shape[1], 3])
        latent_mean,latent_std_dev = self.encoder(placeholder_input) 
        latent_vec = self.reparametrization_trick(latent_mean,latent_std_dev)
        latent_vec = self.sess.run(latent_vec, {placeholder_input: norm_frame_input})
        return latent_vec

    def decode_latent_vec(self,latent_v):
        output_batch = self.decoder(latent_v)
        reconstructed_frames = self.sess.run(output_batch, {self.latent_vec: latent_v})
        return reconstructed_frames

    def synthesize_image(self,img):
        latent_vec = self.get_encoded_vec(img)
        reconstructed_img = self.decode_latent_vec(latent_vec)
        return reconstructed_img[0] #first dimension is the batch size 


    #######################################################################
    # codice per il dump del modello preso da hardmaru
    # https://github.com/hardmaru/WorldModelsExperiments/blob/244f79c2aaddd6ef994d155cd36b34b6d907dcfe/doomrnn/doomrnn.py#L76

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        t_vars = tf.trainable_variables()
        for var in t_vars:
            param_name = var.name
            p = self.sess.run(var)
            model_names.append(param_name) #in realtà non le uso
            params = np.round(p*10000).astype(np.int).tolist()
            model_params.append(params)
            model_shapes.append(p.shape) #in realtà non le uso
        return model_params, model_shapes, model_names


    def save_json(self, jsonfile='models/vae.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def load_json(self, jsonfile='models/vae.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)

    def set_model_params(self, params):
        t_vars = tf.trainable_variables()
        idx = 0
        for var in t_vars:
            pshape = tuple(var.get_shape().as_list()) #leggo la shape della variabile
            p = np.array(params[idx]) #carica il parametro che voglio restorare
            assert pshape == p.shape, "inconsistent shape"
            assign_op, pl = self.assign_ops[var] #ricarico dal dizionario l'op di assegnamento e il placeholder
            self.sess.run(assign_op, feed_dict={pl.name: p/10000.}) #assegno il placeholder alla var e lo riempio col valore del param letto 
            idx += 1