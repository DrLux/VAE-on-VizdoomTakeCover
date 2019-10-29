import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 


from PIL import Image as PilImage



class VAE(object):
    def __init__(self, latent_size, learning_rate,train_epochs):
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs

    '''
    z:  Tensor("Encoder/add:0", shape=(batch_size, latent_size), dtype=float32)
    mean:  Tensor("Encoder/mean/BiasAdd:0", shape=(batch_size, latent_size), dtype=float32)
    std_dev:  Tensor("Encoder/std_dev:0", shape=(batch_size, latent_size), dtype=float32)
    '''
    def encoder(self,data):
        print("data shape: ", data.shape)
        #data shape:  (10, 64, 64, 3)
        batch_size = data.shape[0]
        with tf.variable_scope("Encoder"):
            input_layer = tf.layers.conv2d(inputs=data, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv1", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv1/Relu:0", shape=(batch_size, 31, 31, 32), dtype=float32)
            
            hidden_layer = tf.layers.conv2d(input_layer, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv2", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv2/Relu:0", shape=(batch_size, 14, 14, 64), dtype=float32)
            
            hidden_layer = tf.layers.conv2d(hidden_layer, filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv3", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv3/Relu:0", shape=(batch_size, 6, 6, 128), dtype=float32)

            hidden_layer = tf.layers.conv2d(hidden_layer, filters=256, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv4", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv4/Relu:0", shape=(1batch_size, 2, 2, 256), dtype=float32)
            
            last_layer = tf.layers.flatten(hidden_layer) # == tf.reshape(hidden_layer, [-1, 2*2*256])
            #Tensor("Encoder/flatten/Reshape:0", shape=(batch_size, 1024), dtype=float32)

            # Local latent variables
            self.mean_ = tf.layers.dense(last_layer, units=self.latent_size, name='mean')
            #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)
            self.std_dev = tf.nn.softplus(tf.layers.dense(last_layer, units=self.latent_size), name='std_dev')  # softplus to force >0
            #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)

            # Reparametrization trick
            # il -1 è un placeholder per "batch size" --> crea PROBLEMI qui
            epsilon = tf.random_normal([10, self.latent_size], name='epsilon')
            #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)
            self.z = self.mean_ + tf.multiply(epsilon, self.std_dev)
            #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)

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

            img = tf.layers.conv2d_transpose(hidden_layer, 3, 6, strides=2, activation=tf.nn.sigmoid, name="reconstructor",reuse=tf.AUTO_REUSE)  
            #img:  Tensor("Decoder/reconstructor/Sigmoid:0", shape=(batch_size, 64, 64, 3), dtype=float32)
            
            return img

    def define_optimizer(self):
        # Link encoder and decoer
        self.input_batch = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        
        self.encoder(self.input_batch)
        self.output_batch = self.decoder(self.z)
  
        with tf.name_scope('loss'):
            kl_tolerance=0.5

            # reconstruction loss (MEAN SQUARE ERROR):  
            img_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_batch - self.output_batch),reduction_indices = [1,2,3]))
             
            #binary cross entropy Version
            #asssume che gli input siano 0 e 1, quindi funziona solo su bernulli (casi bianco e nero) 
            #formula: https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
            #img_loss = tf.reduce_mean(tf.reduce_sum(self.input_batch * - tf.math.log(self.output_batch) + (1 - self.input_batch) * - tf.math.log(1 - self.output_batch), reduction_indices = 1))


            # augmented kl loss per dim
            # formula: https://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
            kl_loss = - 0.5 * tf.reduce_sum((1 + tf.log(tf.square(self.std_dev)) - tf.square(self.mean_) - tf.square(self.std_dev)), reduction_indices = 1)

            kl_loss = tf.maximum(kl_loss, kl_tolerance * self.latent_size)
            kl_loss = tf.reduce_mean(kl_loss)
        
            self.loss = img_loss + kl_loss
            
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train_vae(self,dataset):
        self.define_optimizer()

        init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
        gpu_options = tf.GPUOptions(allow_growth=True)

        # Training loop
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            writer = tf.summary.FileWriter('./logs', sess.graph)

            sess.run(init_vars)
            #merged_summary_op = tf.summary.merge_all()

            sess.run(dataset.iterator.initializer)
            #print('Actual epoch: {}'.format(epoch))

            for itr in range(self.train_epochs):
                print_loss,_ = sess.run([self.loss, self.optimizer], {self.input_batch: dataset.get_single_batch(),})
                print("Loss: ", print_loss)
                print("######################## Iteration ", itr, " ########################")


                #sess.run(self.optimizer, feed_dict={self.input_batch: dataset.get_single_batch()})
                #summ, target, output_ = sess.run([merged_summary_op, dataset.input_batch, output_batch])
