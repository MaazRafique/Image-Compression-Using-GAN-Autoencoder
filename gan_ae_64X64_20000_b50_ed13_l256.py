#####################################
######## 64X64_20000_b50_ed13_l256
#######################################
from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


class AdversarialAutoencoder():
    def __init__(self):
        self.blocksize = 64
        self.img_rows = self.blocksize
        self.img_cols = self.blocksize
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.encoded_dim = 13
        optimizer = Adam(0.0002, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        # Build and compile the encoder / decoder
        self.encoder = self.build_encoder()
        self.encoder.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)
        self.decoder = self.build_decoder()
        self.decoder.compile(loss=['mse'],
            optimizer=optimizer)
        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)
        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)
        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)

    def build_encoder(self):
        # Encoder
        encoder = Sequential()
        encoder.add(Flatten(input_shape=self.img_shape))
        encoder.add(Dense(256))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(Dense(256))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(Dense(self.encoded_dim))
        encoder.summary()
        img = Input(shape=self.img_shape)
        encoded_repr = encoder(img)
        return Model(img, encoded_repr)

    def build_decoder(self):
        # Decoder
        decoder = Sequential()
        decoder.add(Dense(256, input_dim=self.encoded_dim))
        decoder.add(LeakyReLU(alpha=0.2))
        decoder.add(Dense(256))
        decoder.add(LeakyReLU(alpha=0.2))
        decoder.add(Dense(np.prod(self.img_shape), activation='tanh'))
        decoder.add(Reshape(self.img_shape))
        decoder.summary()
        encoded_repr = Input(shape=(self.encoded_dim,))
        gen_img = decoder(encoded_repr)
        return Model(encoded_repr, gen_img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.encoded_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()
        encoded_repr = Input(shape=(self.encoded_dim, ))
        validity = model(encoded_repr)
        return Model(encoded_repr, validity)

    def train(self, epochs, batch_size=50):
        # Load the dataset
        imagearray=[]
        # Load loca images from fodler     
        folderpath='drive/DATA/mobile_valid'
        filenames = os.listdir(folderpath)
        for filename in filenames:
          imagefile = Image.open(folderpath+'/'+filename)
          imagewidth,imageheight = imagefile.size
          widthvalue = imagewidth % self.blocksize
          heightvalue = imageheight % self.blocksize
          if widthvalue > 0:
            width = imagewidth + (self.blocksize-widthvalue)
          else:
            width = imagewidth
          if heightvalue > 0:
            height = imageheight + (self.blocksize-heightvalue)
          else:
            height = imageheight          
          imagefile = imagefile.resize((width, height), Image.ANTIALIAS)
          r,g,b = imagefile.split()
          imageR=np.array(r)
          imageG=np.array(g)
          imageB=np.array(b)
          for y in range(0, height, self.blocksize): # blocks size height
            for x in range(0, width, self.blocksize): # blocks size width
              imagearray.append(imageR[y:y+self.blocksize,x:x+self.blocksize])
          for y in range(0, height, self.blocksize): # blocks size height
            for x in range(0, width, self.blocksize): # blocks size width
              imagearray.append(imageG[y:y+self.blocksize,x:x+self.blocksize])
          for y in range(0, height, self.blocksize): # blocks size height
            for x in range(0, width, self.blocksize): # blocks size width
              imagearray.append(imageB[y:y+self.blocksize,x:x+self.blocksize])
        X_train = np.array(imagearray)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            latent_fake = self.encoder.predict(imgs)
            fake = np.zeros((batch_size, 1))
            # Train the discriminator
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = d_loss_fake
            valid_y = np.ones((batch_size, 1))
            # Train the generator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid_y])
            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
        self.save_model()

    def save_model(self):
        def save(model, model_name):
            model_path = "drive/RACE/%s.json" % model_name
            weights_path = "drive/RACE/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])
        save(self.encoder, "gan_ae_encoder_local")
        save(self.decoder, "gan_ae_decoder_local")

if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=20000, batch_size=50)    