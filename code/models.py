"""
shazzzam cam
CS1430 - Computer Vision
Spring 2022
"""

from multiprocessing import pool
import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras import Sequential

import hyperparams as hp


class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
   
        pretrained_model = tf.keras.applications.ResNet50(include_top = False, 
                                                   input_shape=(hp.img_size,hp.img_size,3),
                                                   pooling='avg', classes=hp.num_classes,
                                                   weights='imagenet')
        for layer in pretrained_model.layers:
               layer.trainable = False

        self.resnet_model = Sequential(pretrained_model)

        self.resnet_model.add(Flatten())
        self.resnet_model.add(Dense(512, activation = 'relu'))
        self.resnet_model.add(Dense(15, activation='softmax'))


    def call(self, x):
        """ Passes input image through the network. """
        x = self.resnet_model(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):        
        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = cce(labels,predictions)
        return loss

class DenseNet(tf.keras.Model):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
   
        pretrained_model = tf.keras.applications.densenet.DenseNet169(include_top = False, 
                                                   input_shape=(hp.img_size,hp.img_size,3),
                                                   pooling='avg', classes=hp.num_classes,
                                                   weights='imagenet')
        for layer in pretrained_model.layers:
               layer.trainable = False

        self.densenet_model = Sequential(pretrained_model)

        self.densenet_model.add(Flatten())
        self.densenet_model.add(Dense(512, activation = 'relu'))
        self.densenet_model.add(Dense(15, activation='softmax'))


    def call(self, x):
        """ Passes input image through the network. """
        x = self.densenet_model(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):        
        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = cce(labels,predictions)
        return loss

class InceptionV3(tf.keras.Model):
    def __init__(self):
        super(InceptionV3, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
   
        pretrained_model = tf.keras.applications.inception_v3.InceptionV3(include_top = False, 
                                                   input_shape=(hp.img_size,hp.img_size,3),
                                                   pooling='avg', classes=hp.num_classes,
                                                   weights='imagenet')
        for layer in pretrained_model.layers:
               layer.trainable = False

        self.inceptionv3_model = Sequential(pretrained_model)

        self.inceptionv3_model.add(Flatten())
        self.inceptionv3_model.add(Dense(512, activation = 'relu'))
        self.inceptionv3_model.add(Dense(15, activation='softmax'))


    def call(self, x):
        """ Passes input image through the network. """
        x = self.inceptionv3_model(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):        
        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = cce(labels,predictions)
        return loss

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # Don't change the below:

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        
        for layer in self.vgg16:
               layer.trainable = False
        
        self.head = [
               Flatten(),
                Dense(units=15,activation='softmax')]

        # Don't change the below:
        self.vgg16 = Sequential(self.vgg16, name="vgg_base")
        self.head = Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = cce(labels,predictions)
        return loss
        pass
