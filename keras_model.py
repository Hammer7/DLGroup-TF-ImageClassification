import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras_datagenerator import DataGenerator
from dataset import Dataset

class KerasModel():
    def __init__(self, dataset):
        self.dataset = dataset

        ki = tf.keras.initializers.HeNormal()
        bi = tf.keras.initializers.Ones()
        input = Input(shape=(128, 128, 3)) #shape=(None, None, 3)
        #VGG starts from 64 filter channel!
        s1_conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(input)
        s1_conv1 = BatchNormalization()(s1_conv1)
        s1_conv2 = Conv2D(32, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(s1_conv1)
        s1_conv2 = BatchNormalization()(s1_conv2)
        s1_pool = MaxPooling2D((2, 2), strides=(2, 2))(s1_conv2)

        s2_conv1 = Conv2D(64, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(s1_pool)
        s2_conv1 = BatchNormalization()(s2_conv1)
        s2_conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(s2_conv1)
        s2_conv2 = BatchNormalization()(s2_conv2)
        s2_pool = MaxPooling2D((2, 2), strides=(2, 2))(s2_conv2)

        s3_conv1 = Conv2D(128, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(s2_pool)
        s3_conv1 = BatchNormalization()(s3_conv1)
        s3_conv2 = Conv2D(128, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(s3_conv1)
        s3_conv2 = BatchNormalization()(s3_conv2)
        s3_conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(s3_conv2)
        s3_conv3 = BatchNormalization()(s3_conv3)
        s3_pool = MaxPooling2D((2, 2), strides=(2, 2))(s3_conv3)

        s4_conv1 = Conv2D(256, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(s3_pool)
        s4_conv1 = BatchNormalization()(s4_conv1)
        s4_conv2 = Conv2D(256, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(s4_conv1)
        s4_conv2 = BatchNormalization()(s4_conv2)
        s4_conv3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=ki, bias_initializer=bi, activation='relu')(s4_conv2)
        s4_conv3 = BatchNormalization()(s4_conv3)
        s4_pool = MaxPooling2D((2, 2), strides=(2, 2))(s4_conv3)

        #vgg has stage 5!
        flatten = Flatten()(s4_pool)
        #vgg use 4096
        dense1 = Dense(units=2048, kernel_initializer=ki, bias_initializer=bi, activation='relu')(flatten)
        dense1 = BatchNormalization()(dense1)
        dense2 = Dense(units=2048, kernel_initializer=ki, bias_initializer=bi, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        output = Dense(units=4, kernel_initializer=ki, bias_initializer=bi, activation='softmax')(dense2)
        self.model = Model(inputs=input, outputs=output)

        self.model.compile(optimizer=Adam(lr=1e-3), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        self.model.summary()

    def train(self, logdir):
        cp_path = os.path.join(logdir, 'models', 'checkpoints')
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        train_gen = DataGenerator(self.dataset, Dataset.TRAIN, 32)
        val_gen = DataGenerator(self.dataset, Dataset.VAL, 32)
        r = self.model.fit_generator(generator=train_gen,
                            validation_data=val_gen,
                            use_multiprocessing=False,
                            callbacks=[model_checkpoint],
                            epochs = 40,
                            shuffle=True,
                            verbose=1)

        self.model.load_weights(cp_path)

        return

    def predict_dataset(self):
        for subset, images in self.dataset.images_dict.items():
            labels = self.dataset.labels_dict[subset]
            oh_labels = to_categorical(labels, 4)
            norm_images = (images - self.dataset.mean) / self.dataset.stddev
            loss, acc = self.model.evaluate(norm_images, oh_labels, verbose=2)
            print(f'Evaluate: {subset} loss = {loss}, acc = {acc}')

            pred = self.model.predict(norm_images)
            pred = np.argmax(pred, axis=1)
            acc = np.mean(pred==labels)
            print(f'Predict: {subset} acc = {acc}')
            #Train = 0.93, Val = 0.93, Test = 0.89
