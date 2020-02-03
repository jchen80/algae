import glob
import numpy as np
import imageio
import os.path as path
from scipy import misc
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical 

# Importing images for training, testing, and validation
file_paths =  glob.glob('/Users/juliechen/documents/algae/algModPics/*')
imgs = [imageio.imread(path) for path in file_paths]
imgs = np.asarray(imgs)
print ("shape before resizing: " + str(imgs[0].shape))
image_list=[]
for img in imgs:
    img = misc.imresize(img, (75, 75, 3))
    image_list.append(img)
image_tuple=tuple(image_list)
stacked_imgs=np.stack(image_tuple, axis=0)
print ("shape after resizing:" + str(stacked_imgs.shape))

num_images=imgs.shape[0]
labels= np.zeros(num_images)
for n in range(num_images):
    fname = path.basename(file_paths[n])[0]
    labels[n]=fname

# Split 10% of the images and labels to testing data
train_X, test_X, train_Y, test_Y = train_test_split(stacked_imgs, labels, test_size=0.1, random_state=78)

print ("shape of training data images:" +str(train_X.shape))
print ("shape of test data images:" + str(test_X.shape))

#Change the labels from categorical to one-hot encoding
#train_Y = to_categorical(train_Y)
#test_Y = to_categorical(test_Y)

#Split 10% of training into validation data
train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=28)


#Load realPredict image data -- no labels
"""
This section loads real images which were used during the project to test the ability of the CNN model to accurately classify images taken by a drone over algae simulated water surfaces. Users can set up their real test images under the appropriate path directory and use the built in class function realTest to have the model classify the image according to risk of algae contamination.
"""
#file_paths =  glob.glob(r'C:\Users\algae\Pictures\Testing Image Data\Run1\Water1\*')
#rImgs = [misc.imread(path) for path in file_paths]
#rImgs = np.asarray(rImgs)

#rImage_list=[]
#for rImg in rImgs:
#    rImg = misc.imresize(rImg, (75, 75, 3))
#    rImage_list.append(rImg)
#rImage_tuple=tuple(rImage_list)
#rStacked_imgs=np.stack(rImage_list, axis=0)

#real_X= rStacked_imgs

#Constructs and compiles CNN network model
import tensorflow
from tensorflow.python import keras
from keras.models import Sequential, Input, Model, load_model
from keras import optimizers, losses
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.applications import VGG16

class CNN_Model:
    def __init__(self, train_X, train_Y, test_X, test_Y, valid_X, valid_Y, b_size, ep, num_classes, in_shape):
        self.train_X=train_X
        self.train_Y=train_Y
        self.test_X=test_X
        self.test_Y= test_Y
        self.valid_X=valid_X
        self.valid_Y=valid_Y
        self.b_size=b_size
        self.ep=ep
        self.num_classes=num_classes
        self.in_shape=in_shape
        self.model =  Sequential()
    def construct(self):
        self.model.add(Conv2D(32, kernel_size=(8, 8), activation = 'linear', input_shape=self.in_shape, padding ='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D((2, 2), padding='same'))
        self.model.add (Conv2D (64, (8, 8), activation ='linear', padding ='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2,2), padding= 'same'))
        self.model.add(Conv2D(128, (8,8), activation='linear', padding ='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='linear'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dense(self.num_classes, activation='softmax'))                                                                                                                                           
    def train(self):
        #Training the model
        self.model.compile(loss= 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        train = self.model.fit(self.train_X, self.train_Y, batch_size=self.b_size,epochs=self.ep,verbose=1,validation_data=(self.valid_X, self.valid_Y))
    def save(self, name):
        self.model.save(name)
    def load(self, name):
        self.model =load_model(name)
    def test(self):
        test_eval = self.model.evaluate(self.test_X, self.test_Y, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])                                            
    def manualCheck(self, test_num):
        prediction= (self.model.predict(self.test_X, batch_size=self.b_size, verbose=1, steps=None))
        print (prediction[0: test_num ,0: self.num_classes])
        print (test_Y[0: test_num ,0: self.num_classes])
    def realTest(self, real_X):
        realPrediction= (self.model.predict(real_X, batch_size=self.b_size, verbose=1, steps=None))
        print (realPrediction)

class Transfer_CNN_Model(CNN_Model):
    def __init__(self, train_X, train_Y, test_X, test_Y, valid_X, valid_Y, b_size, ep, num_classes, in_shape):
        CNN_Model.__init__(self, train_X, train_Y, test_X, test_Y, valid_X, valid_Y, b_size, ep, num_classes, in_shape)
        self.mod =  VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=num_classes)
    def construct(self):
        for layer in self.mod.layers[:-4]:
            layer.trainable= False
        self.model.add(self.mod)
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
    def train(self):
    #Training the model
        self.model.compile(loss= 'sparse_categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.00001), metrics=['accuracy'])
        train = self.model.fit(self.train_X, self.train_Y, batch_size=self.b_size,epochs=self.ep,verbose=1,validation_data=(self.valid_X, self.valid_Y))
    def save(self, name):
        self.model.save(name)
    def load(self, name):
        self.model =load_model(name)
    def test(self):
        test_eval = self.model.evaluate(self.test_X, self.test_Y, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])
    def manualCheck(self, test_num):
    #Runs a manual check of model on a select number of photos to allow user to verify model for demonstration purposes.
        prediction= (self.model.predict(self.test_X, batch_size=self.b_size, verbose=1, steps=None))
        test_Y1=to_categorical (self.test_Y)
        print ("Predicted likelihood of algae contamination for a manual check of "+str(test_num)+" sample images:")
        print ("[P(algae), P(non-algae)]")
        print (prediction[0: test_num ,0: self.num_classes])
        #This next line is modified because the labels were not one-hot encoded in with the algae dataset
        print ("True labels for the manual check:")
        print (test_Y1[0: test_num])
    def realTest(self, real_X):
        realPrediction= (self.model.predict(real_X, batch_size=self.b_size, verbose=1, steps=None))
        for pred in realPrediction:
            if pred[1]>= pred[0] *4:
                print ('High Risk Algae')
            elif pred[1]>=pred[0]*2:
                print ('Moderate High Risk Algae')
            elif pred[1]>=pred[0]:
                print ('Moderate Risk Algae')
            elif pred[1]>=pred[0]*0.5:
                print ('Moderate Low Risk Algae')
            else:
                print ('Low Risk Algae')
        print (realPrediction)

algMod= Transfer_CNN_Model(train_X, train_Y, test_X, test_Y, valid_X, valid_Y, 64, 16, 2, (75, 75, 3))
algMod.construct()
algMod.model.summary()
#algMod.train()
#algMod.save('algMod.h5py')
algMod.load('algMod.h5py')
algMod.test()
algMod.manualCheck(10)
#algMod.realTest(real_X)


