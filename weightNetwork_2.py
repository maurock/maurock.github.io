from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
import tensorflow as tf
import keras.optimizers
import numpy as np
import cv2
import os
import parser
import matplotlib
from PIL import Image
from keras import backend as K
from keras.utils import np_utils
from sklearn.utils import shuffle
import sklearn.cross_validation
K.clear_session()
'''
Define VGG16 function
'''
def VGG_16(shape_output, weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),data_format='channels_last', input_shape=(224,224,3)))
    model.add(Convolution2D(64, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(shape_output, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def create_dict(a):
    b = np.arange(0, np.unique(a).size, 1)
    c = np.unique(sorted(a))
    new_dict = dict(zip(c, b))
    return new_dict

def reduceLabels(a):
    new_dict = create_dict(a)
    return np.array(list(map(new_dict.get, a)))

def expandLabels(array, num):
    new_dict = create_dict(array)
    for key, value in new_dict.items():
        if value == num:
            return key


'''
#Load images in a numpy array


path_input = 'input_data\\'
arrayImg = []

for subdirect in os.listdir(path_input):
    img = cv2.imread(path_input+subdirect)
    img = cv2.resize(img, (224, 224))
    img = image.img_to_array(img)
    #img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    arrayImg.append(img)
#print(arrayImg[0])
arrayImg = np.array(arrayImg)

'''
#Create label array
'''

categories = os.listdir("C:\\Users\\ZINKCLOUD07\\PycharmProjects\\TextImageNetwork\\ImagesCifar10\\")
categories_array = np.array(categories)

file = open("labels.txt", "w")
for subdirect in os.listdir("C:\\Users\\ZINKCLOUD07\\PycharmProjects\\TextImageNetwork\\input_data\\"):
    splitLabel = subdirect.split('_')
    string = splitLabel[0]
    file.write(string+"\n")
file.close()


file = open("labels.txt", "r")
labels = []
for text in file:
    text=text[:-1]
    labels.append(categories.index(text))

labels = np.array(labels)
reduced_labels = reduceLabels(labels)
print(reduced_labels)

#Shuffle all together for validation set
arrayImg, reduced_labels, labels = shuffle(arrayImg, reduced_labels, labels, random_state= 313)
np.save('c10_images.npy', arrayImg)
np.save('c10_reduced_labels.npy', reduced_labels)
np.save('c10_labels.npy', labels)
'''

arrayImg = np.load('c10_images.npy')
reduced_labels =np.load('c10_reduced_labels.npy')
labels = np.load('c10_labels.npy')


'''
Fit model to training set
'''
print('Categories in output: ' + np.str(len(np.unique(labels))))

shape_output = len(np.unique(labels))
model = VGG_16(shape_output=shape_output, weights_path='C101_test4_augm.h5')

lr_init = 0.00002
optimizer = keras.optimizers.Adam(lr= lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss= 'sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#hist = model.fit(x=arrayImg, y=reduced_labels, epochs=10, verbose=1, batch_size = 8, validation_split=0.2)


#Con Augmentation
X_train, X_val, Y_train, Y_val = sklearn.cross_validation.train_test_split(arrayImg, reduced_labels, test_size=0.2, random_state=5)

datagen = ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = datagen.flow(X_train, Y_train, batch_size=8)
hist = model.fit_generator(train_generator,
                    nb_epoch=5,
                    steps_per_epoch=X_train.shape[0] // 8,
                    validation_data=(X_val,Y_val))


print(hist.history)
model.save('C101_test5_augm.h5')


'''
Evaluate the model

imgTest = cv2.imread("input_data\\accordionimage_0021.jpg.jpg")
imgTest = cv2.resize(imgTest, (224, 224))
cv2.imshow('img',imgTest)
cv2.waitKey(0)
imgTest = image.img_to_array(imgTest)
imgTest = np.expand_dims(imgTest, axis=0)
imgTest = preprocess_input(imgTest)


prediction = model.predict(imgTest)
print(prediction[0])
pos = np.where(prediction[0]==max(prediction[0]))
print(pos[0])
print(np.str(categories[expandLabels(np.unique(labels),pos[0][0])]))
'''

'''
prediction = model.predict(imgTest)
print(prediction)
pos = np.where(prediction==1)
cat_pred = np.where(np.unique(labels)==pos[0])
print(categories_array[cat_pred[0]])
#model.evaluate(X_test, Y_test, verbose=1)
'''
