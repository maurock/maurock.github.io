from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import cv2
import os
from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers
from keras.utils import np_utils
import pandas as pd
from sklearn.utils import shuffle
from keras import optimizers
import random
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import time

start_time = time.clock()

#Counter: Drawing numbers found in the concessions, but not present in the image database
IMAGES_NOT_FOUND=0

'''
Define VGG16 Neural Network.
@input: shape_input (int), number of categories in input
@input: shape_output (int), number of categories in output
@input -optional: weights_path (String), path of the weight in case of pre-trained Network.
                  The weight for the pre-training needs to be generate with the same Network.
@output: model, the model of the network
Notes: the size of the image to process, in the variable 'main_input', is 224x224. In order to customize it,
images need to be resized accordingly.
'''
def VGG_16( shape_input, shape_output, weights_path=None):

    #main input: process the images
    main_input = Input(shape=(224,224,3), name='main_input')
    x = ZeroPadding2D((1,1),data_format='channels_last')(main_input)
    x = Convolution2D(64, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = ZeroPadding2D((1,1))(x)
    x = (Convolution2D(256, 3, 3, activation='relu'))(x)
    x = (ZeroPadding2D((1,1)))(x)
    x = (Convolution2D(256, 3, 3, activation='relu'))(x)
    x = (ZeroPadding2D((1,1)))(x)
    x = (Convolution2D(256, 3, 3, activation='relu'))(x)
    x = (MaxPooling2D((2,2), strides=(2,2)))(x)
    x = (ZeroPadding2D((1,1)))(x)
    x = (Convolution2D(512, 3, 3, activation='relu'))(x)
    x = (ZeroPadding2D((1,1)))(x)
    x = (Convolution2D(512, 3, 3, activation='relu'))(x)
    x = (ZeroPadding2D((1,1)))(x)
    x = (Convolution2D(512, 3, 3, activation='relu'))(x)
    x = (MaxPooling2D((2,2), strides=(2,2)))(x)
    x = (ZeroPadding2D((1,1)))(x)
    x = (Convolution2D(512, 3, 3, activation='relu'))(x)
    x = (ZeroPadding2D((1,1)))(x)
    x = (Convolution2D(512, 3, 3, activation='relu'))(x)
    x = (ZeroPadding2D((1,1)))(x)
    x = (Convolution2D(512, 3, 3, activation='relu'))(x)
    x = (MaxPooling2D((2,2), strides=(2,2)))(x)
    x = (Flatten())(x)

    #auxiliary_input: process the textual data
    auxiliary_input = Input(shape=(shape_input,), name='aux_input')
    y = (Dense(100, activation='relu'))(auxiliary_input)
    auxiliary_output = (Dense(shape_output, activation='softmax'))(y)
    j = keras.layers.concatenate([auxiliary_output, x])

    x = (Dense(4096, activation='relu'))(j)
    x = (Dropout(0.25))(x)
    x = (Dense(4096, activation='relu'))(x)
    x = (Dropout(0.25))(x)
    output = (Dense(shape_output, activation='softmax'))(x)

    model = Model(inputs=[main_input, auxiliary_input], outputs=output)

    if weights_path:
        model.load_weights(weights_path)

    return model

'''
Create dictionaries
@input: array of strings
@output: dictionary
Example:
input =[ca, ku, pi]        output = ['ca' : 0, 'ku : 1', ..]
'''
def create_dict(a):
    b = np.arange(0, np.unique(a).size, 1)
    c = np.unique(sorted(a))
    new_dict = dict(zip(c, b))
    return new_dict

'''
Convert elements in the dictionary to unique integers.
@input: array of strings
@output: value of the dictionary for each string
Example:
input = [ca, ku, pi]       output = [0,1,2]
'''
def reduceLabels(a):
    new_dict = create_dict(a)
    return np.array(list(map(new_dict.get, a)))

'''
Return the key of the dictionary, given an array and a value
@input: array, value of the desired key
@output: key of the value
'''
def expandLabels(array, num):
    new_dict = create_dict(array)
    for key, value in new_dict.items():
        if value == num:
            return key

'''
Print value of learning rate, to check whether the learning rate decay happens or not.
The decay is done in the ReduceLROnPlateau()
'''
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

'''
Load images in a numpy array. and create label for text
'''
path_input = 'ConcessionImagesFiltered\\'       #path containing images
arrayImg = []                                   #array where images are appended
label = []                                      #array where labels for images are appended
splitLabel = []                                 #array where the Drawing Number is split in ATA-SUBATA, and the arbitrary number
path_concession_csv = 'C:\\Users\\ZINKCLOUD07\\PycharmProjects\\TextImageNetwork\\concessions.csv'      #path where the csv of the concessions is located
table=pd.read_csv(path_concession_csv, names=['id','drawing_number','frequency'])                       #id: name of the concession, frequency: number of times the drawing number
                                                                                                        #appears in the all the concessions
id = np.array(table['id'])
new_dict = dict(zip(np.array(table['id']), np.array(table['drawing_number'])))                          #create dictionary correlating id to drawing number
print(new_dict)

cont_img=0                  #counter for images processed
output_label=[]             #array containing the arbitrary part of the drawing number

''''
Split the name of the image in order to create the array of labels (ATA) and output_label (arbitrary code) with a dictionary
containing the association File : drawing number.
Also, load an array with all the images.
label = firts 4 characters
output_label = everything else
example:
File= CG-004722042-A-182.png
string = CG-004722042
temp_label = V53580010
label = V535
output_label = 80010
'''
for subdirect in os.listdir(path_input):
    cont_img=cont_img+1
    img = cv2.imread(path_input+subdirect)
    img = cv2.resize(img, (224, 224)).astype(np.float64)
    img = image.img_to_array(img)
    img = preprocess_input(img)
    splitLabel = subdirect.split('-')
    if(len(splitLabel)==3):
        string ='%s-%s' % (splitLabel[0], splitLabel[1])
        try:
            temp_label = new_dict[string]
            label.append(temp_label[:4])
            output_label.append(temp_label[4:9])
            arrayImg.append(img)
        except KeyError:
            IMAGES_NOT_FOUND = IMAGES_NOT_FOUND + 1
            print('images not found: ' + np.str(IMAGES_NOT_FOUND) + '/' + np.str(cont_img))
    else:
        string = '%s-%s-%s' % (splitLabel[0], splitLabel[1], splitLabel[2])
        try:
            temp_label=new_dict[string]
            label.append(temp_label[:4])
            output_label.append(temp_label[4:9])
            arrayImg.append(img)
        except KeyError:
            IMAGES_NOT_FOUND = IMAGES_NOT_FOUND + 1
            print('images not found: ' + np.str(IMAGES_NOT_FOUND) + '/' + np.str(cont_img))

'''
output_label = XXXX-5123    (the second part)
reduced_output_label = 1 1 2 3 2
label = F551-XXXX          (the first part)
reduced_labels = [1 1 2 3 1 4]
array_categorical = [1 0 0 0], [0 1 0 0]
'''
reduced_output_labels = reduceLabels(output_label)
arrayImg = np.array(arrayImg)
label = np.array(label)
reduced_labels = reduceLabels(label)
array_categorical = np_utils.to_categorical(reduced_labels)
arrayImg, array_categorical, reduced_output_labels = shuffle(arrayImg, array_categorical, reduced_output_labels, random_state= 313)


'''
Fit model to training set
'''
print('Unique label in input: ' + np.str(np.unique(label)))
print('Categories in output: ' + np.str(len(np.unique(output_label))))
shape_input = len(np.unique(label))
shape_output = len(np.unique(output_label))
model = VGG_16(shape_input= shape_input , shape_output= shape_output, weights_path= 'air_dataset_test8.h5')


#sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
lr_init = 0.00002

#set learning rate decay            <NOT USED>
def scheduler(epoch):
    if epoch > 5:
        if epoch > 10:
            K.set_value(model.optimizer.lr, (lr_init / 2)/2)
        else:
            K.set_value(model.optimizer.lr, lr_init/2)
    return K.get_value(model.optimizer.lr)

#Allows scheudle the learning rate to decay in a fixed way along the epochs
#learning_rate_reduction = LearningRateScheduler(scheduler)

optimizer = keras.optimizers.Adam(lr= lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#lr_metric in case we schedule the elarning rate
lr_metric = get_lr_metric(optimizer)
model.compile(loss= 'sparse_categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'] )

model.fit(x=[arrayImg, array_categorical], y= reduced_output_labels, epochs=20, verbose=2,
          batch_size = 8, validation_split=0.2, shuffle=True)
model.save('air_dataset_test10.h5')

print(time.clock() - start_time, "seconds")