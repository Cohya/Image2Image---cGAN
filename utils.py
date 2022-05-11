import numpy as np 
import pandas as pd 
import glob
from PIL import Image 
import matplotlib.pyplot as plt
import tensorflow as tf 
import os 
import pickle
# list_of_files_norm_train = glob.glob('data/edges2shoes/edges2shoes/train/*')
# # # list_of_files_norm_test = glob.glob('data/edges2shoes/edges2shoes/Val/*')
# # # file  = list_of_files_norm_test[0]

# # # list_of_files_norm_train = glob.glob('data/edges2shoes/edges2shoes/train/*')


# files_locations = list_of_files_norm_train[0:10]

IMG_WIDTH = 256
IMG_HEIGHT = 256
def read_png_get_numpy_array(files_locations):
    # 80% for train
    n = len(files_locations) 
    X = np.zeros(shape = (int(n), 256,256,3)) # The edgws 
    Y = np.zeros(shape = (int(n), 256,256,3)) # The colored image 
    count = 0
    for file in files_locations:
        # print(file)
        file_Image = Image.open(file)
        image_data = np.asarray(file_Image)
        edge = image_data[:,0:256,:]
        colored = image_data[:,256:,:]
        X[count,:,:,:] = edge
        Y[count,:,:,:] = colored
        count += 1
    
    X = np.dot(X, [0.2989, 0.5870, 0.1140])
    X = X[...,np.newaxis]
    return X, Y

def get_shoose_data(files_locations):
    X, Y = read_png_get_numpy_array(files_locations)
    return X.astype(np.float32), Y.astype(np.float32)

def get_shoose_data_normelized_1_to_1(files_locations):
    X, Y = read_png_get_numpy_array(files_locations)
    X = X/255.0
    Y = Y / 255.0
    
    # move to -1 t0 1
    X = 2*X - 1
    Y = 2* Y -1
    return X.astype(np.float32), Y.astype(np.float32)

def get_shoose_data_normelized_1_to_1_with_mirror(files_locations):
    X, Y = read_png_get_numpy_array(files_locations)
    X = X/255.0
    Y = Y / 255.0
    
    # move to -1 t0 1
    X = 2*X - 1
    Y = 2* Y -1
    
    numb = X.shape[0]
    
    for i in range(numb):
        
        if np.random.random(1) > 0.5:
            #random mirroring
            X[i,:,:,:] = np.fliplr(X[i,:,:,:])
            Y[i,:,:,:] = np.fliplr(Y[i,:,:,:])
    return X.astype(np.float32), Y.astype(np.float32)

def read_png_get_numpy_arrayX3D(files_locations):
    # 80% for train
    n = len(files_locations) 
    X = np.zeros(shape = (int(n), 256,256,3)) # The edgws 
    Y = np.zeros(shape = (int(n), 256,256,3)) # The colored image 
    count = 0
    for file in files_locations:
        # print(file)
        file_Image = Image.open(file)
        image_data = np.asarray(file_Image)
        edge = image_data[:,0:256,:]
        colored = image_data[:,256:,:]
        X[count,:,:,:] = edge
        Y[count,:,:,:] = colored
        count += 1

    return X, Y

def get_shoose_data_normelized_1_to_1_X3d_with_mirror(files_locations):
    X, Y = read_png_get_numpy_arrayX3D(files_locations)
    X = X/255.0
    Y = Y / 255.0
    
    # move to -1 t0 1
    X = 2*X - 1
    Y = 2* Y -1
    
    numb = X.shape[0]
    
    for i in range(numb):
        
        if np.random.random(1) > 0.5:
            #random mirroring
            X[i,:,:,:] = np.fliplr(X[i,:,:,:])
            Y[i,:,:,:] = np.fliplr(Y[i,:,:,:])
    return X.astype(np.float32), Y.astype(np.float32)
    
def back_to_0_1(X):
    Y = (X+1) / 2
    return Y 

def get_choose_Validation():
    list_of_files_norm_test = glob.glob('data/edges2shoes/edges2shoes/Val/*')

    n = len(list_of_files_norm_test) 
    X = np.zeros(shape = (int(n), 256,256,3)) # The edgws 
    Y = np.zeros(shape = (int(n), 256,256,3)) # The colored image 
    count = 0
    for file in list_of_files_norm_test:
        file_Image = Image.open(file)
        image_data = np.asarray(file_Image)
        edge = image_data[:,0:256,:]
        colored = image_data[:,256:,:]
        X[count,:,:,:] = edge
        Y[count,:,:,:] = colored
        
        
        
    return X, Y
    

# read the image and split 
def load(image_file, side_of_real = 'RIGHT'):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    
    w = tf.shape(image)[1]
    
    w = w //2 
    if side_of_real == 'RIGHT':
        # print('hallo')
        real_image = image[:,w:,:]
        input_image = image[:,:w,:]
    else:
        # print('hallo2')   
        input_image = image[:,w:,:]
        real_image = image[:,:w,:]
    
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    
    return input_image, real_image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    real_image = tf.image.resize(real_image, [height, width], 
                                 method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return input_image, real_image


                        
def random_crop(input_image, real_image,img_height, img_width):
    stacked_image = tf.stack([input_image, real_image], axis = 0)
    cropped_image = tf.image.random_crop(stacked_image, size = [2, img_height, img_width,3])
    return cropped_image[0], cropped_image[1]


# normelizing the image to [-1,1]
def normalize(input_image, real_image):
    input_image = 2*(input_image/255.0) - 1 
    real_image = 2*( real_image / 255.0) - 1
    
    return input_image, real_image

def random_jitter(input_image, real_image):
    #resizing the image to 286x286x3
    input_image, real_image = resize(input_image, real_image,286,286)
    
    #randomly cropping to 256x256
    input_image, real_image = random_crop(input_image, real_image,256,256)
    
    if tf.random.uniform(()) > 0.5:
        #random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
        
    return input_image, real_image

 

def load_image_train(image_file,side_of_real = 'RIGHT'):
    input_image, real_image = load(image_file, side_of_real = side_of_real)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    
    return input_image, real_image


def save_model(model_name, model):
    exist_folder = os.path.isdir('Model')
    if exist_folder == False:
        path = os.getcwd() + '\\model'
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s faild" % path)
        else:
            print("Successfuly creat the directory %s" %path)
    model_name_path = 'model/'+ model_name        
    with open(model_name_path, 'wb') as model_name:
        pickle.dump(model, model_name)
        print('model saved succesfully')
        
def load_model(path):
    with open(path, 'rb') as model_path:
        model = pickle.load(model_path)
        
    return model
        
        
        
# X, Y = get_shoose_data_normelized_1_to_1_X3d_with_mirror(files_locations)

# image_file = files_locations[1]
# input_image, real_image =  load_image_train(image_file )

# # inp, re = random_crop(inp, re) 
# # # casting to int for matplotlib to show the image
# plt.figure()
# plt.imshow(((input_image+1)/2))
# plt.figure()
# plt.imshow(((real_image+1)/2))
    

# plt.figure(figsize=(6, 6))
# for i in range(4):
#   rj_inp, rj_re = random_jitter(inp, re)
#   plt.subplot(2, 2, i+1)
#   plt.imshow(rj_inp/255.0)
#   plt.axis('off')
# plt.show()