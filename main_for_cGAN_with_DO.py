import sys 

sys.path.append("../")
import scipy as sp 
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
from datetime import datetime
import glob
import pickle
from sklearn.utils import shuffle
from cGAN_activation_befor_conc_and_DO import cGAN, Unet, PatchGanDiscriminator ,lrelu

from utils import save_model, load_model,get_shoose_data_normelized_1_to_1_with_mirror
# Forr choose the size is 256,256
colors_of_BCs = 1 # adding one more for noise
colors_of_generator = 3
#[feature maps out, filter_sizes, strides size, batch norm] the last layer of the encoder
# should be, maybe, without normalization  
#{mo, filtersz, stride, apply_batch_norm, f, padding, apply_zero_padding, Drop_out, add_bias} 
g_sizes = {
  'encoder': [(64, 4, 2, False, lrelu,'SAME', False, False, False), 
              (128, 4, 2, True,lrelu,'SAME',False, False, False), 
              (256,4,2, True,lrelu,'SAME',False, False, False), 
              (512, 4, 2, True,lrelu,'SAME',False, False, False),
              (512,4,2, True,lrelu,'SAME',False ,False,False),
              (512,4,2, True,lrelu,'SAME',False, False, False),
              (512,4,2, True ,lrelu,'SAME',False, False, False),
              (512,4,2, True ,lrelu,'SAME',False, False, False)],
  'decoder_activation': tf.nn.relu,
  'colors_last_layer_decoder': [(colors_of_generator ,4,2, False, False, True)],
  'output_activation': tf.nn.tanh,
  'Dropout_in_decoder': 3
}# tf.identity        # The dropout in decoder is for the first 3 layers in the decoder

# unet = Unet(g_sizes = g_sizes,img_size = (256,256), color = colors_of_BCs  , batch_size = 32)
# X = np.random.random(size = [1,256,256,1]) #[x, y]
# Y = unet.forward(X, is_training=True)
# # X1 = unet.g_convlayers_encoder[0].forward(X, is_training = False)
#g_sizes, (256,128), 2)


# plt.figure(1)
# plt.imshow(X[0,:,:,0])
# F = unet.forward(X, is_training = False)
# plt.figure(2)
# plt.imshow(F[0,:,:,0])
# # unet.g_convlayers_encoder[0].W.shape
# o = unet.g_convlayers_encoder[0].forward(X, is_training = False)
# o2 = unet.g_convlayers_encoder[1].forward(o, is_training = False)
#{mo, filtersz, stride, apply_batch_norm, f, padding, apply_zero_padding, Drop_out, add_bias} 
d_sizes = {
  'conv_layers': [(64, 4, 2, False, lrelu,'SAME',False, False, False)
                  ,(128, 4, 2, True,lrelu,'SAME',False, False, False),
                  (256,4,2, True,lrelu,'SAME',True, False, False),
                  (512, 4 ,1, True,lrelu,'VALID',True,False, False)],
  'last_layers': [(1,4,1, False,tf.identity,'VALID',False, False, True)],
}



# patchnet_dis = PatchGanDiscriminator(d_sizes, img_size = (256,256),
                                      # num_colors = colors_of_generator  + colors_of_BCs)

# X =  patchnet_dis.forward(np.random.random(size = [1,256,256,4]))

# X =  patchnet_dis.forward(np.random.random(size = [1,256,256,4]))
# Z = np.random.random(size = [3,265,128,2])

cgan = cGAN(imgsize = (256,256), num_colors_BC = 1, num_colors_results = 3, d_sizes = d_sizes,
            g_sizes = g_sizes, batch_size = 10)


X = glob.glob('../data/edges2shoes/edges2shoes/train/*')
X = shuffle(X)
Xtrain = X[0:4000]
Xtest = X[2000:2500]
cgan.fit(Xtrain, Targets = [], Xtest = Xtest, Ytest = [], epochs=20, batch_size=4, 
         save_wights_during_training = True,take_generated_images_during_traning = True, 
          typee="image", data_type = 'From_folder', no_of_save_image_per_epoch =5, load=True
          ,lambdaa = 100 , step_for_save_epochs=1)

# load only the weights(include beta and gamma is batch norm was applied in the original net)
# of the discriminator and generator with the norm params (running mean and var)
cgan.load(load_norm_params =True)

#Save the model as full model (all the weights and the running mean/var and gamma +beta) 
save_model('cgan',cgan)
#load the model with all params inside 
cgan = load_model('model/cgan')
#make a prediction

Xvalid = X[6005:6006]
Xv, Tv = get_shoose_data_normelized_1_to_1_with_mirror(Xvalid)
Predicted = cgan.get_sample_from_generator(Xv,is_training = False)

cgan.images_of_samples_from_generator_contour(Xv, Tv, typee = "image")









