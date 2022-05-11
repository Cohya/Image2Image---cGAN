import sys 

sys.path.append("../")
import os 
import scipy as sp 
import random
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
# import util
from datetime import datetime
import pickle
import glob
from pathlib import Path
from sklearn.utils import shuffle
from utils import  get_shoose_data_normelized_1_to_1_with_mirror ,back_to_0_1,get_shoose_data_normelized_1_to_1


# Global 
decay = 0.99
if not os.path.exists('samples'):
    os.mkdir('samples')
    
if not os.path.exists('samples/train'):
    os.mkdir('samples/train')

if not os.path.exists('samples/test'):
    os.mkdir('samples/test')
# batch_size = 64
# Make dir to save samples 
def identity(x):
    return x
    
    
def lrelu(x, alpha = 0.2):
    return tf.maximum(alpha*x, x)


class ConvLayer(object):
    def __init__(self, name, mi, mo, apply_batch_norm, dim_output_x, dim_output_y, filtersz = 4, stride = 2, f=tf.nn.relu, Pad = 'SAME',
                 apply_zero_padding = 'False', Drop_out = False, add_bias = False):
        # mi = input feature map size 
        # mo = output feature map size
        # Gets an existing variable with this name or creat a new one 
        
        # if apply_zero_padding: # for applying batch norm befiore zero padding
        #     dim_output_x = dim_output_x -2
        #     dim_output_y = dim_output_y -2
        self.add_bias = add_bias
        self.W = tf.Variable(initial_value = tf.random.normal(shape = [filtersz,filtersz, mi, mo],
                                                              stddev = 0.02), name = "W_%s" % name) 
                                 
        if self.add_bias:
            self.b = tf.Variable(initial_value = tf.zeros(mo,), name = "b_%s" % name)
         
        self.apply_batch_norm = apply_batch_norm                       
        
        #for batch norm 
        
        self.gamma = tf.Variable(initial_value = tf.ones(shape = [mo,]) , name ="gamma_%s" % name)
                                     
        self.beta = tf.Variable(initial_value  = tf.zeros(shape = [mo,]), name = "beta_%s" % name)
        
        self.running_mean = tf.Variable(initial_value = tf.zeros(shape = [dim_output_x,dim_output_y, mo]) ,
                                            name = "running_mean_%s" % name,
                                            trainable = False) # Trainable = False [dim_output_x,dim_output_y, mo]
        
        self.running_var = tf.Variable(initial_value = tf.ones(shape = [dim_output_x,dim_output_y, mo]) ,
                                       name = "running_var_%s" % name,
                                       trainable = False) # trainable = False [dim_output_x,dim_output_y, mo]
        
        self.normalization_params = [self.running_mean, self.running_var]
        self.name = name 
        self.f = f
        self.stride = stride
        self.pad = Pad
        self.Drop_out = Drop_out
        self.apply_zero_padding = apply_zero_padding
       
        if self.add_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
        
        if self.apply_batch_norm:
            self.params += [self.gamma, self.beta]
        
    def forward(self, X, is_training):
        conv_out = tf.nn.conv2d(X, filters = self.W, strides=[1, self.stride, self.stride,1],
                                padding=self.pad )
        
        if self.add_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
        # print(conv_out.shape)
        
        # apply batch norm 
        
        if self.apply_batch_norm:
            if is_training:
                batch_mean, batch_var = tf.nn.moments(conv_out, [0,1,2])
                
                self.running_mean.assign(self.running_mean * decay + batch_mean * (1 - decay))
               
                self.running_var.assign(self.running_var * decay + batch_var * (1 - decay))
                    
                self.normalization_params = [self.running_mean, self.running_var]   
                # with tf.control_dependencies([batch_mean, batch_var]):
                conv_out = tf.nn.batch_normalization(conv_out ,
                                                 batch_mean, 
                                                 batch_var ,
                                                self.beta,
                                                self.gamma,
                                                1e-3)
            else:
                
                  # this is for the testing 
                conv_out = tf.nn.batch_normalization(conv_out,
                                            self.running_mean,
                                            self.running_var,
                                            self.beta,
                                            self.gamma,
                                            1e-3)
                
        if self.Drop_out:
            if is_training:
                conv_out = tf.nn.dropout(conv_out, 0.5)
            # print("Drop")
            
            
        conv_out = self.f(conv_out)
        
        if self.apply_zero_padding:
            conv_out = tf.keras.layers.ZeroPadding2D(padding=1)(conv_out)
                
        # if self.apply_batch_norm: 
        #     if is_training:
        #         batch_mean, batch_var = tf.nn.moments(conv_out, [0])
        #         self.running_mean =  (self.running_mean*decay + batch_mean * (1-decay))
                                                   
        #         self.running_var = (self.running_var * decay + batch_var * (1-decay))
                    
        #         conv_out = (conv_out - self.running_mean)/(tf.math.sqrt(self.running_var + 1e-5))
                    
        #         conv_out = self.gamma * conv_out + self.beta
        #     else:
        #         conv_out = (conv_out - self.running_mean)/(tf.math.sqrt(self.running_var + 1e-5))
                    
        #         conv_out = self.gamma * conv_out + self.beta
        
        
                
        return conv_out 
        

class FractionallyStrideConvLayer(object):
    def __init__(self, name, mi, mo, output_shpae, apply_batch_norm, filtersz = 4, stride = 2,f = tf.nn.relu , Drop_out = False,
                 add_bias = False):
        # mi = input feature maps 
        # mo = output feature maps 
        # Note: shape is specified in the opposite wat from regular
        self.add_bias = add_bias
        
        self.W = tf.Variable(initial_value = tf.random.normal(shape = [filtersz, filtersz, mo, mi], stddev=0.02),
                             name = "W_%s" % name) # look the mo mi is opposite way 
        
        if self.add_bias:
            self.b = tf.Variable(initial_value = tf.zeros(shape = [mo,]), name = "b_%s" % name)
        
        self.apply_batch_norm = apply_batch_norm
        
        #for batch norm

        self.gamma = tf.Variable(initial_value = tf.ones(shape = [mo,]) , name ="gamma_%s" % name)
                                     
        self.beta = tf.Variable(initial_value  = tf.zeros(shape = [mo,]), name = "beta_%s" % name)
        
        self.running_mean = tf.Variable(initial_value = tf.zeros(shape = output_shpae[1:]) ,
                                            name = "running_mean_%s" % name,
                                            trainable = False) # Trainable = False
        
        self.running_var = tf.Variable(initial_value = tf.ones(shape = output_shpae[1:]) ,
                                       name = "running_var_%s" % name, 
                                       trainable = False) # trainable = False        
        
        self.normalization_params = [self.running_mean, self.running_var]
        
        self.name = name 
        self.f = f
        self.stride = stride
        
        self.Drop_out = Drop_out
        self.output_shape = output_shpae
        
        if self.add_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
        
        if self.apply_batch_norm:
            self.params += [self.gamma, self.beta]
        
    def forward(self, X, is_training = True):
        gh = [len(X)]
        gh += self.output_shape[1:]
        conv_out = tf.nn.conv2d_transpose(input = X, filters = self.W, 
                                          output_shape=gh ,
                                          strides=[1, self.stride, self.stride, 1])#output_shape=self.output_shape
        if self.add_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
        # print(conv_out.shape)
        # apply batch_normalization
        
        if self.apply_batch_norm:
            if is_training:
                batch_mean, batch_var = tf.nn.moments(conv_out, [0,1,2])
                # print("Batch_maen_conv:", batch_mean.shape)
                self.running_mean.assign(self.running_mean * decay + batch_mean * (1 - decay))
               
                self.running_var.assign(self.running_var * decay + batch_var * (1 - decay))
                self.normalization_params = [self.running_mean, self.running_var]   
                # with tf.control_dependencies([batch_mean, batch_var]):
                conv_out = tf.nn.batch_normalization(conv_out ,
                                                batch_mean,
                                                batch_var,
                                                self.beta,
                                                self.gamma,
                                                1e-3)
            else:
                
                  # this is for the testing 
                conv_out = tf.nn.batch_normalization(conv_out,
                                            self.running_mean,
                                            self.running_var,
                                            self.beta,
                                            self.gamma,
                                            1e-3)
        if self.Drop_out:
            if is_training:
                conv_out = tf.nn.dropout(conv_out, 0.5)
            # print("Dropout")
        # if self.apply_batch_norm:
        #     if is_training:
        #         batch_mean, batch_var = tf.nn.moments(conv_out, [0])
        #         self.running_mean =  (self.running_mean*decay + batch_mean * (1-decay))
                                               
        #         self.running_var = (self.running_var * decay + batch_var * (1-decay))
                
        #         conv_out = (conv_out - self.running_mean)/(tf.math.sqrt(self.running_var + 1e-5))
                
        #         conv_out = self.gamma * conv_out + self.beta
        #     else:
                
        #         conv_out = (conv_out - self.running_mean)/(tf.math.sqrt(self.running_var + 1e-5))
                
        #         conv_out = self.gamma * conv_out + self.beta 
        # without activation because we concatenate 
        
        
        
        return self.f(conv_out)
    
    
    
class DenseLayer(object):
    def __init__(self, name, M1, M2, apply_batch_norm, f = tf.nn.relu):
        
        self.W = tf.Variable(initial_value = tf.random.normal(shape = [M1, M2] , stddev=0.02),
                             name = "W_%s" % name)
        
        self.b = tf.Variable(initial_value = tf.zeros(shape = [M2,]), name = "d_%s" %name)
        
        self.apply_batch_norm = apply_batch_norm
        
        #for batch norm 
        # if self.apply_batch_norm:
        self.gamma = tf.Variable(initial_value = tf.ones(shape = [M2,]) , name ="gamma_%s" % name)
                                     
        self.beta = tf.Variable(initial_value  = tf.zeros(shape = [M2,]), name = "beta_%s" % name)
        
        self.running_mean = tf.Variable(initial_value = tf.zeros(shape = [M2,]) ,
                                            name = "running_mean_%s" % name,
                                            trainable = False) # Trainable = False
        
        self.running_var = tf.Variable(initial_value = tf.ones(shape = [M2,]) ,
                                       name = "running_var_%s" % name, 
                                       trainable = False) # trainable = False  
        
        self.normalization_params = [self.running_mean, self.running_var]
        
        self.f = f
        self.name = name

        
        self.params = [self.W, self.b]
        if self.apply_batch_norm:
            self.params += [self.gamma, self.beta]
        
    def forward(self, X, is_training):
        a = tf.matmul(X, self.W) + self.b
        
        
        if self.apply_batch_norm:
            if is_training:
                batch_mean, batch_var = tf.nn.moments(a, [0])
                
                self.running_mean.assign(self.running_mean * decay + batch_mean * (1 - decay))
               
                self.running_var.assign(self.running_var * decay + batch_var * (1 - decay))
                self.normalization_params = [self.running_mean, self.running_var]
                # with tf.control_dependencies([batch_mean, batch_var]):
                a = tf.nn.batch_normalization(a ,
                                                batch_mean,
                                                batch_var,
                                                self.beta,
                                                self.gamma,
                                                1e-3)
            else:
                
                  # this is for the testing 
                a = tf.nn.batch_normalization(a,
                                            self.running_mean,
                                            self.running_var,
                                            self.beta,
                                            self.gamma,
                                            1e-3)  
        # if self.apply_batch_norm:
        #     if is_training:
        #         batch_mean, batch_var = tf.nn.moments(a, [0])
        #         self.running_mean =  (self.running_mean*decay + batch_mean * (1-decay))
                                               
        #         self.running_var = (self.running_var * decay + batch_var * (1-decay))
                
        #         a = (a - self.running_mean)/(tf.math.sqrt(self.running_var + 1e-5))
                
        #         a= self.gamma * a + self.beta
        #     else:
        #         a = (a - self.running_mean)/(tf.math.sqrt(self.running_var + 1e-5))
                
        #         a= self.gamma * a + self.beta
                
        return self.f(a)

# Build generator 

class Unet(object):
    def __init__(self, g_sizes, img_size, color, batch_size = 32):
        # Determind the size of the data at eact step 
        dims = [img_size]
        dimx = img_size[0]
        dimy = img_size[1]
        self.g_sizes = img_size
        self.drop_out = g_sizes['Dropout_in_decoder']
        # Encoder sizes 
        for _, _, stride, _ ,_, _,_,_,_ in g_sizes['encoder']:
            dimx = int(np.ceil(float(dimx) / stride))
            dimy = int(np.ceil(float(dimy) / stride))
            dims.append((dimx, dimy))  
            
        print("Encoder Dims:", dims)
        self.dims_encoder = dims
        
        # dimx = self.dims_encoder[-1][0]
        # dimy = self.dims_encoder[-1][1]
        self.dims_decoder = []
        #Decoder

        for _,_, stride, _,_ ,_ , _, _,_ in reversed(g_sizes['encoder'][:-1]):
            # Determind the size of the data at each step
            dimx = int(np.ceil(float(dimx) * stride))
            dimy = int(np.ceil(float(dimy) * stride))
            
            self.dims_decoder.append((dimx, dimy))
        # Fro the last layer 
        self.dims_decoder.append((int(np.ceil(float(dimx) * stride)), int(np.ceil(float(dimy) * stride))))
        print("Decoder dims:", self.dims_decoder)
        
        
        # Build the convlayer - Use leaky relu 
        
        # activations for the encoder
        activation_functions = [lrelu] * (len(g_sizes['encoder'])-1) + [tf.nn.relu]
        self.g_convlayers_encoder = []
        
        mi = color
        print("Encoder:")
        for i in range(len(g_sizes['encoder'])):
            name = "Conv_encoder_%s" % i 
            mo, filtersz, stride, apply_batch_norm, f, padding, apply_zero_padding, Drop_out, add_bais = g_sizes['encoder'][i] 
            # print("mo:", mo)
            # f = activation_functions[i]
            output_shape = [batch_size,  int(self.dims_encoder[i+1][0]), int(self.dims_encoder[i+1][1]), mo]
            # print("Output_shape:", output_shape)
            layer = ConvLayer(name, mi, mo, apply_batch_norm, dim_output_x=output_shape[1], 
                              dim_output_y= output_shape[2], filtersz=filtersz, stride=stride,f = f, Pad = padding,
                              apply_zero_padding= apply_zero_padding, Drop_out = Drop_out, add_bias= add_bais)
                              # stride=stride, f = f)
            
            # print("mi:",mi)
            print("mi:", mi, "mo:", mo, "output shape", output_shape, "Batch_norm:", apply_batch_norm, "activation:", str(f)[10:15],
                  "DO:", Drop_out,"zero_padding:", apply_zero_padding, "bias:", add_bais)
            mi = mo
            self.g_convlayers_encoder.append(layer)
         
            
         
        # Decoder - Use Relu
        # activation_functions_decoder =  (len(g_sizes['encoder'])) * [tf.nn.relu]
        
        self.g_convlayers_decoder = []
                
        

        decoder = []
        num = len(g_sizes['encoder'])-1
        for i in range(len(g_sizes['encoder'])):
            decoder.append(g_sizes['encoder'][num - i]  )
        
        decoder.pop(0)
        ## Change the last layer to be with batch norm
        y = list(decoder[-1])
        y[3] = True
        decoder[-1] = y
        # decoder[-1][3] = True
        # print(decoder)
        
        print("Decoder:")
        number_of_DO = g_sizes['Dropout_in_decoder']
        for i in range(len(decoder) ):
            name = "fsConv_decoder_%s" % i 
            if i <  number_of_DO:
                Drop_out = True
            else:
                Drop_out = False
                
            if i == 0: # bottleNek
                mo, filtersz, stride, _,_,_,_,_ ,add_bias= decoder[i]
                apply_batch_norm =True
                f = g_sizes['decoder_activation']
                output_shape = [batch_size, int(self.dims_decoder[i][0]),int(self.dims_decoder[i][1]),mo]
                layer = FractionallyStrideConvLayer(name, mi, mo, output_shpae= output_shape,
                                                    apply_batch_norm= apply_batch_norm, filtersz=filtersz,
                                                    stride=stride, f = f, Drop_out= Drop_out, add_bias = add_bais)
                print("mi:", mi, "mo:", mo, "output shape", output_shape, "Batch_norm:", apply_batch_norm, "activation:", str(f)[10:15],
                      "DO:",Drop_out, "bias:" , add_bais)
                mi = 2 * mo
                # print("mi:", mi)
            else:
                if i <= self.drop_out - 1:
                    DO = True
                else:
                    DO = False
                mo, filtersz, stride, apply_batch_norm,_,_,_,_, add_bias = decoder[i]
                f = g_sizes['decoder_activation']
                output_shape = [batch_size, int(self.dims_decoder[i][0]),int(self.dims_decoder[i][1]),mo]
                layer = FractionallyStrideConvLayer(name, mi, mo, output_shpae= output_shape,
                                                    apply_batch_norm= apply_batch_norm, filtersz=filtersz,
                                                    stride=stride, f = f, Drop_out= Drop_out, add_bias=add_bias)
                print("mi:", mi, "mo:", mo, "output shape", output_shape, "Batch_norm:", apply_batch_norm,"activation:", str(f)[10:15],
                      "DO:", DO, "bais:", add_bias)
                mi =  2* mo
                # print("mi:", mi)
                
            self.g_convlayers_decoder.append(layer)
                
            
        i += 1
        name = "fsConv_decoder_last" 
        # Build the last layer 
        f = g_sizes['output_activation']
        mo , filtersz, stride, apply_batch_norm, Drop_out, add_bias = g_sizes['colors_last_layer_decoder'][0]
        output_shape = [batch_size, int(self.dims_decoder[i-1][0])*2,int(self.dims_decoder[i-1][1])*2,mo]
        layer = FractionallyStrideConvLayer(name, mi, mo, output_shpae= output_shape,
                                                apply_batch_norm= apply_batch_norm, filtersz=filtersz,
                                                stride=stride, f = f, Drop_out= Drop_out, add_bias=add_bias)
        print("mi:", mi, "mo:", mo, "output shape", output_shape, "Batch_norm:", apply_batch_norm, "activation:" , 
              str(f)[10:15], "DO:", Drop_out, "bias:", add_bias)
        
        self.g_convlayers_decoder.append(layer)
        
        
        ## Collect all tranable params from the layers
        self.params = []
        for layer in self.g_convlayers_encoder:
            self.params += layer.params
            
        for layer in self.g_convlayers_decoder:
            self.params += layer.params
            
        ## collect all the normalization values 
        self.g_normalization_params = []
        
        for layer in self.g_convlayers_encoder:
            self.g_normalization_params += layer.normalization_params
            
        for layer in self.g_convlayers_decoder:
            self.g_normalization_params += layer.normalization_params
            
            
    def forward(self, X, is_training = True):
        output = X
        
        # Collect the outputs from the encoder
        self.encoder_outputs = []
        count00 = 0
        for layer in self.g_convlayers_encoder:
            output = layer.forward(output, is_training)
            count00 += 1
            self.encoder_outputs.append(output)
            # print(output.shape)
            
        count = 2
        # count_for_dropOut = 1  
        for layer in self.g_convlayers_decoder:
            output = layer.forward(output, is_training)

            if count <= count00:
                output = tf.concat([output, self.encoder_outputs[-count]], 3) # 3 is for the dimention we concate 

            # else:
            #     print("skip")
            # print(layer.f) 
            # if is_training == True and count_for_dropOut<= self.drop_out:
            #     output = tf.nn.dropout(output, 0.5)
            #     count_for_dropOut +=1
                # print('DO')
            # output =layer.f(output) 
            # print(np.min(output))
            count += 1
            
            # print(output.shape)
            # print(np.min(output))
        return output
        
        
### BUild discriminator 
         
class PatchGanDiscriminator(object):
    def __init__(self, d_sizes,  img_size, num_colors):
        self.d_convlayers = []
        dims = [img_size]
        dimx = img_size[0]
        dimy = img_size[1]

        
        mi = num_colors # Results and Boundarty conditions
        count = 0
        # mo, filtersz, stride, apply_batch_norm, f, padding, apply_zero_padding, Drop_out, add_bias
        for (mo, filtersz, stride, apply_batch_norm, activation,padding, apply_zero_padding, Drop_out ,add_bias) in d_sizes['conv_layers']:
            # mo = feature maps out
            name = "convlayer_%s" % count 
           
            # if apply_zero_padding:
                # dimx = dimx +2
                # dimy = dimy +2
            # dimx = int(np.ceil(float(dimx) /stride))
            # dimy = int(np.ceil(float(dimx) /stride))
           
            dimx = int(np.abs((dimx - 4 + 2 *1) / stride)+1) 
            dimy = int(np.abs((dimy - 4 + 2 *1) / stride)+1)
            # if apply_zero_padding:
            #     dimx = dimx +2
            #     dimy = dimy +2
                
            # print("Dimx:", dimx, "Dimy:", dimy)
            
            layer = ConvLayer(name, mi, mo, apply_batch_norm, dim_output_x= dimx,
                              dim_output_y= dimy, filtersz=filtersz, stride=stride,
                              f = activation , Pad = padding, apply_zero_padding = apply_zero_padding, Drop_out=Drop_out, add_bias=add_bias) #lrelu
            output_shape = [None, dimx,dimy,mo] 
            print("mi:", mi, ",mo:", mo, ",output shape", output_shape, ",Batch_norm:", apply_batch_norm, 
                  ",activation:", str(activation)[10:15],
                  "DO:", Drop_out, "bias:", add_bias, "Zero_pad:", apply_zero_padding)
            
            self.d_convlayers.append(layer)
            mi = mo
            count += 1
            
        ## BUild last layer
        name = "convlayer_%s" % count 
        
        mo, filtersz, stride, apply_batch_norm, activation,padding, apply_zero_padding, Drop_out, add_bias   = d_sizes['last_layers'][0]

        dimx = int(np.abs((dimx - 4 + 2 *1) / stride)+1)
        dimy = int(np.abs((dimy - 4 + 2 *1) / stride)+1)
        if apply_zero_padding:
            dimx = dimx +2
            dimy = dimy +2
        
        
            
        # print("Dimx:", dimx, "Dimy:", dimy)
        # Activation identity because we use sigmoid cross entropy function for the loss
        layer = ConvLayer(name, mi, mo, apply_batch_norm, dim_output_x= dimx,
                          dim_output_y= dimy, filtersz=filtersz, stride=stride, f = activation,
                          Pad = padding, apply_zero_padding= apply_zero_padding, Drop_out=Drop_out, add_bias= add_bias)#tf.identity
        
        output_shape = [None, dimx,dimy,mo] 
        print("mi:", mi, ",mo:", mo, ",output shape", output_shape, ",Batch_norm:", apply_batch_norm,
              ",activation:", str(activation)[10:18],
              "DO:", Drop_out, "bias:", add_bias,  "Zero_pad:", apply_zero_padding)
        
        
        self.d_convlayers.append(layer)
        
        #Collect all the trainable params 
        self.params = []
        
        for layer in self.d_convlayers:
            self.params += layer.params
            
        #collect the normalization params 
        
        self.d_normalization_params = []
        
        for layer in self.d_convlayers:
            self.d_normalization_params += layer.normalization_params
            
        
    def forward(self, X, is_training = False):
        
        output = X
        
        for layer in self.d_convlayers:
            output = layer.forward(output, is_training)
            # print(output.shape)
        logits = output
        return logits
            
        
    
class cGAN(object):
    def __init__(self, imgsize, num_colors_BC, num_colors_results, d_sizes, g_sizes, batch_size):
        
        # Save for later 
        self.d_sizes = d_sizes
        self.g_sizes = g_sizes
        self.num_colors_BC = num_colors_BC
        self.im_size = imgsize

        # Build Discriminator
        print("Build the Discriminator")
        #Getting the results and the boundary conditions matrices
        self.discriminator = PatchGanDiscriminator(d_sizes= d_sizes, img_size= imgsize,
                                                   num_colors= (num_colors_results + num_colors_BC))
        
        self.discriminator_trainable_params = self.discriminator.params
        
        self.discriminator_norm_params = self.discriminator.d_normalization_params
        
        # Build Generator  - > getting only the BC
        print("Build the generator")
        self.generator = Unet(g_sizes=g_sizes, img_size= imgsize, color= num_colors_BC, batch_size=batch_size)
        
        self.generator_trainable_params = self.generator.params
        
        self.generator_norm_params = self.generator.g_normalization_params
        
        
        
    def fit(self, X = [], Targets = [], Xtest = [], Ytest= [], epochs = 2, batch_size = 10, optimizers = [tf.keras.optimizers.Adam] *2,
            load = False,  file_name_dis_w = 'last', file_name_gen_w = 'last',  save_wights_during_training = True, 
            step_for_save_epochs = 1, take_generated_images_during_traning = True,no_of_save_image_per_epoch = 30,
            physics_matrix = True,  typee = "image", data_type = 'From_folder' ,lambdaa = 100):
        
        self.discriminator_optimizer = optimizers[0](lr = 0.0002, beta_1=0.5, beta_2=0.999)#should be 0.0002
        self.generator_optimizer  = optimizers[1](lr = 0.0002, beta_1=0.5, beta_2=0.999)
        
        self.lambdaa  = lambdaa #100
        
        if load:
            self.load(file_name_dis_w = file_name_dis_w, file_name_gen_w = file_name_gen_w, file_name_dis_norm_params = 'last',
                      file_name_gen_norm_params = 'last', 
                      load_norm_params = load)
            
        d_costs = []
        g_costs = []
        self.batch_size = batch_size   

        Y = Targets # should be a list of images if the data is to big 

        N = len(X) # if the data_type is too big X is the list of the images 
        n_batches = N // batch_size
        
        take_image = int(np.ceil(n_batches / no_of_save_image_per_epoch))
            
        pic_num = 0
        pic_num_test  = 0
        
        for epoch in range(epochs):
            print("epoch:", epoch)
            
            if data_type == 'From_folder':
                X = shuffle(X)
            else:
                
                X, Y = shuffle(X, Y)
                
            t0 = datetime.now()
            j=0
            if  save_wights_during_training ==True and   epoch % step_for_save_epochs == 0 or  save_wights_during_training ==True and epoch == epochs -1:
                print("Saving the Discriminator and generator weights")    
                self.save(epoch, j)
            
            # count_batches = 0
            for j in range(n_batches):
                
                if data_type == 'From_folder':
                    # if (j+1) % (500/batch_size) == 0 :
                    #     t3 = datetime.now()
                    #     print("Time to load new 500 images...")
                    #     batch_global = X[count_batches*500:(count_batches+1) * 500]
                        
                    #     batch_dataX, batch_dataY = get_shoose_data(batch_global)
                    #     print(datetime.now() - t3)
                    #     count_batches += 1
                    #     theta = 0
                    batch = X[j*batch_size:(j+1)*batch_size]
                    # print("Load images ...")
                    # t3 = datetime.now()
                    Xbatch, Ybatch = get_shoose_data_normelized_1_to_1(batch)#get_shoose_data_normelized_1_to_1_with_mirror(batch)
                    # print("Time for loading:" ,datetime.now() - t3)
                    
                    # Xbatch = batch_dataX[theta*batch_size:(theta+1)*batch_size]
                    # Ybatch = batch_dataY[theta*batch_size:(theta+1)*batch_size]
                    # theta += 1
                    #Normalization 
                    # Xbatch = Xbatch/255.0
                    # Ybatch = Ybatch/255.0
                    
                else: 
                     
                    Xbatch = X[j*batch_size:(j+1) * batch_size]
                    Ybatch = Y[j*batch_size: (j+1) *batch_size]
   
                # save weights epoch % step_for_save_epochs == 0 and
                
                # add_noise_to_X : Z ~N(0,0.2)
                # Z = tf.random.normal(shape = Xbatch.shape, mean=0, stddev=1)
                
                # X_batch_noise = tf.concat([Xbatch, Z], axis = 3)
                # Train the discriminator 
                
                logits_real, logits_fake, d_cost = self.train_step_discriminator(Xbatch  , Ybatch)
                
                # Train the generator 
                g_costs1 = self.train_step_generator(Xbatch, Ybatch)
                # g_costs2 = self.train_step_generator(Xbatch, Ybatch)
                
                # g_cost = (g_costs1+ g_costs2) / 2
                g_cost = g_costs1
                g_costs.append(tf.reduce_mean(g_cost))
                
                d_costs.append(tf.reduce_mean(d_cost))
                
                if  save_wights_during_training ==True and  j % 1000 == 0:
                    print("Saving the Discriminator and generator weights")    
                    self.save(epoch, j )
                

                
                if j % 50 == 0 :
                   d_acc = self.discriminator_accuracy(logits_real, logits_fake)
                   print("Epoch: %d, Batch: %d/%d, time: %s, d_acc: %.2f, d_cost: %.2f, g_cost: %.2f" %
                         (epoch, j, n_batches, datetime.now() - t0, d_acc, tf.reduce_mean(d_cost),
                          tf.reduce_mean(g_cost)))
                   t0 = datetime.now()
                   
                if j % take_image  ==0 and take_generated_images_during_traning == True:
                   print("Save train image...")

                   self.images_of_samples_from_generator_contour(Xbatch[1:5,:,:,:], Ybatch[1:5,:,:,:], typee = typee )
                   # plt.savefig('samples/samples_at_iter_%d_%d.png' % (j, epoch))
                   plt.savefig('samples/train/samples_at_iter_%d.png' % (pic_num))
                   pic_num += 1
                   plt.close('all')
                   
                   
                if j == take_image and take_generated_images_during_traning == True:
                   print("Save test image...")
                   test = random.sample(Xtest,5)
                   X_test, Y_test = get_shoose_data_normelized_1_to_1(test)#get_shoose_data_normelized_1_to_1_with_mirror(test)

                   self.images_of_samples_from_generator_contour(X_test,Y_test , typee = typee )
                   # plt.savefig('samples/samples_at_iter_%d_%d.png' % (j, epoch))
                   plt.savefig('samples/test/samples_at_iter_%d.png' % (pic_num))
                   pic_num_test += 1
                   plt.close('all')
                   
                   
        print("Saving the Discriminator and generator weights")  
        self.save(epoch, j)          
        plt.clf()
        plt.plot(d_costs, label = 'discriminator cost')
        plt.plot(g_costs, label = 'generator cost')
        plt.legend()
        plt.savefig('cost_vs_iteration.png')
        print("Done, the model was trained as you wish!")
                   
               
                   
                
    
    def train_step_discriminator(self, X, Targets):
        
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            
            logits_real, logits_fake, cost_dis = self.discriminator_cost( X, Targets, is_training = True)
            
        d_gradients = tape.gradient(cost_dis, self.discriminator_trainable_params)
        self.discriminator_optimizer.apply_gradients(zip(d_gradients, self.discriminator_trainable_params))
        
        return logits_real, logits_fake, cost_dis
    
    def train_step_generator(self, X, Targets):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            _ , g_cost = self.generator_cost(X, Targets, is_training = True)
            
        g_gradients = tape.gradient(g_cost, self.generator_trainable_params)
        self.generator_optimizer.apply_gradients(zip(g_gradients, self.generator_trainable_params))
        
        return g_cost

        
        
    def get_sample_from_generator(self, Z, is_training): # Z is the BC's
        return self.generator.forward(Z, is_training=is_training)
    
    def discriminator_logits(self, X, is_training):# X is BC's and the results
        logits = self.discriminator.forward(X, is_training = is_training)
        
        # flatten 
        # logits = tf.reshape(logits, shape = [logits.shape[0] , logits.shape[1]*logits.shape[2]
                                             # *logits.shape[3]])
        
        return logits
    
    def discriminator_cost_real_images(self, X, Targets, is_training): #X should be [BC, results]
        Z = tf.concat([X,Targets],3)
        logits = self.discriminator_logits(Z, is_training)
        # flatten 
        # logits = tf.reshape(logits, shape = [logits.shape[0] , logits.shape[1]*logits.shape[2]
        #                                      *logits.shape[3]])
        #smoothing
        D =tf.random.uniform(shape = [1], minval= 0.8 , maxval=1)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,
                                                       labels = tf.ones_like(logits))#*D)
        
        #remember from the sample X all the values should be True which == 1
        return logits, cost 
     
    def discriminator_cost_fake_images(self, X, is_training):
        sample_from_generator = self.get_sample_from_generator(X, is_training) # Z is BC
        # now we need to concatonate the REAL results
        input_for_discriminator = tf.concat([X,sample_from_generator],3)
        
        logits = self.discriminator_logits(input_for_discriminator, is_training)
        
        # # flatten 
        # logits = tf.reshape(logits, shape = [logits.shape[0] , logits.shape[1]*logits.shape[2]
        #                                      *logits.shape[3]])
        #smoothing
        D =tf.random.uniform(shape = [1], minval= 0 , maxval=0.2)
        
        #I removed the tf_reduce mean
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, 
                                                       labels = tf.zeros_like(logits))#*D)
        
        
        #remember all the images from the generator is fake 
        return logits , cost , sample_from_generator
        
    
    def discriminator_cost(self, X, Targets, is_training):
         
        #min max cost
         logits_fake, cost_fake , sample_from_generator= self.discriminator_cost_fake_images(X, is_training)
         logits_real, cost_real = self.discriminator_cost_real_images(X, Targets, is_training)
         cost_min_max = (cost_fake + cost_real)
         
         ## Add the L1 
         # cost_L1 = tf.reduce_mean(tf.math.abs(Targets- sample_from_generator))
         
         cost_dis =  cost_min_max #+ self.lambdaa * cost_L1 
         return  logits_real, logits_fake, cost_dis
    
    def generator_cost(self, Z,Targets, is_training):#Z = BC
        sample_from_generator = self.get_sample_from_generator(Z, is_training)
        
        # now we need to concatonate the REAL results
        input_for_discriminator = tf.concat([Z,sample_from_generator],3)
        
        logits = self.discriminator_logits(input_for_discriminator, is_training)
        
        # # flatten 
        # logits = tf.reshape(logits, shape = [logits.shape[0] , logits.shape[1]*logits.shape[2]
        #                                      *logits.shape[3]])
        #smoothing
        D =tf.random.uniform(shape = [1], minval= 0.8 , maxval=1)
        cost =  tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,
                                                       labels = tf.ones_like(logits) )#*D )
        
        # because if the discriminator will say that it is a true image we are good !!!
        #Add L1 loss
        # cost_L1 = tf.reduce_mean(tf.math.abs(Targets- sample_from_generator))
        cost_L1 = tf.reduce_mean(tf.keras.losses.mean_absolute_error(Targets,sample_from_generator))
        cost_generator=  cost  + (self.lambdaa * cost_L1 )
        
        return logits , cost_generator
    
    
    def discriminator_accuracy(self, logits_real, logits_fake):
        # Remember the final dense layer of the discriminator is without activation 
        logits_real = tf.reshape(logits_real, shape = [logits_real.shape[0] , logits_real.shape[1]*logits_real.shape[2]
                                              *logits_real.shape[3]])
        logits_fake= tf.reshape(logits_fake, shape = [logits_fake.shape[0] , logits_fake.shape[1]*logits_fake.shape[2]
                                              *logits_fake.shape[3]])
        real_prediction = tf.cast(logits_real > 0 , tf.float32)
        fake_prediction =tf.cast(logits_fake < 0 , tf.float32)
        
        num_predictions = 2.0 * self.batch_size *logits_fake.shape[1]#* (logits_fake.shape[1] * logits_fake.shape[2])
        
        num_correct = tf.reduce_sum(real_prediction) + tf.reduce_sum(fake_prediction)
        
        return num_correct / num_predictions
    
    def images_of_samples_from_generator(self,Z, Targets): # Z is the BC and Tergets are the results 
        # n -> number of images 
        
        samples = self.get_sample_from_generator(Z, is_training = False)
        samples = samples.numpy()
        n = samples.shape[0]
        
        dimy =self.im_size[0] 
        dimx = self.im_size[1]
        
        flat_image1 = np.empty((dimy*2 , dimx * n))
        flat_image2 = np.empty((dimy*2 , dimx * n))
        k = 0
        
        for i in range(n):
            flat_image1[ 0 * dimy : dimy, i* dimx: (1+i) * dimx] = samples[k,:,:,0]
            flat_image2[ 0 * dimy : (1) *dimy, i* dimx: (i+1) * dimx] = samples[k,:,:,1]
            flat_image1[ 1 * dimy : 2 *dimy, i* dimx: (1+i) * dimx] = Targets[k,:,:,0]
            flat_image2[ 1 * dimy : (2) *dimy, i* dimx: (i+1) * dimx] = Targets[k,:,:,1]
            k +=1 

                
        plt.imshow(flat_image1)
                
    def images_of_samples_from_generator_contour(self,Z, Targets, typee = "image"): # Z is the BC and Tergets are the results 
        # n -> number of images 
        
        samples = self.get_sample_from_generator(Z, is_training = False)
        samples = samples.numpy()
        
        n = samples.shape[0]
        

        fig, axs = plt.subplots(3, n)
        
        k = 0
        # print(np.min(np.min(Z)))
        # print(np.min(np.min(Targets)))        
        # print(np.min(np.min(samples)))
        
        samples = back_to_0_1(samples)
        Z = back_to_0_1(Z)
        Targets = back_to_0_1(Targets)
        
        if typee == "image":
            print('Image')
            if n == 1:
                    axs[0].imshow(samples[k,:,:,:])#.astype(np.uint8))
                    axs[0].set_title('Fake')
                    axs[1].imshow(Targets[k,:,:,:])#.astype(np.uint8))
                    axs[1].set_title('Real')
                    axs[2].imshow(Z[k,:,:,0], cmap = 'gray')
                    # axs[2, i].imshow(Z[k,:,:,:])
                    axs[2].set_title('Input')
            else:
                for i in range(n):
                    axs[0,i].imshow(samples[k,:,:,:])#.astype(np.uint8))
                    axs[0, i].set_title('Fake')
                    axs[1,i].imshow(Targets[k,:,:,:])#.astype(np.uint8))
                    axs[1, i].set_title('Real')
                    axs[2, i].imshow(Z[k,:,:,0], cmap = 'gray')
                    # axs[2, i].imshow(Z[k,:,:,:])
                    axs[2,i].set_title('Input')
                    k += 1
            
        else:
                    
            for i in range(n):
                axs[0,i].contourf(samples[k,:,:,0])
                axs[0, i].set_title('Fake')
                axs[1,i].contourf(Targets[k,:,:,0])
                axs[1, i].set_title('Real')
                k += 1
        
        
        return 
            
        
    def save(self, n_epoch, j):
        exist_folder = os.path.isdir('weights_and_layers')
        
        if exist_folder == False :
            path = os.getcwd() + '\\weights_and_layers'
            path_dis = os.getcwd() + '\\weights_and_layers\\discriminator'
            path_generator = os.getcwd() + '\\weights_and_layers\\generator'
            try:
                os.mkdir(path)
                os.mkdir(path_dis)
                os.mkdir(path_generator)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully creat the directory %s" % path)
                
        exist_folder_for_normalization = os.path.isdir('normalization_params')
        
        if exist_folder_for_normalization == False:
            path = os.getcwd() + '\\normalization_params'
            path_dis = path + '\\discriminator'
            path_generator = path + '\\generator'
            
            try:
                os.makedirs(path)
                os.makedirs(path_dis)
                os.makedirs(path_generator)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully creat the directory %s" % path)
        
        ## Save the weights and the normalization consts
        file_name_dis = 'weights_and_layers/' 'discriminator/' 'discriminator_trainable_params.dic_epoch_' + str(n_epoch)+ '_'+str(j)
        file_name_gen = 'weights_and_layers/' 'generator/' 'generator_trainable_params.dic_epoch_' + str(n_epoch) + '_' + str(j)
        with open(file_name_dis,'wb') as model_params_dis:
            pickle.dump(self.discriminator_trainable_params, model_params_dis) # save teh model in binary style ('wb) as file name model.dic
        
        with open(file_name_gen, 'wb') as model_params_gen:
            pickle.dump(self.generator_trainable_params, model_params_gen)
       
        
        file_name_dis_norm = 'normalization_params/' 'discriminator/' 'discriminator_norm_params.dic_epoch_' + str(n_epoch) + '_'+ str(j)
        file_name_gen_norm = 'normalization_params/' 'generator/' 'generator_norm_params.dic_epoch_' + str(n_epoch) + '_'+ str(j)
        with open(file_name_dis_norm, 'wb') as model_norm_param:
            # self.discriminator_norm_params = self.discriminator.collect_running_params()
            pickle.dump(self.discriminator_norm_params, model_norm_param)
            
        with open(file_name_gen_norm,'wb') as model_norm_param:
            # self.generator_norm_params = self.generator.collect_running_params()
            pickle.dump(self.generator_norm_params, model_norm_param)
            
        # Keep only the last 6 (the most updated) - weights
        
        # for filename in sorted(os.listdir("weights_and_layers/discriminator/"))[:-2]:
        #     filename_relPath = os.path.join("weights_and_layers/discriminator/",filename)
        #     os.remove(filename_relPath)
            
        paths_dis = sorted(Path("weights_and_layers/discriminator/").iterdir(), key=os.path.getmtime)
        for path in  paths_dis[:-5]:
            os.remove(path)  
        
        # Keep only the last 6
        # for filename in sorted(os.listdir("weights_and_layers/generator/"))[:-2]:
        #     filename_relPath = os.path.join("weights_and_layers/generator/",filename)
        #     os.remove(filename_relPath)     
        
        paths_ge = sorted(Path("weights_and_layers/generator/").iterdir(), key=os.path.getmtime)
        for path in  paths_ge[:-5]:
            os.remove(path)  
            
            
        paths_dis_norm = sorted(Path("normalization_params/discriminator/").iterdir(), key=os.path.getmtime)
        for path in  paths_dis_norm[:-5]:
            os.remove(path)  
        
        # Keep only the last 6
        # for filename in sorted(os.listdir("weights_and_layers/generator/"))[:-2]:
        #     filename_relPath = os.path.join("weights_and_layers/generator/",filename)
        #     os.remove(filename_relPath)     
        
        paths_ge_norm = sorted(Path("normalization_params/generator/").iterdir(), key=os.path.getmtime)
        for path in  paths_ge_norm[:-5]:
            os.remove(path)  
            
        
            
    def load(self, file_name_dis_w = 'last', file_name_gen_w = 'last', file_name_dis_norm_params = 'last',
             file_name_gen_norm_params = 'last', load_norm_params = False):
        
        if file_name_dis_w == 'last':
            list_of_files = glob.glob('weights_and_layers/discriminator/*')
            latest_file = max(list_of_files, key=os.path.getctime)
              
            path_weights = latest_file
            print('Laods:', path_weights[33:])
                       
            try:
                with open(path_weights,'rb') as model_params_dis:
                    params_loaded_dis = pickle.load(model_params_dis)

                    
            except OSError:
                print('Filed to load the saved weights for discriminator!')
            else:
                print('Saved discriminator weights were loaded!')
                
        else:
            path_weights = 'weights_and_layers/discriminator/' + str(file_name_dis_w)
            try:
               with open(path_weights, 'rb') as model_params_dis:
                    params_loaded_dis = pickle.load(model_params_dis)
            except OSError:
                print('Filed to load the saved weights for discriminator!')
            else:
                print('Saved discriminator weights were loaded!')   
                
        count = 0
        for param in self.discriminator_trainable_params:
            param.assign(params_loaded_dis [count])
            count += 1 
            
            
        if file_name_gen_w == 'last':
            list_of_files = glob.glob('weights_and_layers/generator/*')
            latest_file = max(list_of_files, key=os.path.getctime)
            
            path = latest_file
            print('Laods:', path[29:])
            try:
                with open(path,'rb') as model_params_gen:
                    params_loaded_gen = pickle.load(model_params_gen)
            except OSError:
                print('Filed to load the saved weights for discriminator!')
            else:
                print('Saved generator weights were loaded!')
                
        else:
            path = 'weights_and_layers/generator/' + str(file_name_gen_w)
            try:
               with open(path, 'rb') as model_params_gen:
                     params_loaded_gen = pickle.load(model_params_gen)
            except OSError:
                print('Filed to load the saved weights for discriminator!')
            else:
                print('Saved generator weights were loaded!')   
                
        count = 0
        for param in self.generator_trainable_params:
            param.assign(params_loaded_gen[count])
            count += 1 
                
                    
                    
            
        if load_norm_params:
            
            if file_name_dis_norm_params == 'last':               
                list_of_files_norm = glob.glob('normalization_params/discriminator/*')
                latest_file_norm = max(list_of_files_norm, key=os.path.getctime)
                path_norm = latest_file_norm
                print('Loads:', path_norm[35:])
           
                
                try:       
                    with open(path_norm, 'rb') as model_norm_dis:
                        params_norm_dis = pickle.load(model_norm_dis)
                        
                except OSError:
                    print('Filed to load the saved norm params for discriminator!')
                else:
                    print('Saved discriminator norm params were loaded!')
                    
            else:
                path_norm = 'normalization_params/discriminator/' + str(file_name_dis_norm_params)
                try:
                   with open(path_norm, 'rb') as model_norm_dis:
                        params_norm_dis = pickle.load(model_norm_dis)
                except OSError:
                    print('Filed to load the saved norm params for discriminator!')
                else:
                    print('Saved discriminator norm params were loaded!')   
                    
            count = 0
            for norm_param in self.discriminator_norm_params:
                print(params_norm_dis[count].name)
                norm_param.assign(params_norm_dis[count])
     
                count += 1 
               
                
            if file_name_gen_norm_params == 'last':
                list_of_files = glob.glob('normalization_params/generator/*')
                latest_file = max(list_of_files, key=os.path.getctime)
                
                path = latest_file
                print('Laods:', path[31:])
                try:
                    with open(path,'rb') as model_norm_params_gen:
                        params_norm_gen = pickle.load(model_norm_params_gen)
                except OSError:
                    print('Filed to load the saved weights for discriminator!')
                else:
                    print('Saved generator weights were loaded!')
                    
            else:
                path = 'normalization_params/generator/' + str(file_name_gen_norm_params)
                try:
                   with open(path, 'rb') as model_norm_params_gen:
                         params_norm_gen = pickle.load(model_norm_params_gen)
                except OSError:
                    print('Filed to load the saved weights for discriminator!')
                else:
                    print('Saved generator weights were loaded!')   
                    
            count = 0
            for param in self.generator_norm_params:
                param.assign(params_norm_gen[count])
                count += 1 
        
    
        
        
        
