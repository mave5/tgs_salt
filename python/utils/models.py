from keras.layers import Input, merge, Convolution2D, Deconvolution2D
from keras.layers import AtrousConvolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Dense
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import layer_utils
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Concatenate
from keras.layers import Reshape,Permute
from keras.layers.advanced_activations import ELU,PReLU
from keras.regularizers import l2
from keras.layers import ZeroPadding2D
from keras.layers import Cropping2D


def conv2dcustom(x_input,filters=8,kernel_size=3,strides=1,w2reg=None,pool=False,padding='same',batchNorm=False,activation='relu',data_format='channels_first'):
    x1 = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,data_format=data_format,kernel_regularizer=w2reg,strides=strides)(x_input)
    if batchNorm==True:
        x1=BatchNormalization(axis=1)(x1)
        
    if activation=='leaky':        
        x1 = LeakyReLU(0.1)(x1)
    else:
        x1=Activation('relu')(x1)        

    if pool==True:
        x1=MaxPooling2D(pool_size=(2, 2),data_format=data_format)(x1)
    
    return x1


# model
def model_skip(params):

    h=params['h']
    w=params['w']
    z=params['z']
    lr=params['learning_rate']
    loss=params['loss']
    C=params['initial_channels']
    numOfOutputs=params['numOfOutputs']
    dropout_rate=params['dropout_rate']
    data_format=params["data_format"]
    batchNorm=params['batchNorm']
    w2reg=params['w2reg']
    initStride=params['initStride']
    reshape4softmax=params['reshape4softmax']
    
    
    inputs = Input((z,h, w))
    inputsPadded=ZeroPadding2D(padding=((13,14), (13,14)),data_format=data_format)(inputs)
    conv1=conv2dcustom(filters=C,x_input=inputsPadded,strides=initStride,w2reg=w2reg,activation='leaky')    
    pool1=conv2dcustom(filters=C,x_input=conv1,w2reg=w2reg,pool=True,activation='leaky')    

    conv2=conv2dcustom(filters=2*C,x_input=pool1,w2reg=w2reg,activation='leaky')    
    pool2=conv2dcustom(filters=2*C,x_input=conv2,w2reg=w2reg,pool=True,activation='leaky')    
    
    conv3=conv2dcustom(filters=4*C,x_input=pool2,w2reg=w2reg,activation='leaky')    
    pool3=conv2dcustom(filters=4*C,x_input=conv3,w2reg=w2reg,pool=True,activation='leaky')    

    conv4=conv2dcustom(filters=8*C,x_input=pool3,w2reg=w2reg,activation='leaky')    
    pool4=conv2dcustom(filters=8*C,x_input=conv4,w2reg=w2reg,pool=True,activation='leaky')    

    conv5=conv2dcustom(filters=16*C,x_input=pool4,w2reg=w2reg,activation='leaky')    
    conv5=conv2dcustom(filters=16*C,x_input=conv5,w2reg=w2reg,pool=False,activation='leaky')    
    
    # dropout
    conv5 =Dropout(dropout_rate)(conv5)
    
    up7=UpSampling2D(size=(2, 2),data_format=data_format)(conv5)
    concat = Concatenate(axis=1)
    up7 = concat([up7, conv4])

    conv7=conv2dcustom(filters=8*C,x_input=up7,w2reg=w2reg,pool=False,activation='leaky')    
    
    up8 = concat([UpSampling2D(size=(2, 2),data_format=data_format)(conv7), conv3])

    conv8=conv2dcustom(filters=4*C,x_input=up8,w2reg=w2reg,pool=False,activation='leaky')        
    
    up9 = concat([UpSampling2D(size=(2, 2),data_format=data_format)(conv8), conv2])
    
    conv9=conv2dcustom(filters=2*C,x_input=up9,w2reg=w2reg,pool=False,activation='leaky')        

    up10 = concat([UpSampling2D(size=(2, 2),data_format=data_format)(conv9), conv1])
    conv10=conv2dcustom(filters=C,x_input=up10,w2reg=w2reg,pool=False,activation='leaky')        

    conv10 = UpSampling2D(size=(initStride, initStride),data_format=data_format)(conv10)
    conv10=conv2dcustom(filters=C,x_input=conv10,w2reg=w2reg,pool=False,activation='leaky')        
    
    conv10 = Conv2D(numOfOutputs, 1, data_format=data_format,kernel_regularizer=w2reg)(conv10)
    conv10=Cropping2D(cropping=((13,14),(13,14)),data_format=data_format)(conv10)

    if reshape4softmax:
        # reshape for softmax
        output=Reshape((numOfOutputs,h*w)) (conv10)
        # permute for softmax
        output=Permute((2,1))(output)
        # softmax
        output=Activation('softmax')(output)
    else:        
        output=Activation('sigmoid')(conv10)
    
    model = Model(inputs=inputs, outputs=output)

    if loss=='dice':
        model.compile(optimizer=Adam(lr), loss=dice_coef_loss, metrics=[dice_coef])
    else:
        #model.compile(loss='binary_crossentropy', optimizer=Adam(lr))
        model.compile(loss=loss, optimizer=Adam(lr))
    
    return model




# model
def model_classification(params):

    h=params['h']
    w=params['w']
    z=params['z']
    lr=params['learning_rate']
    loss=params['loss']
    C=params['initial_channels']
    numOfOutputs=params['numOfOutputs']
    dropout_rate=params['dropout_rate']
    data_format=params["data_format"]
    batchNorm=params['batchNorm']
    w2reg=params['w2reg']
    initStride=params['initStride']
    
    
    inputs = Input((z,h, w))
    #inputsPadded=ZeroPadding2D(padding=((13,14), (13,14)),data_format="channels_first")(inputs)
    conv1=conv2dcustom(filters=C,x_input=inputs,strides=initStride,w2reg=w2reg,activation='leaky',data_format=data_format)    
    pool1=conv2dcustom(filters=C,x_input=conv1,w2reg=w2reg,pool=True,activation='leaky')    

    conv2=conv2dcustom(filters=2*C,x_input=pool1,w2reg=w2reg,activation='leaky')    
    pool2=conv2dcustom(filters=2*C,x_input=conv2,w2reg=w2reg,pool=True,activation='leaky')    
    
    conv3=conv2dcustom(filters=4*C,x_input=pool2,w2reg=w2reg,activation='leaky')    
    pool3=conv2dcustom(filters=4*C,x_input=conv3,w2reg=w2reg,pool=True,activation='leaky')    

    conv4=conv2dcustom(filters=8*C,x_input=pool3,w2reg=w2reg,activation='leaky')    
    pool4=conv2dcustom(filters=8*C,x_input=conv4,w2reg=w2reg,pool=True,activation='leaky')    

    conv5=conv2dcustom(filters=16*C,x_input=pool4,w2reg=w2reg,activation='leaky')    
    conv5=conv2dcustom(filters=16*C,x_input=conv5,w2reg=w2reg,pool=False,activation='leaky')    
    
    # flatten
    flattenConv5=Flatten()(conv5)
    
    # dropout
    flattenConv5 =Dropout(dropout_rate)(flattenConv5)
    
    
    output=Dense(numOfOutputs,activation="sigmoid")(flattenConv5)
    
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=loss, optimizer=Adam(lr))
    
    return model


