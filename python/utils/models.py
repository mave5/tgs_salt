from keras.layers import Input, merge, Convolution2D, Deconvolution2D
from keras.layers import AtrousConvolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Dense
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,RMSprop,Nadam
from keras.utils import layer_utils
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Concatenate
from keras.layers import Reshape,Permute
from keras.layers.advanced_activations import ELU,PReLU
from keras.regularizers import l2
from keras.layers import ZeroPadding2D
from keras.layers import Cropping2D
from keras.layers import AveragePooling2D

def conv2dcustom(x_input,filters=8,kernel_size=3,strides=1,w2reg=None,pool=False,padding='same',batchNorm=False,activation='relu',data_format='channels_first'):
    x1 = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,data_format=data_format,kernel_regularizer=w2reg,strides=strides)(x_input)
    if batchNorm==True:
        x1=BatchNormalization(axis=1)(x1)
        
    if activation=='leaky':        
        x1 = LeakyReLU(0.1)(x1)
    elif activation=='relu':
        x1=Activation(activation)(x1)        
    elif activation=='sigmoid':
        x1=Activation('sigmoid')(x1)        
    else:
        pass

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
    if w2reg==True:
        w2reg=l2(1e-4)
    
    initStride=params['initStride']
    reshape4softmax=params['reshape4softmax']
    optimizer=params["optimizer"]
    
    
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

    if reshape4softmax== True:
        # reshape for softmax
        output=Reshape((numOfOutputs,h*w)) (conv10)
        # permute for softmax
        output=Permute((2,1))(output)
        # softmax
        output=Activation('softmax')(output)
    else:        
        output=Activation('sigmoid')(conv10)
    
    model = Model(inputs=inputs, outputs=output)

    if optimizer=='RMSprop':
        optimizer = RMSprop(lr)
    elif optimizer=='Adam':       
        optimizer = Adam(lr)
    elif optimizer=='Nadam':       
        optimizer = Nadam(lr,clipvalue=1.0)        

    if loss=='dice':
        model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])
    else:
        model.compile(loss=loss, optimizer=optimizer)
        
    return model



# model
def model_skip2(params):

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
    cropping_padding=params['cropping_padding']
    w2reg=params['w2reg']
    if w2reg==True:
        w2reg=l2(1e-4)
    else:
        w2reg=None
    
    initStride=params['initStride']
    reshape4softmax=params['reshape4softmax']
    optimizer=params["optimizer"]
    
    
    inputs = Input((z,h, w))
    inputsPadded=ZeroPadding2D(padding=(cropping_padding, cropping_padding),data_format=data_format)(inputs)
    
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
    
    conv10 = Conv2D(numOfOutputs, 1, data_format=data_format,kernel_regularizer=w2reg,activation=None)(conv10)
    conv10=Cropping2D(cropping=((13,14),(13,14)),data_format=data_format)(conv10)

    if reshape4softmax== True:
        # reshape for softmax
        output=Reshape((numOfOutputs,h*w)) (conv10)
        # permute for softmax
        output=Permute((2,1))(output)
        # softmax
        output=Activation('softmax')(output)
    else:        
        output=Activation('sigmoid')(conv10)
    
    model = Model(inputs=inputs, outputs=output)

    if optimizer=='RMSprop':
        optimizer = RMSprop(lr)
    elif optimizer=='Adam':       
        optimizer = Adam(lr)
    elif optimizer=='Nadam':       
        optimizer = Nadam(lr,clipvalue=1.0)        

    if loss=='dice':
        model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])
    else:
        model.compile(loss=loss, optimizer=optimizer)
        

        
    
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
    if w2reg==True:
        w2reg=l2(1e-4)

    initStride=params['initStride']
    padding=params["padding"]
    padding="same"
    
    inputs = Input((z,h, w))
    conv1=conv2dcustom(filters=C,x_input=inputs,strides=initStride,w2reg=w2reg,activation='leaky',data_format=data_format,padding=padding)    
    pool1=conv2dcustom(filters=C,x_input=conv1,w2reg=w2reg,pool=True,activation='leaky',padding=padding)    

    conv2=conv2dcustom(filters=2*C,x_input=pool1,w2reg=w2reg,activation='leaky',padding=padding)    
    pool2=conv2dcustom(filters=2*C,x_input=conv2,w2reg=w2reg,pool=True,activation='leaky',padding=padding)    
    
    conv3=conv2dcustom(filters=4*C,x_input=pool2,w2reg=w2reg,activation='leaky',padding=padding)    
    pool3=conv2dcustom(filters=4*C,x_input=conv3,w2reg=w2reg,pool=True,activation='leaky',padding=padding)    

    conv4=conv2dcustom(filters=8*C,x_input=pool3,w2reg=w2reg,activation='leaky',padding=padding)    
    pool4=conv2dcustom(filters=8*C,x_input=conv4,w2reg=w2reg,pool=True,activation='leaky',padding=padding)    

    conv5=conv2dcustom(filters=16*C,x_input=pool4,w2reg=w2reg,activation='leaky',padding=padding)    
    conv5=conv2dcustom(filters=16*C,x_input=conv5,w2reg=w2reg,pool=False,activation='leaky',padding=padding)    
    
    # flatten
    flattenConv5=Flatten()(conv5)
    
    # dropout
    flattenConv5 =Dropout(dropout_rate)(flattenConv5)
    
    
    output=Dense(numOfOutputs,activation="sigmoid")(flattenConv5)
    
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=loss, optimizer=Adam(lr))
    
    return model


def model_classification2(params):

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
    if w2reg==True:
        w2reg=l2(1e-4)
    initStride=params['initStride']
    padding=params["padding"]
    optimizer=params["optimizer"]
    
    inputs = Input((z,h, w))
    conv1=conv2dcustom(filters=C,x_input=inputs,strides=initStride,w2reg=w2reg,activation='leaky',data_format=data_format,padding=padding)    
    pool1=conv2dcustom(filters=C,x_input=conv1,w2reg=w2reg,pool=True,activation='leaky',padding=padding)    

    conv2=conv2dcustom(filters=2*C,x_input=pool1,w2reg=w2reg,activation='leaky',padding=padding)    
    pool2=conv2dcustom(filters=2*C,x_input=conv2,w2reg=w2reg,pool=True,activation='leaky',padding=padding)    
    
    conv3=conv2dcustom(filters=4*C,x_input=pool2,w2reg=w2reg,activation='leaky',padding=padding)    
    pool3=conv2dcustom(filters=4*C,x_input=conv3,w2reg=w2reg,pool=True,activation='leaky',padding=padding)    

    conv4=conv2dcustom(filters=8*C,x_input=pool3,w2reg=w2reg,activation='leaky',padding=padding)    
    conv4=conv2dcustom(filters=8*C,x_input=conv4,w2reg=w2reg,pool=False,activation='leaky',padding=padding)    

    #conv5=conv2dcustom(filters=16*C,x_input=pool4,w2reg=w2reg,activation='leaky',padding=padding)    
    #conv5=conv2dcustom(filters=16*C,x_input=conv5,w2reg=w2reg,pool=False,activation='leaky',padding=padding)    
    
    # flatten
    flattenConv5=Flatten()(conv4)
    
    # dropout
    flattenConv5 =Dropout(dropout_rate)(flattenConv5)
    
    
    output=Dense(numOfOutputs,activation="sigmoid")(flattenConv5)
    
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=loss, optimizer=Adam(lr))
    
    return model




def model_classification_skip(params):
    
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
    if w2reg==True:
        w2reg=l2(1e-4)
    initStride=params['initStride']
    padding=params["padding"]
    optimizer=params["optimizer"]
    
    xin = Input((z,h, w))
    x1=conv2dcustom(filters=C,x_input=xin,strides=initStride,w2reg=w2reg,activation='leaky',data_format=data_format,padding=padding,pool=True)    
    x1=conv2dcustom(filters=C,x_input=x1,w2reg=w2reg,activation='leaky',padding=padding)    
    x1_ident = AveragePooling2D(pool_size=(2*initStride, 2*initStride))(xin)
    x1_merged = merge([x1, x1_ident],mode='concat', concat_axis=1)

    x2_1=conv2dcustom(filters=3*C,x_input=x1_merged,pool=True,w2reg=w2reg,activation='leaky',padding=padding)    
    x2_1=conv2dcustom(filters=3*C,x_input=x2_1,w2reg=w2reg,activation='leaky',padding=padding)    
    x2_ident = AveragePooling2D()(x1_ident)
    x2_merged = merge([x2_1,x2_ident],mode='concat', concat_axis=1)

    #by branching we reduce the #params
    x3_1 = conv2dcustom(filters=8*C,x_input=x2_merged,pool=True,w2reg=w2reg,activation='leaky',padding=padding)    
    x3_1 = conv2dcustom(filters=8*C,x_input=x3_1,w2reg=w2reg,activation='leaky',padding=padding)    
    x3_ident = AveragePooling2D()(x2_ident)
    x3_merged = merge([x3_1,x3_ident],mode='concat', concat_axis=1)

    x4_1 = conv2dcustom(filters=9*C,x_input=x3_merged,pool=True,w2reg=w2reg,activation='leaky',padding=padding)    
    x4_1 = conv2dcustom(filters=9*C,x_input=x4_1,w2reg=w2reg,activation='leaky',padding=padding)    
    x4_ident = AveragePooling2D()(x3_ident)
    x4_merged = merge([x4_1,x4_ident],mode='concat', concat_axis=1)    

    x5_1 = conv2dcustom(filters=9*C,x_input=x4_merged,pool=False,w2reg=w2reg,activation='leaky',padding="same")    
    x5_1 = conv2dcustom(filters=9*C,x_input=x5_1,w2reg=w2reg,activation='leaky',padding="valid")    
    
    # last layer of encoding    
    flatten_x=Flatten() (x5_1)
    
    flatten_x =Dropout(dropout_rate)(flatten_x)
    
    output=Dense(numOfOutputs,activation="sigmoid",W_regularizer=w2reg)(flatten_x)
    
    model = Model(inputs=xin, outputs=output)

    
    if optimizer=='RMSprop':
        optimizer = RMSprop(lr)
    elif optimizer=='Adam':       
        optimizer = Adam(lr)
    elif optimizer=='Nadam':       
        optimizer = Nadam(lr,clipvalue=1.0)        
    model.compile(loss=loss, optimizer=optimizer)

    return model


def model_classification_skip2(params):
    
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
    if w2reg==True:
        w2reg=l2(1e-4)
    initStride=params['initStride']
    padding=params["padding"]
    optimizer=params["optimizer"]
    
    xin = Input((z,h, w))
    x1=conv2dcustom(filters=C,x_input=xin,strides=initStride,w2reg=w2reg,activation='leaky',data_format=data_format,padding=padding,pool=True)    
    x1=conv2dcustom(filters=C,x_input=x1,w2reg=w2reg,activation='leaky',padding=padding)    
    x1_ident = AveragePooling2D(pool_size=(2*initStride, 2*initStride))(xin)
    x1_merged = merge([x1, x1_ident],mode='concat', concat_axis=1)

    x2_1=conv2dcustom(filters=2*C,x_input=x1_merged,pool=True,w2reg=w2reg,activation='leaky',padding=padding)    
    x2_1=conv2dcustom(filters=2*C,x_input=x2_1,w2reg=w2reg,activation='leaky',padding=padding)    
    x2_ident = AveragePooling2D()(x1_ident)
    x2_merged = merge([x2_1,x2_ident],mode='concat', concat_axis=1)

    #by branching we reduce the #params
    x3_1 = conv2dcustom(filters=4*C,x_input=x2_merged,pool=True,w2reg=w2reg,activation='leaky',padding=padding)    
    x3_1 = conv2dcustom(filters=4*C,x_input=x3_1,w2reg=w2reg,activation='leaky',padding=padding)    
    x3_ident = AveragePooling2D()(x2_ident)
    x3_merged = merge([x3_1,x3_ident],mode='concat', concat_axis=1)

    x4_1 = conv2dcustom(filters=8*C,x_input=x3_merged,pool=True,w2reg=w2reg,activation='leaky',padding=padding)    
    x4_1 = conv2dcustom(filters=8*C,x_input=x4_1,w2reg=w2reg,activation='leaky',padding=padding)    
    x4_ident = AveragePooling2D()(x3_ident)
    x4_merged = merge([x4_1,x4_ident],mode='concat', concat_axis=1)    

    x5_1 = conv2dcustom(filters=16*C,x_input=x4_merged,pool=False,w2reg=w2reg,activation='leaky',padding="same")    
    x5_1 = conv2dcustom(filters=16*C,x_input=x5_1,w2reg=w2reg,activation='leaky',padding="valid")    
    
    # last layer of encoding    
    flatten_x=Flatten() (x5_1)
    
    flatten_x =Dropout(dropout_rate)(flatten_x)
    
    output=Dense(numOfOutputs,activation="sigmoid",W_regularizer=w2reg)(flatten_x)
    
    model = Model(inputs=xin, outputs=output)

    
    if optimizer=='RMSprop':
        optimizer = RMSprop(lr)
    elif optimizer=='Adam':       
        optimizer = Adam(lr)
    elif optimizer=='Nadam':       
        optimizer = Nadam(lr,clipvalue=1.0)        
    model.compile(loss=loss, optimizer=optimizer)

    return model


def dense_branch(xstart, name, outsize=1,activation='sigmoid'):
    xdense_ = Dense(32,W_regularizer=l2(1e-4))(xstart)
    #xdense_ = BatchNormalization()(xdense_)
	# xdense_ = GaussianDropout(0)(xdense_)
    xdense_ = LeakyReLU(.1)(xdense_)
    xout = Dense(outsize,activation=activation, name=name,W_regularizer=l2(1e-4))(xdense_)
    return xout



def model_classification_skip3(params):
    
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
    if w2reg==True:
        w2reg=l2(1e-4)
    initStride=params['initStride']
    padding=params["padding"]
    optimizer=params["optimizer"]
    activation=params["activation"]
    
    xin = Input((z,h, w))
    x1=conv2dcustom(filters=C,x_input=xin,strides=initStride,w2reg=w2reg,activation=activation,data_format=data_format,padding=padding,pool=True)    
    x1=conv2dcustom(filters=C,x_input=x1,w2reg=w2reg,activation=activation,padding=padding)    
    x1_ident = AveragePooling2D(pool_size=(2*initStride, 2*initStride))(xin)
    x1_merged = merge([x1, x1_ident],mode='concat', concat_axis=1)

    x2_1=conv2dcustom(filters=3*C,x_input=x1_merged,pool=True,w2reg=w2reg,activation=activation,padding=padding)    
    x2_1=conv2dcustom(filters=3*C,x_input=x2_1,w2reg=w2reg,activation=activation,padding=padding)    
    x2_ident = AveragePooling2D()(x1_ident)
    x2_merged = merge([x2_1,x2_ident],mode='concat', concat_axis=1)

    #by branching we reduce the #params
    x3_1 = conv2dcustom(filters=8*C,x_input=x2_merged,pool=True,w2reg=w2reg,activation=activation,padding=padding)    
    x3_1 = conv2dcustom(filters=8*C,x_input=x3_1,w2reg=w2reg,activation=activation,padding=padding)    
    x3_ident = AveragePooling2D()(x2_ident)
    x3_merged = merge([x3_1,x3_ident],mode='concat', concat_axis=1)

    x4_1 = conv2dcustom(filters=9*C,x_input=x3_merged,pool=True,w2reg=w2reg,activation=activation,padding=padding)    
    x4_1 = conv2dcustom(filters=9*C,x_input=x4_1,w2reg=w2reg,activation=activation,padding=padding)    
    x4_ident = AveragePooling2D()(x3_ident)
    x4_merged = merge([x4_1,x4_ident],mode='concat', concat_axis=1)    

    x5_1 = conv2dcustom(filters=9*C,x_input=x4_merged,pool=False,w2reg=w2reg,activation=activation,padding="same")    
    x5_1 = conv2dcustom(filters=9*C,x_input=x5_1,w2reg=w2reg,activation=activation,padding="valid")    
    
    # last layer of encoding    
    flatten_x=Flatten() (x5_1)
    
    flatten_x =Dropout(dropout_rate)(flatten_x)
    
    
    output= dense_branch(flatten_x,name='classiflyOut',outsize=numOfOutputs,activation='sigmoid')
    
    
    model = Model(inputs=xin, outputs=output)

    
    if optimizer=='RMSprop':
        optimizer = RMSprop(lr)
    elif optimizer=='Adam':       
        optimizer = Adam(lr)
    elif optimizer=='Nadam':       
        optimizer = Nadam(lr,clipvalue=1.0)        
    model.compile(loss=loss, optimizer=optimizer)

    return model


