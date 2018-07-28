from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pickle
import os
import numpy as np
import matplotlib.pylab as plt
import cv2
import time
from utils import models
from sklearn.model_selection import ShuffleSplit
import pandas as pd
import datetime


def createSubmission(rlcDict,configs):
    submissionDF = pd.DataFrame.from_dict(rlcDict,orient='index')
    submissionDF.index.names = ['id']
    submissionDF.columns = ['rle_mask']    

    # Create submission DataFrame
    now = datetime.datetime.now()
    info=configs.experiment
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    submissionFolder=os.path.join(configs.path2experiment,"submissions")
    if not os.path.exists(submissionFolder):
        os.mkdir(submissionFolder)
        print(submissionFolder+ ' created!')
    path2submission = os.path.join(submissionFolder, 'submission_' + suffix + '.csv')
    print(path2submission)
    submissionDF.to_csv(path2submission)
    submissionDF.head()
    


def converMasksToRunLengthDict(Y_leaderboard,ids_leaderboard):
    print("wait to convert RLC ...")
    runLengthDict={}
    for id_ind,id_leader in enumerate(ids_leaderboard):
        #mask_pred=np.round(Y_leaderboard[id_ind,0]).astype("uint8")
        mask_pred=Y_leaderboard[id_ind,0].astype("uint8")
        runLengthMaskPred=runLengthEncoding(mask_pred)
        id_=id_leader[:-4] # remove .png
        runLengthDict[id_]=runLengthMaskPred 
    print("RLC completed!")
    return runLengthDict        


def getOutputAllFolds(X,configs):
    nFolds=configs.nFolds
    Y_predAllFolds=[]
    for foldnm in range(1,nFolds+1):
        print('fold: %s' %foldnm)
    
        Y_pred=getYperFold(X,configs,foldnm)    
        array_stats(Y_pred)
        disp_imgs_masks(X,Y_pred>=.5)
        Y_predAllFolds.append(Y_pred)        
        print('-'*50)
        
    # convert to array
    Y_predAllFolds=np.hstack(Y_predAllFolds)
    print ('ensemble shape:', Y_predAllFolds.shape)
    Y_leaderboard=np.mean(Y_predAllFolds,axis=1)[:,np.newaxis]>=0.5
    array_stats(Y_leaderboard)
        
    return Y_leaderboard        

def getYperFold(X,configs,foldnm):
    # load weights
    model=createModel(configs)    
    weightfolder=os.path.join(configs.path2experiment,"fold"+str(foldnm))
    
    # path to weights
    path2weights=os.path.join(weightfolder,"weights.hdf5")
    if  os.path.exists(path2weights):
        model.load_weights(path2weights)
        print ('%s loaded!' %path2weights)
    else:
        raise IOError ('weights does not exist!')

    # prediction
    Y_pred=model.predict(preprocess(X,configs.normalizationParams))
    return Y_pred


def run_length_decoding(mask_rle, shape):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return
    Returns:
        numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T

def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1] + rle[-2] == x.size:
        rle[-2] = rle[-2] - 1

    return rle

def runLengthEncoding(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
    

def createModel(configs):
    if configs.model_type=="skip":
        model = models.model_skip(configs.trainingParams)
    elif configs.model_type=="encoder_decoder":
        model = models.model_encoder_decoder(configs.trainingParams)
    else:
        raise IOError("%s not found!" %configs.model_type)
    model.summary()
    return model


def trainNfolds(X,Y,configs):
    nFolds=configs.nFolds
    skf = ShuffleSplit(n_splits=nFolds, test_size=configs.test_size, random_state=321)
    
    # loop over folds
    foldnm=0
    scores_nfolds=[]
    
    print ('wait ...')
    for train_ind, test_ind in skf.split(X,Y):
        foldnm+=1    
    
        train_ind=list(np.sort(train_ind))
        test_ind=list(np.sort(test_ind))
        
        X_train,Y_train=X[train_ind],np.array(Y[train_ind],'uint8')
        X_test,Y_test=X[test_ind],np.array(Y[test_ind],'uint8')
        
        array_stats(X_train)
        array_stats(Y_train)
        array_stats(X_test)
        array_stats(Y_test)
        print ('-'*30)
    
        weightfolder=os.path.join(configs.path2experiment,"fold"+str(foldnm))
        if  not os.path.exists(weightfolder):
            os.makedirs(weightfolder)
            print ('weights folder created')    
    
        # path to weights
        path2weights=os.path.join(weightfolder,"weights.hdf5")
        
        # train test on fold #
        trainingParams=configs.trainingParams
        trainingParams['foldnm']=foldnm
        trainingParams['learning_rate']=configs.initialLearningRate
        trainingParams['weightfolder']=weightfolder
        trainingParams['path2weights']=path2weights

        # create model        
        model=createModel(configs)
        
        data=X_train,Y_train,X_test,Y_test
        train_test_model(data,trainingParams,model)
        
        # loading best weights from training session
        if  os.path.exists(path2weights):
            model.load_weights(path2weights)
            print ('weights loaded!')
        else:
            raise IOError('weights does not exist!!!')
        
        score_test=model.evaluate(preprocess(X_test,configs.normalizationParams),Y_test,verbose=0,batch_size=8)
        print ('score_test: %.5f' %(score_test))    
        Y_pred=model.predict(preprocess(X_test,configs.normalizationParams))>=0.5
        dicePerFold,_=calc_dice(Y_test,Y_pred)
        print('average dice: %.2f' %dicePerFold)
        print ('-' *30)
        # store scores for all folds
        scores_nfolds.append(score_test)
    
    print ('average score for %s folds is %s' %(nFolds,np.mean(scores_nfolds)))
    return

# calcualte dice
def calc_dice(X,Y,d=0):
    N=X.shape[d]    
    # intialize dice vector
    dice=np.zeros([N,1])

    for k in range(N):
        x=X[k,0] >.5 # convert to logical
        y =Y[k,0]>.5 # convert to logical

        # number of ones for intersection and union
        intersectXY=np.sum((x&y==1))
        unionXY=np.sum(x)+np.sum(y)

        if unionXY!=0:
            dice[k]=2* intersectXY/(unionXY*1.0)
            #print 'dice is: %0.2f' %dice[k]
        else:
            dice[k]=1
            #print 'dice is: %0.2f' % dice[k]
        #print 'processing %d, dice= %0.2f' %(k,dice[k])
    return np.mean(dice),dice    

def preprocess(X,params):    
    # type of normalization
    norm_type=params['normalization_type']
    X=np.array(X,'float32')
    
    if norm_type =='zeroMeanUnitStd':
        for index in range(np.shape(X)[1]):
            meanX=np.mean(X[:,index,:,:])
            X[:,index] -= meanX
            stdX = np.std(X[:,index,:,:])
            if stdX!=0.0:
                X[:,index] /= stdX
    elif norm_type == 'zeroMeanUnitStdPerSample':
        for img_ind in range(np.shape(X)[0]):        
            for index in range(np.shape(X)[1]):
                meanX=np.mean(X[img_ind,index,:,:])
                stdX = np.std(X[img_ind,index,:,:])
                X[img_ind,index] -= meanX
                if stdX!=0.0:
                    X[img_ind,index] /= stdX
    elif norm_type == 'minus1_plus1': # [-1 to +1]
        X=np.array(X,'float32')
        X /= 255.
        X -= 0.5
        X *= 2.
    elif norm_type =='zeroMeanUnitStdGlobal':
        meanX=params['meanX'] # global mean
        stdX=params['stdX'] # global std
        for index in range(np.shape(X)[1]):
            X[:,index] -= meanX
            if stdX!=0.0:
                X[:,index] /= stdX
    elif norm_type == 'scale':
        X/=np.max(X)
    elif norm_type is None:
        return X
    else:
        raise IOError('no normalization was specified!')


def data_generator(X_train,Y_train,batch_size,augmentationParams):
    image_datagen = ImageDataGenerator(**augmentationParams)
    mask_datagen = ImageDataGenerator(**augmentationParams)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(Y_train, augment=True, seed=seed)
    
    image_generator = image_datagen.flow(X_train,batch_size=batch_size,seed=seed)
    mask_generator = mask_datagen.flow(Y_train,batch_size=batch_size,seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    steps_per_epoch=len(X_train)/batch_size    

    return train_generator,steps_per_epoch


# train test model
def train_test_model(data,params_train,model):
    X_train,Y_train,X_test,Y_test=data
    foldnm=params_train['foldnm']  
    pre_train=params_train['pre_train'] 
    batch_size=params_train['batch_size'] 
    augmentation=params_train['augmentation'] 
    weightfolder=params_train['weightfolder'] 
    path2weights=params_train['path2weights'] 
    normalizationParams=params_train["normalizationParams"]
    augmentationParams=params_train["augmentationParams"]
    
    path2model=os.path.join(weightfolder,"model.hdf5")    
    
    print('batch_size: %s, Augmentation: %s' %(batch_size,augmentation))
    print ('fold %s training in progress ...' %foldnm)
    
    # load last weights
    if pre_train== True:
        if  os.path.exists(path2weights):
            model.load_weights(path2weights)
            print ('previous weights loaded!')
        else:
            raise IOError('weights does not exist!!!')
    else:
        if  os.path.exists(path2weights):
            model.load_weights(path2weights)
            print (path2weights)
            print ('previous weights loaded!')
            train_status='previous weights'
            return train_status
    
    # path to csv file to save scores
    path2scorescsv = os.path.join(weightfolder,'scores.csv')
    first_row = 'train,test'
    with open(path2scorescsv, 'w+') as f:
        f.write(first_row + '\n')
           
    # initialize     
    start_time=time.time()
    scores_test=[]
    scores_train=[]
    if params_train['loss']=='dice': 
        best_score = 0
        previous_score = 0
    else:
        best_score = 1e6
        previous_score = 1e6
    patience = 0
    
    # augmentation data generator
    train_generator,steps_per_epoch=data_generator(X_train,Y_train,batch_size,augmentationParams)

    
    for epoch in range(params_train['nbepoch']):
    
        print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model.optimizer.lr.get_value()))
        #seed = np.random.randint(0, 999999)
    
        if augmentation:
            hist=model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,epochs=1, verbose=0)
            #X_batch,Y_batch=iterate_minibatches(X_train,y_train,X_train.shape[0],shuffle=False,augment=True)  
            #hist=model.fit(preprocess(X_batch,normalizationParams), Y_batch, batch_size=batch_size,epochs=1, verbose=0)
        else:
            hist=model.fit(preprocess(X_train,normalizationParams), Y_train, batch_size=batch_size,epochs=1, verbose=0)
            
        # evaluate on test and train data
        score_test=model.evaluate(preprocess(X_test,normalizationParams),Y_test,verbose=0)
        score_train=np.mean(hist.history['loss'])
       
        print ('score_train: %s, score_test: %s' %(score_train,score_test))
        scores_test=np.append(scores_test,score_test)
        scores_train=np.append(scores_train,score_train)    

        # check for improvement    
        if (score_test<=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)  
            model.save(path2model)
            
        # learning rate schedule
        if score_test>previous_score:
            #print "Incrementing Patience."
            patience += 1

        # learning rate schedule                
        if patience == params_train['max_patience']:
            params_train['learning_rate'] = params_train['learning_rate']/2
            print ("Upating Current Learning Rate to: ", params_train['learning_rate'])
            model.optimizer.lr.set_value(params_train['learning_rate'])
            print ("Loading the best weights again. best_score: ",best_score)
            model.load_weights(path2weights)
            patience = 0
        
        # save current test score
        previous_score = score_test    
        
        # store scores into csv file
        with open(path2scorescsv, 'a') as f:
            string = str([score_train,score_test])
            f.write(string + '\n')
           
    
    print ('model was trained!')
    elapsed_time=(time.time()-start_time)/60
    print ('elapsed time: %d  mins' %elapsed_time)      

    # train test progress plots
    plt.figure(figsize=(10,10))
    plt.plot(scores_test)
    plt.plot(scores_train)
    plt.title('train-validation progress',fontsize=20)
    plt.legend(('test','train'),fontsize=20)
    plt.xlabel('epochs',fontsize=20)
    plt.ylabel('loss',fontsize=20)
    plt.grid(True)
    plt.savefig(weightfolder+'/train_val_progress.png')
    plt.show()
    
    print ('training completed!')
    train_status='completed!'
    return train_status    


def overlay_contour(img,mask):
    mask=np.array(mask,"uint8")
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    return img   
 
def disp_imgs_masks(X,Y,r=2,c=3):
    assert np.ndim(X)==np.ndim(Y)        
    n=r*c
    plt.figure(figsize=(12,8))
    indices=np.random.randint(len(X),size=n)
    for k,ind in enumerate(indices):
        img=X[ind,0]
        mask=Y[ind,0]
        img=overlay_contour(img,mask)    
        plt.subplot(r,c,k+1)
        plt.imshow(img,cmap="gray");        
        plt.title(ind)    
    plt.show()
def array_stats(X):
    X=np.asarray(X)
    print ('array shape: ',X.shape, X.dtype)
    #print 'min: %.3f, max:%.3f, avg: %.3f, std:%.3f' %(np.min(X),np.max(X),np.mean(X),np.std(X))
    print ('min: {}, max: {}, avg: {:.3}, std:{:.3}'.format( np.min(X),np.max(X),np.mean(X),np.std(X)))
    

def load_data(configs,data_type="train"):
    path2pickle=os.path.join(configs.path2data,data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        X=data["X"]
        Y=data["Y"]
        ids=data["ids"]
        return X,Y.astype("uint8"),ids
    
    if data_type=="train":
        path2dataTT=configs.path2train
    elif data_type=="test":
        path2dataTT=configs.path2test
    else:
        raise IOError("data type unknown!")
    
    ids = next(os.walk(path2dataTT+"images"))[2]
    # Get and resize train images and masks
    X = np.zeros((len(ids), configs.img_channel,configs.img_height, configs.img_width ), dtype=np.uint8)
    Y = np.zeros((len(ids), configs.img_channel,configs.img_height, configs.img_width ), dtype=np.bool)
    print('loading images ...')

    for n, id_ in enumerate(ids):
        # loading image
        img = image.load_img(path2dataTT + '/images/' + id_)
        X[n] = image.img_to_array(img)[0]
        
        # loading mask
        if data_type=="train":
            mask=image.load_img(path2dataTT + '/masks/' + id_)
            Y[n] = image.img_to_array(mask)[0]

    data = { "X": X, "Y": Y, "ids": ids }
    pickle.dump( data, open( path2pickle, "wb" ) )
    
    # load
    data = pickle.load( open( path2pickle, "rb" ) )
    X=data["X"]
    Y=data["Y"]
    ids=data["ids"]
    
    return X,Y.astype("uint8"),ids