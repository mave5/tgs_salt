import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
import matplotlib.pylab as plt
import cv2
import time
from utils import models
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import datetime
from glob import glob
from keras.models import load_model
# elastic augmentation
#from scipy.ndimage.filters import gaussian_filter

def separateLargeMasks(X,Y,configs):
    if configs.nonZeroMasksOnly:
        sumY=np.sum(Y,axis=(1,2,3))
        largeMasksInds=np.where(sumY>configs.largeMaskThreshold)
        #nzMaskIndices=np.where(np.any(Y,axis=(1,2,3)))[0]
        X=X[largeMasksInds]
        Y=Y[largeMasksInds]
        array_stats(X) 
        array_stats(Y)
        disp_imgs_masks(X,Y)
    return X,Y        

def histogramEqualization(X_train=None,X_test=None,histeq=False):
    # histogram equalization
    if histeq is True:
        print('histogram equlization ...')
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if X_train is not None:
            for k in range(X_train.shape[0]):
                X_train[k,:] = clahe.apply(X_train[k,0])
        if X_test is not None:                
            for k in range(X_test.shape[0]):
                X_test[k,:] = clahe.apply(X_test[k,0])        
    
    return X_train,X_test    
        

def rotateData(X,rotationAngle=180):
    n,c,h,w=X.shape
    Xr=np.zeros_like(X)
    for k in range(n):
        for k2 in range(c):
            Xr[k,k2]=np.rot90(X[k,k2],rotationAngle/90)
    return Xr

def generate_smooth_transformation_fields(im_shape, elastic_args):
    # typical values: nr_transformations=1000, alpha=2, sigma=.1
    nr_transformations=elastic_args["nr_of_random_transformations"]
    alpha=elastic_args["alpha"]
    sigma=elastic_args["sigma"]
    
    alpha *= (np.prod(im_shape) ** 0.5)
    sigma *= (np.prod(im_shape) ** 0.5)
    smooth_transformation_fields = np.zeros((nr_transformations, im_shape[0], im_shape[1]), dtype=np.float32)
    for t in range(nr_transformations):
        rand = np.random.rand(*im_shape) * 2 - 1
        smooth_transformation_fields[t] = cv2.GaussianBlur(rand, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT) * alpha
    return smooth_transformation_fields

# smooth_transformation_fields is a numpy array of N precalculated fields of dimensions N*h*w
def elastic_transform_multi_fast(x, y, kwargs, smooth_transformation_fields):
    N = smooth_transformation_fields.shape[0]
    x_t = x.copy()
    y_t = np.zeros_like(x) if y is None else y.copy()

    x_grid, y_grid = np.meshgrid(np.arange(x.shape[3]), np.arange(x.shape[2]))

    for k in range(x.shape[0]):
        if np.random.random() < kwargs['elastic_probability']:
            #Generate X and Y coordinates from random transformation + grid
            x_coords = (smooth_transformation_fields[np.random.randint(0, N)] + x_grid).astype('float32')
            y_coords = (smooth_transformation_fields[np.random.randint(0, N)] + y_grid).astype('float32')

            x_t[k, :] = cv2.remap(x[k, 0], x_coords, y_coords, cv2.INTER_CUBIC)
            for k2 in range(y.shape[1]):
                y_t[k, k2, :] = cv2.remap(y[k, k2, :], x_coords, y_coords, cv2.INTER_CUBIC)
    return x_t.astype('uint8'), y_t.astype('uint8')


def elastic_transform_multi_fast_float(x, y, kwargs, smooth_transformation_fields):
    N = smooth_transformation_fields.shape[0]
    x_t = np.zeros_like(x)
    if y is None:
        y= np.zeros_like(x)
    y_t = np.zeros_like(y)

    x_grid, y_grid = np.meshgrid(np.arange(x.shape[3]), np.arange(x.shape[2]))

    for k in range(x.shape[0]):
        if np.random.random() < kwargs['elastic_probability']:
            #Generate X and Y coordinates from random transformation + grid
            x_coords = (smooth_transformation_fields[np.random.randint(0, N)] + x_grid).astype('float32')
            y_coords = (smooth_transformation_fields[np.random.randint(0, N)] + y_grid).astype('float32')
            
            for k3 in range(x.shape[1]):
                x_t[k, k3] = cv2.remap(x[k, k3], x_coords, y_coords, cv2.INTER_CUBIC)
            for k2 in range(y.shape[1]):
                y_t[k, k2, :] = cv2.remap(y[k, k2, :], x_coords, y_coords, cv2.INTER_CUBIC)
    return x_t, y_t


def gammaAugmentImage(image,gamma=0.1):
    if np.random.random()<0.5:
        g_rand=(2 * np.random.rand() - 1) # random number between -1 to +1
        g = g_rand * gamma + 1.
    else:
        g= 1
    image=(255 * (image / 255.) ** g).astype("uint8")
    return image    

def gammaAugmentBatch(X,gamma=0.1):
    n,c,h,w=X.shape
    for i in range(n):
        X[i]=gammaAugmentImage(X[i],gamma)
    return X


def randomCroppingImage(image,alfaMax=0.06):
    C,H,W=image.shape
    if np.random.rand()<0.5:
        alfaH,alfaW=np.random.uniform(0.0,alfaMax),np.random.uniform(0.0,alfaMax)             
        dH,dW=int(alfaH*H),int(alfaW*W)            
        h0=np.random.randint(0,H-dH)
        w0=np.random.randint(0,W-dW)
        randomPixels=np.random.uniform(0.0,np.max(image),size=(dH,dW)) # data type is float
        #randomPixels=np.random.randint(0,np.max(image),size=(dH,dW))
        image[:,h0:h0+dH,w0:w0+dW]=randomPixels # data type with the same as image
    return image        

def randomCroppingBatch(X,alfaMax=0.1):
    # alfaMax: size of erasing: percentage of H and W
    N,C,H,W=X.shape
    for k in range(N):
        X[k]=randomCroppingImage(X[k],alfaMax)
    return X    


def train_test_evalMetric_preprocess(data,params_train,model):
    X_train,Y_train,X_test,Y_test=data
    foldnm=params_train['foldnm']  
    pre_train=params_train['pre_train'] 
    batch_size=params_train['batch_size'] 
    augmentation=params_train['augmentation'] 
    weightfolder=params_train['weightfolder'] 
    path2weights=params_train['path2weights'] 
    normalizationParams=params_train["normalizationParams"]
    augmentationParams=params_train["augmentationParams"]
    nbepoch=params_train["nbepoch"]
    path2model=os.path.join(weightfolder,"model.hdf5")    
    elastic_arg=params_train["elastic_arg"]
    
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
    if params_train['evalMetric']=='evalMetric': 
        best_score = 0
        previous_score = 0
    else:
        best_score = 1e6
        previous_score = 1e6
    patience = 0
    
    # elstic transformation
    smooth_transformation_fields = generate_smooth_transformation_fields(X_train.shape[2:],elastic_arg)
    
    for epoch in range(params_train['nbepoch']):
        
        # perform random cropping/erasing
        #if params_train["randomCropping"] is True:
            #X_train_t=randomCroppingBatch(X_train)

        # perform elastic transformation
        if params_train["elasticTransform"] is True:
            X_train_t,Y_train_t=elastic_transform_multi_fast(X_train,Y_train,elastic_arg,smooth_transformation_fields)
        else:
            X_train_t,Y_train_t=X_train.copy(),Y_train.copy()

    
        print ('epoch: %s / %s,  Current Learning Rate: %.1e' %(epoch,nbepoch,model.optimizer.lr.get_value()))
        #seed = np.random.randint(0, 999999)
    
        if augmentation:
            # augmentation data generator
            train_generator,steps_per_epoch=data_generator(X_train_t,Y_train_t,batch_size,augmentationParams)
            
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

        # evaluation metric        
        Y_pred=model.predict(preprocess(X_test,normalizationParams))>=0.5
        evalMetricPerFold,_=computeEvalMetric(Y_test,Y_pred)
        print('eval metric: %.3f' %evalMetricPerFold)
        score_test=evalMetricPerFold

        # check for improvement    
        if (score_test>=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)  
            model.save(path2model)
            
        # learning rate schedule
        if score_test<previous_score:
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
    #plt.show()
    
    print ('training completed!')
    train_status='completed!'
    return train_status    


def gammaAugment(image,gamma=0.1):
    if np.random.random()<0.5:
        g_rand=(2 * np.random.rand() - 1) # random number between -1 to +1
        g = g_rand * gamma + 1.
    else:
        g= 1
    image=(255 * (image / 255.) ** g).astype("float32")
    return image    


def unpadArray(Y,padSize=(13,14)):
    before,after=padSize  
    if (before==0 and after==0):
        return Y
    Y=Y[:,:,before:-after,before:-after]
    return Y

def padArrays(X,Y,padSize=(13,14)):
    if padSize==(0,0):
        return X,Y
    X=np.pad(X,((0,0),(0,0),padSize,padSize),"constant")
    if Y is not None:
        Y=np.pad(Y,((0,0),(0,0),padSize,padSize),"constant")
    return X,Y    

def computeEvalMetricPositive(Y_gt,Y_pred):
    nzMaskIndices=np.where(np.any(Y_gt,axis=(1,2,3)))[0]
    Y_gt=Y_gt[nzMaskIndices]
    Y_pred=Y_pred[nzMaskIndices]
    avgMetric,_=computeEvalMetric(Y_gt,Y_pred)
    return avgMetric
    
def re_arange(ids,ids_pred,Y_pred):
    Y_pred_rearanged=np.zeros_like(Y_pred)
    for k0,id0 in enumerate(ids):
        id_ind=ids_pred.index(id0)
        Y_pred_rearanged[k0]=Y_pred[id_ind]
    return Y_pred_rearanged

def getOutputEnsemble_evalMetric(path2allExperiments,experiments,data_type="train",path2data="../data/"):
    path2pickle=os.path.join(path2data,data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        X=data["X"]
        Y=data["Y"]
        ids=data["ids"]
    
    Y_predAllExperiments=np.zeros_like(Y,"float32")
    for experiment in experiments:
        print("experiment:%s" %experiment)
        path2experiment=os.path.join(path2allExperiments,experiment)
        path2predictions=os.path.join(path2experiment,"predictions")
    
        # load predictions
        path2pickle=glob(path2predictions+"/Y_pred_"+data_type+"*.p")[0]
        data_pred = pickle.load( open( path2pickle, "rb" ) )
        Y_pred=data_pred["Y"]
        ids_pred=data_pred["ids"]
        if ids!=ids_pred:
            Y_pred=re_arange(ids,ids_pred,Y_pred)
                
        array_stats(Y_pred)
        disp_imgs_masks(X,Y_pred>=0.5)
        Y_predAllExperiments+=Y_pred/len(experiments)        
        
        if Y is not None:
            avgEvalMetricAll,_=computeEvalMetric(Y,Y_pred>=0.5)
            print("average eval metric for all samples: %.3f" %avgEvalMetricAll)
            avgEvalMetricPositive=computeEvalMetricPositive(Y,Y_pred>=0.5)
            print("average eval metric for positive samples: %.3f" %avgEvalMetricPositive)
        print("-"*50)
    
    # convert to array
    #Y_predAllExperiments=np.hstack(Y_predAllExperiments)
    print ('ensemble shape:', Y_predAllExperiments.shape)
    Y_pred_ensemble=np.mean(Y_predAllExperiments,axis=1)[:,np.newaxis] #>=0.5
    array_stats(Y_pred_ensemble)

    if Y is not None:
        avgEvalMetricAll,_=computeEvalMetric(Y,Y_pred_ensemble>=0.5)
        print("average eval metric for all samples: %.3f" %avgEvalMetricAll)
        avgEvalMetricPositive=computeEvalMetricPositive(Y,Y_pred_ensemble>=0.5)
        print("average eval metric for positive samples: %.3f" %avgEvalMetricPositive)
    
    return Y_pred_ensemble


def multiplyVectorByMatrix(y,Y,threshold=0.5):
    # Y is N*C*H*W
    # y is N*1
    y_bin=y>=threshold
    for k in range(len(Y)):
        Y[k]=y_bin[k]*Y[k]
    return Y

def getOutputEnsembleClassify_evalMetric(path2allExperiments,experiments,data_type="train",path2data="../data/"):
    path2pickle=os.path.join(path2data,data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        X=data["X"]
        Y=data["Y"]
        ids=data["ids"]
    
    Y_predAllExperiments=np.zeros_like(X,"float32")
    y_predAllExperiments=np.zeros((X.shape[0],1),"float32")
    for experiment in experiments:
        print("experiment:%s" %experiment)
        path2experiment=os.path.join(path2allExperiments,experiment)
        path2predictions=os.path.join(path2experiment,"predictions")
    
        # load predictions
        path2pickle=glob(path2predictions+"/Y_pred_"+data_type+"*.p")[0]
        data_pred = pickle.load( open( path2pickle, "rb" ) )
        Y_pred=data_pred["Y"]
        y_pred=data_pred["y"]
        ids_pred=data_pred["ids"]
        
        if ids!=ids_pred:
            Y_pred=re_arange(ids,ids_pred,Y_pred)
            y_pred=re_arange(ids,ids_pred,y_pred)
                
        array_stats(Y_pred)
        array_stats(y_pred)
        disp_imgs_masks(X,Y_pred>=0.5)
        Y_predAllExperiments+=Y_pred/len(experiments)        
        y_predAllExperiments+=y_pred/len(experiments)                
        
        if Y is not None:
            avgEvalMetricAll,_=computeEvalMetric(Y,Y_pred>=0.5)
            print("average eval metric for all samples: %.3f" %avgEvalMetricAll)
            avgEvalMetricPositive=computeEvalMetricPositive(Y,Y_pred>=0.5)
            print("average eval metric for positive samples: %.3f" %avgEvalMetricPositive)
            # after classification
            Y_pred=multiplyVectorByMatrix(y_pred,Y_pred)
            avgEvalMetricAllAfter,_=computeEvalMetric(Y,Y_pred>=0.5)
            print("average eval metric for all samples after classification: %.3f" %avgEvalMetricAllAfter)
            
        print("-"*50)
    
    # convert to array
    print ('Y ensemble shape:', Y_predAllExperiments.shape)
    print ('y ensemble shape:', y_predAllExperiments.shape)
    Y_pred_ensemble=np.mean(Y_predAllExperiments,axis=1)[:,np.newaxis] 
    y_pred_ensemble=np.mean(y_predAllExperiments,axis=1)[:,np.newaxis]

    array_stats(Y_pred_ensemble)
    array_stats(y_pred_ensemble)

    if Y is not None:
        avgEvalMetricAll,_=computeEvalMetric(Y,Y_pred_ensemble>=0.5)
        print("average eval metric for all samples: %.3f" %avgEvalMetricAll)
        avgEvalMetricPositive=computeEvalMetricPositive(Y,Y_pred_ensemble>=0.5)
        print("average eval metric for positive samples: %.3f" %avgEvalMetricPositive)
        
        # after classification
        Y_pred_ensemble=multiplyVectorByMatrix(y_pred_ensemble,Y_pred_ensemble)
        avgEvalMetricAllAfter,_=computeEvalMetric(Y,Y_pred_ensemble>=0.5)
        print("average eval metric for all samples after classification: %.3f" %avgEvalMetricAllAfter)
    
    return Y_pred_ensemble


def getOutputEnsemble(path2allExperiments,experiments,data_type="train"):
    Y_predAllExperiments=[]
    for experiment in experiments:
        print("experiment:%s" %experiment)
        path2experiment=os.path.join(path2allExperiments,experiment)
        path2predictions=os.path.join(path2experiment,"predictions")
    
        # load predictions
        path2pickle=glob(path2predictions+"/Y_pred_"+data_type+"*.p")[0]
        data = pickle.load( open( path2pickle, "rb" ) )
        Y_pred=data["Y"]
        array_stats(Y_pred)
        disp_imgs_masks(Y_pred,Y_pred>=0.5)
        Y_predAllExperiments.append(Y_pred)        
        print("-"*50)
    
    # convert to array
    Y_predAllExperiments=np.hstack(Y_predAllExperiments)
    print ('ensemble shape:', Y_predAllExperiments.shape)
    Y_pred_ensemble=np.mean(Y_predAllExperiments,axis=1)[:,np.newaxis] #>=0.5
    array_stats(Y_pred_ensemble)
    
    return Y_pred_ensemble


def storePredictionsEnsemble(path2predictions,Y_pred,suffix=""):
    path2pickle=os.path.join(path2predictions,"Y_pred_"+suffix+".p")    
    data = { "Y": Y_pred }
    pickle.dump( data, open( path2pickle, "wb" ) )
    print("predictions stored!")
    return

def storePredictions(configs,Y_pred,suffix=""):
    path2pickle=os.path.join(configs.path2predictions,"Y_pred_"+suffix+"_"+configs.experiment+".p")    
    data = { "Y": Y_pred }
    pickle.dump( data, open( path2pickle, "wb" ) )
    print("predictions stored!")
    return

def storePredictionsWithIds(configs,Y_pred,ids,suffix=""):
    path2pickle=os.path.join(configs.path2predictions,"Y_pred_"+suffix+"_"+configs.experiment+".p")    
    data = { "Y": Y_pred,"ids":ids }
    pickle.dump( data, open( path2pickle, "wb" ) )
    print("predictions stored!")
    return

def storePredictionsWithIds_classify(configs,Y_pred,y_pred,ids,suffix=""):
    path2pickle=os.path.join(configs.path2predictions,"Y_pred_"+suffix+"_"+configs.experiment+".p")    
    data = { "Y": Y_pred,"ids":ids, "y": y_pred }
    pickle.dump( data, open( path2pickle, "wb" ) )
    print("predictions stored!")
    return


def updateRecoredInConfigs(path2Configs,recordName,recordValue,overwrite=False):
    configsDF=pd.read_csv(path2Configs) # load csv
    NameColumn=configsDF['Name'].as_matrix() # get Name column
    
    if recordName in NameColumn: # check if recored exists
        index=configsDF[configsDF['Name']==recordName].index[0] # get the index of the record
        recordValueLoaded=configsDF.get_value(index,"Value")   # get the record value  
        if (pd.isnull(recordValueLoaded)) or (overwrite is True): # check if recored value is None
            configsDF.set_value(index,'Value',recordValue) # update record value
            configsDF.to_csv(path2Configs,index=False) # store to csv
            print("record value updated!")
        else:
            print("record has values!")
            print(recordValue)
    else:
        print('record does not exist in data frame!')
        row=[len(configsDF),recordName,recordValue]
        configsDF.loc[len(configsDF)]=row
        configsDF.to_csv(path2Configs,index=False) # store to csv
    return configsDF            


def createSubmission(rlcDict,configs):
    submissionDF = pd.DataFrame.from_dict(rlcDict,orient='index')
    submissionDF.index.names = ['id']
    submissionDF.columns = ['rle_mask']    

    # Create submission DataFrame
    now = datetime.datetime.now()
    try:
        info=configs.seg_model_version+"_"+configs.experiment
    except:
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
    

def createSubmissionEnsemble(rlcDict,path2experiment,info="ensemble"):
    submissionDF = pd.DataFrame.from_dict(rlcDict,orient='index')
    submissionDF.index.names = ['id']
    submissionDF.columns = ['rle_mask']    

    # Create submission DataFrame
    now = datetime.datetime.now()
    #info="ensemble"
    
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    submissionFolder=os.path.join(path2experiment,"submissions")
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


def getOutputAllFolds_classify_prob(X,configs):
    nFolds=configs.nFolds
    y_predAllFolds=[]
    for foldnm in range(1,nFolds+1):
        print('fold: %s' %foldnm)
    
        y_pred=getYperFold(X,configs,foldnm)    
        array_stats(y_pred)
        disp_imgs_masks_labels(X,y_pred>=configs.binaryThreshold)
        y_predAllFolds.append(y_pred)        
        print('-'*50)
        
    # convert to array
    y_predAllFolds=np.hstack(y_predAllFolds)
    print ('ensemble shape:', y_predAllFolds.shape)
    y_leaderboard=np.mean(y_predAllFolds,axis=1)[:,np.newaxis]>=configs.binaryThreshold
    array_stats(y_leaderboard)

    # remember that we concatenated mask probability
    # to the second channel of X_leaderboard.
    Y_pred=(X[:,1]>=configs.maskThreshold).astype("uint8")[:,np.newaxis]
    for k in range(len(Y_pred)):
        if y_leaderboard[k,0]==False:
            Y_pred[k,0]=np.zeros_like(Y_pred[k,0],"uint8")
    
    return Y_pred        


def getOutputAllFolds_classify_prob_ySoft(X,configs):
    nFolds=configs.nFolds
    y_predAllFolds=np.zeros((X.shape[0],1))
    for foldnm in range(1,nFolds+1):
        print('fold: %s' %foldnm)
    
        y_pred=getYperFold(X,configs,foldnm)    
        array_stats(y_pred)
        disp_imgs_masks_labels(X,y_pred>=configs.binaryThreshold)
        y_predAllFolds+=y_pred/nFolds
        print('-'*50)
        
    # convert to array
    #y_predAllFolds=np.hstack(y_predAllFolds)
    print ('ensemble shape:', y_predAllFolds.shape)
    y_soft=np.mean(y_predAllFolds,axis=1)[:,np.newaxis]
    y_bin=y_soft>=configs.binaryThreshold
    array_stats(y_soft)

    # remember that we concatenated mask probability
    # to the second channel of X_leaderboard.
    Y_pred_soft=X[:,1][:,np.newaxis]
    Y_pred_bin=(Y_pred_soft>=configs.maskThreshold).astype("uint8")
    Y_pred_bin=multiplyVectorByMatrix(y_bin,Y_pred_bin)
    
    return Y_pred_bin,Y_pred_soft,y_soft        


def getOutputAllFolds_classification(X,configs):
    nFolds=configs.nFolds
    y_predAllFolds=[]
    for foldnm in range(1,nFolds+1):
        print('fold: %s' %foldnm)
    
        y_pred=getYperFold(X,configs,foldnm)    
        array_stats(y_pred)
        disp_imgs_masks_labels(X,y_pred>=.5)
        y_predAllFolds.append(y_pred)        
        print('-'*50)
        
    # convert to array
    y_predAllFolds=np.hstack(y_predAllFolds)
    print ('ensemble shape:', y_predAllFolds.shape)
    y_leaderboard=np.mean(y_predAllFolds,axis=1)[:,np.newaxis]>=configs.binaryThreshold
    array_stats(y_leaderboard)

    # remember that we concatenated 255*Y_pred_leaderboard 
    # to the second channel of X_leaderboard.
    Y_pred=(X[:,1,:]/255).astype("uint8")[:,np.newaxis]
    for k in range(len(Y_pred)):
        if y_leaderboard[k,0]==False:
            Y_pred[k,0]=np.zeros_like(Y_pred[k,0],"uint8")
    
    return Y_pred        


def getOutputAllFolds(X,configs,binaryMask=False):
    nFolds=configs.nFolds
    Y_predAllFolds=np.zeros_like(X,"float32")
    for foldnm in range(1,nFolds+1):
        print('fold: %s' %foldnm)
    
        Y_pred=getYperFold(X,configs,foldnm)    
        array_stats(Y_pred)
        disp_imgs_masks(X,Y_pred>=configs.maskThreshold)
        Y_predAllFolds+=Y_pred/nFolds
        print('-'*50)
        
    # convert to array
    print ('ensemble shape:', Y_predAllFolds.shape)
    if binaryMask is True:
        Y_predAllFolds=Y_predAllFolds>=configs.maskThreshold    
    array_stats(Y_predAllFolds)
        
    return Y_predAllFolds        


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

def getOutputAllFolds_loadModel(X,configs,binaryMask=False):
    nFolds=configs.nFolds
    Y_predAllFolds=[]
    for foldnm in range(1,nFolds+1):
        print('fold: %s' %foldnm)
    
        Y_pred=getYperFold_loadModel(X,configs,foldnm)    
        array_stats(Y_pred)
        disp_imgs_masks(X,Y_pred>=configs.maskThreshold)
        Y_predAllFolds.append(Y_pred)        
        print('-'*50)
        
    # convert to array
    Y_predAllFolds=np.hstack(Y_predAllFolds)
    print ('ensemble shape:', Y_predAllFolds.shape)
    Y_leaderboard=np.mean(Y_predAllFolds,axis=1)[:,np.newaxis] #>=0.5
    if binaryMask is True:
        Y_leaderboard=Y_leaderboard>=configs.maskThreshold    
    array_stats(Y_leaderboard)
        
    return Y_leaderboard        

def getYperFold_loadModel(X,configs,foldnm):
    # load model
    modelFolder=os.path.join(configs.path2experiment,"fold"+str(foldnm))
    
    # path to weights
    path2model=os.path.join(modelFolder,"model.hdf5")
    if  os.path.exists(path2model):
        model=load_model(path2model)
        print ('%s loaded!' %path2model)
    else:
        raise IOError ('model does not exist!')

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
    

def createModel(configs,showModelSummary=False):
    model = getattr(models, configs.model_type)(configs.trainingParams)
    
    if showModelSummary:
        model.summary()
    return model


def balance_data(X,y):
    y1inds=np.where(y==1)[0]
    y0inds=np.where(y==0)[0]
    if len(y1inds)>len(y0inds):
        diffLen=len(y1inds)-len(y0inds)
        extrasNegSampleInds=np.random.choice(y0inds,diffLen)
        y=np.concatenate((y,y[extrasNegSampleInds]))
        X=np.concatenate((X,X[extrasNegSampleInds]))
    else:
        diffLen=len(y0inds)-len(y1inds)
        extrasNegSampleInds=np.random.choice(y1inds,diffLen)
        y=np.concatenate((y,y[extrasNegSampleInds]))
        X=np.concatenate((X,X[extrasNegSampleInds]))
        
    return X,y


def trainNfolds_classification(X,Y,configs,masks=None):
    nFolds=configs.nFolds
    skf = StratifiedShuffleSplit(n_splits=nFolds, test_size=configs.test_size, random_state=321)
    
    # loop over folds
    foldnm=0
    scores_nfolds=[]
    evalMetric_nfolds=[]
    
    print ('wait ...')
    for train_ind, test_ind in skf.split(X,Y):
        foldnm+=1    
    
        train_ind=list(np.sort(train_ind))
        test_ind=list(np.sort(test_ind))
        
        X_train,Y_train=X[train_ind],np.array(Y[train_ind],'uint8')
        if configs.balanceDataFlag is True:
            X_train,Y_train=balance_data(X_train,Y_train)
        X_test,Y_test=X[test_ind],np.array(Y[test_ind],'uint8')
        masks_test=np.array(masks[test_ind],'uint8') # ground truth masks
        
        array_stats(X_train)
        array_stats(Y_train)
        array_stats(X_test)
        array_stats(Y_test)
        print ('-'*30)
    
        weightfolder=os.path.join(configs.path2experiment,"fold"+str(foldnm))
        if  not os.path.exists(weightfolder):
            os.makedirs(weightfolder)
            print (weightfolder +' created')    
    
        # path to weights
        path2weights=os.path.join(weightfolder,"weights.hdf5")
        
        # train test on fold #
        trainingParams=configs.trainingParams
        trainingParams['foldnm']=foldnm
        trainingParams['learning_rate']=configs.initialLearningRate
        trainingParams['weightfolder']=weightfolder
        trainingParams['path2weights']=path2weights
        trainingParams['maskThreshold']=configs.maskThreshold
        trainingParams['binaryThreshold']=configs.binaryThreshold

        # create model        
        model=createModel(configs,configs.showModelSummary)
        
        data=X_train,Y_train,X_test,Y_test
        #train_test_classification(data,trainingParams,model)
        #train_test_classificationEvalMetric(data,trainingParams,model)
        train_test_classificationElastic(data,trainingParams,model)
        
        # loading best weights from training session
        if  os.path.exists(path2weights):
            model.load_weights(path2weights)
            print ('weights loaded!')
        else:
            raise IOError('weights does not exist!!!')
        
        score_test=model.evaluate(preprocess(X_test,configs.normalizationParams),Y_test,verbose=0,batch_size=8)
        print ('score_test: %.5f' %(score_test))    
        print ('-' *30)
        # store scores for all folds
        scores_nfolds.append(score_test)
        
        # compute evaluation metrics
        y_test_pred=model.predict(preprocess(X_test,configs.normalizationParams))
        Y_test_pred=multiplyVectorByMatrix(y_test_pred>=configs.binaryThreshold,X_test[:,1][:,np.newaxis])
        evalMetricPerFold,_=computeEvalMetric(masks_test,Y_test_pred>=configs.maskThreshold)
        evalMetric_nfolds.append(evalMetricPerFold)
        print("eval metric for fold %s: %.3f" %(foldnm,evalMetricPerFold))
        print("-"*50)    
        
    print ('average score for %s folds is %s' %(nFolds,np.mean(scores_nfolds)))
    print ('average eval metric for %s folds is %.3f' %(nFolds,np.mean(evalMetric_nfolds)))
    print("-"*50)
    return evalMetric_nfolds



def trainNfolds(X,Y,configs):
    nFolds=configs.nFolds
    skf = ShuffleSplit(n_splits=nFolds, test_size=configs.test_size, random_state=321)
    
    # loop over folds
    foldnm=0
    scores_nfolds=[]
    evalMatric_nfolds=[]
    dice_nfolds=[]
    maskThreshold=configs.maskThreshold
    
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
            print (weightfolder +' created')    
    
        # path to weights
        path2weights=os.path.join(weightfolder,"weights.hdf5")
        
        # train test on fold #
        trainingParams=configs.trainingParams
        trainingParams['foldnm']=foldnm
        trainingParams['learning_rate']=configs.initialLearningRate
        trainingParams['weightfolder']=weightfolder
        trainingParams['path2weights']=path2weights

        # create model        
        model=createModel(configs,configs.showModelSummary)
        
        data=X_train,Y_train,X_test,Y_test
        #train_test_model(data,trainingParams,model) # best model is saved based on loss value
        #train_test_evalMetric(data,trainingParams,model) # best model is stored based on eval metric
        train_test_evalMetric_preprocess(data,trainingParams,model) # data is preprocessed before augmentation
        
        # loading best weights from training session
        if  os.path.exists(path2weights):
            model.load_weights(path2weights)
            print ('weights loaded!')
        else:
            raise IOError('weights does not exist!!!')
        
        score_test=model.evaluate(preprocess(X_test,configs.normalizationParams),Y_test,verbose=0,batch_size=8)
        print ('score_test: %.5f' %(score_test))    
        Y_pred=model.predict(preprocess(X_test,configs.normalizationParams))>=maskThreshold
        dicePerFold,_=calc_dice(Y_test,Y_pred)
        evalMetricPerFold,_=computeEvalMetric(Y_test,Y_pred)
        print('average dice: %.3f' %dicePerFold)
        print('eval metric: %.3f' %evalMetricPerFold)
        print ('-' *30)
        # store scores for all folds
        scores_nfolds.append(score_test)
        dice_nfolds.append(dicePerFold)
        evalMatric_nfolds.append(evalMetricPerFold)
    
    print ('average score for %s folds is %s' %(nFolds,np.mean(scores_nfolds)))
    print ('average dice for %s folds is %s' %(nFolds,np.mean(dice_nfolds)))
    print ('average eval metric for %s folds is %s' %(nFolds,np.mean(evalMatric_nfolds)))
    print("-"*50)
    return evalMatric_nfolds


def trainNfoldsWithTestAugment(X,Y,configs):
    nFolds=configs.nFolds
    skf = ShuffleSplit(n_splits=nFolds, test_size=configs.test_size, random_state=321)
    
    # loop over folds
    foldnm=0
    scores_nfolds=[]
    evalMatric_nfolds=[]
    dice_nfolds=[]
    maskThreshold=configs.maskThreshold
    
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
            print (weightfolder +' created')    
    
        # path to weights
        path2weights=os.path.join(weightfolder,"weights.hdf5")
        
        # train test on fold #
        trainingParams=configs.trainingParams
        trainingParams['foldnm']=foldnm
        trainingParams['learning_rate']=configs.initialLearningRate
        trainingParams['weightfolder']=weightfolder
        trainingParams['path2weights']=path2weights

        # create model        
        model=createModel(configs,configs.showModelSummary)
        
        data=X_train,Y_train,X_test,Y_test
        #train_test_model(data,trainingParams,model) # best model is saved based on loss value
        #train_test_evalMetric(data,trainingParams,model) # best model is stored based on eval metric
        train_test_evalMetric_preprocess(data,trainingParams,model) # data is preprocessed before augmentation
        
        # loading best weights from training session
        if  os.path.exists(path2weights):
            model.load_weights(path2weights)
            print ('weights loaded!')
        else:
            raise IOError('weights does not exist!!!')
        
        score_test=model.evaluate(preprocess(X_test,configs.normalizationParams),Y_test,verbose=0,batch_size=8)
        print ('score_test: %.5f' %(score_test))    
        
        Y_pred=np.zeros_like(Y_test,dtype=np.float32)
        rotationAngles=[0,90,180,270]
        for rotA in rotationAngles:
            X_test_rot=rotateData(X_test,rotationAngle=rotA)
            Y_pred_rot=model.predict(preprocess(X_test_rot,configs.normalizationParams))#>=maskThreshold
            Y_pred+=rotateData(Y_pred_rot,rotationAngle=-rotA)/len(rotationAngles)
        Y_pred=(Y_pred>=maskThreshold)
        dicePerFold,_=calc_dice(Y_test,Y_pred)
        evalMetricPerFold,_=computeEvalMetric(Y_test,Y_pred)
        print('average dice: %.3f' %dicePerFold)
        print('eval metric: %.3f' %evalMetricPerFold)
        print ('-' *30)
        # store scores for all folds
        scores_nfolds.append(score_test)
        dice_nfolds.append(dicePerFold)
        evalMatric_nfolds.append(evalMetricPerFold)
    
    print ('average score for %s folds is %s' %(nFolds,np.mean(scores_nfolds)))
    print ('average dice for %s folds is %s' %(nFolds,np.mean(dice_nfolds)))
    print ('average eval metric for %s folds is %s' %(nFolds,np.mean(evalMatric_nfolds)))
    print("-"*50)
    return evalMatric_nfolds


def computeEvalMetric(Y1,Y2):
    # Y1 and Y2 shape N*C*H*W
    assert Y1.shape==Y2.shape
    N,C,H,W=Y1.shape
    evalMetric=np.zeros(shape=N)
    for i in range(N):
        y1=Y1[i,0]
        y2=Y2[i,0]
        evalMetric[i]=computeEvalMetricPerSample(y1,y2)
        
    return np.mean(evalMetric),evalMetric    
    
def computeEvalMetricPerSample(gt, prediction):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    iou = computeIoUperSample(gt, prediction)
    precisions = [compute_precision_at(iou, th) for th in thresholds]
    return np.mean(precisions)
    

def compute_precision_at(iou, threshold):
    smooth=1e-9
    tp = iou >= threshold
    fp= iou < threshold
    fn= iou < threshold
    return float(tp+smooth) / (tp + fp + fn+smooth)


def computeIoUperSample(x,y):
    intersectXY=np.sum((x&y==1))
    unionXY=np.sum(x)+np.sum(y)-intersectXY
    smooth=1e-9
    iou= float(intersectXY+smooth)/(unionXY+smooth)
    return iou


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
                if stdX>0.0:
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
    return X

def data_generator(X_train,Y_train,batch_size,augmentationParams):
    image_datagen = ImageDataGenerator(**augmentationParams)
    augmentationParams_mask=augmentationParams.copy()
    augmentationParams_mask["preprocessing_function"]=None
    mask_datagen = ImageDataGenerator(**augmentationParams_mask)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = np.random.randint(1e6)
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(Y_train, augment=True, seed=seed)
    
    image_generator = image_datagen.flow(X_train,batch_size=batch_size,seed=seed)
    mask_generator = mask_datagen.flow(Y_train,batch_size=batch_size,seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    steps_per_epoch=len(X_train)/batch_size    

    return train_generator,steps_per_epoch


def data_generator_classification(X_train,y_train,batch_size,augmentationParams):
    image_datagen = ImageDataGenerator(**augmentationParams)
    
    seed=np.random.randint(1e6)
    image_datagen.fit(X_train, augment=True, seed=seed)
    
    image_generator = image_datagen.flow(X_train,y_train,batch_size=batch_size,seed=seed)

    # combine generators into one which yields image and masks
    train_generator = image_generator
    steps_per_epoch=len(X_train)/batch_size    

    return train_generator,steps_per_epoch


# train test model
def train_test_classification(data,params_train,model):
    X_train,Y_train,X_test,Y_test=data
    foldnm=params_train['foldnm']  
    pre_train=params_train['pre_train'] 
    batch_size=params_train['batch_size'] 
    augmentation=params_train['augmentation'] 
    weightfolder=params_train['weightfolder'] 
    path2weights=params_train['path2weights'] 
    normalizationParams=params_train["normalizationParams"]
    augmentationParams=params_train["augmentationParams"]
    nbepoch=params_train["nbepoch"]
    
    path2model=os.path.join(weightfolder,"model.hdf5")    
    
    print("-"*50)
    print('batch_size: %s, Augmentation: %s' %(batch_size,augmentation))
    print ('fold %s training in progress ...' %foldnm)
    print("-"*50)
    
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
    
    
    for epoch in range(params_train['nbepoch']):
    
        print ('epoch: %s / %s,  Current Learning Rate: %.1e' %(epoch,nbepoch,model.optimizer.lr.get_value()))

        if augmentation:
            # augmentation data generator
            train_generator,steps_per_epoch=data_generator_classification(X_train,Y_train,batch_size,augmentationParams)
            
            hist=model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,epochs=1, verbose=0)
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
    #plt.show()
    
    print ('training completed!')
    train_status='completed!'
    return train_status    


def train_test_classificationElastic(data,params_train,model):
    X_train,Y_train,X_test,Y_test=data
    foldnm=params_train['foldnm']  
    pre_train=params_train['pre_train'] 
    batch_size=params_train['batch_size'] 
    augmentation=params_train['augmentation'] 
    weightfolder=params_train['weightfolder'] 
    path2weights=params_train['path2weights'] 
    normalizationParams=params_train["normalizationParams"]
    augmentationParams=params_train["augmentationParams"]
    nbepoch=params_train["nbepoch"]
    elastic_arg=params_train["elastic_arg"]
    
    path2model=os.path.join(weightfolder,"model.hdf5")    
    
    print("-"*50)
    print('batch_size: %s, Augmentation: %s' %(batch_size,augmentation))
    print ('fold %s training in progress ...' %foldnm)
    print("-"*50)
    
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
    
    
    
    # elstic transformation
    print("elastic transformation ... !")
    smooth_transformation_fields = generate_smooth_transformation_fields(X_train.shape[2:],elastic_arg)
    
    for epoch in range(params_train['nbepoch']):
    
        print ('epoch: %s / %s,  Current Learning Rate: %.1e' %(epoch,nbepoch,model.optimizer.lr.get_value()))
        # perform elastic transformation
        if params_train["elasticTransform"] is True:
            X_train_t,_=elastic_transform_multi_fast_float(X_train,None,elastic_arg,smooth_transformation_fields)
        else:
            X_train_t=X_train.copy()

        if augmentation:
            # augmentation data generator
            train_generator,steps_per_epoch=data_generator_classification(X_train_t,Y_train,batch_size,augmentationParams)
            
            hist=model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,epochs=1, verbose=0)
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
    #plt.show()
    
    print ('training completed!')
    train_status='completed!'
    return train_status    



def train_test_classificationEvalMetric(data,params_train,model):
    X_train,Y_train,X_test,Y_test=data
    foldnm=params_train['foldnm']  
    pre_train=params_train['pre_train'] 
    batch_size=params_train['batch_size'] 
    augmentation=params_train['augmentation'] 
    weightfolder=params_train['weightfolder'] 
    path2weights=params_train['path2weights'] 
    normalizationParams=params_train["normalizationParams"]
    augmentationParams=params_train["augmentationParams"]
    nbepoch=params_train["nbepoch"]
    binaryThreshold=params_train["binaryThreshold"]
    maskThreshold=params_train["maskThreshold"]
    
    path2model=os.path.join(weightfolder,"model.hdf5")    
    
    print("-"*50)
    print('batch_size: %s, Augmentation: %s' %(batch_size,augmentation))
    print ('fold %s training in progress ...' %foldnm)
    print("-"*50)
    
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
    
    
    for epoch in range(params_train['nbepoch']):
    
        print ('epoch: %s / %s,  Current Learning Rate: %.1e' %(epoch,nbepoch,model.optimizer.lr.get_value()))
    
        if augmentation:
            # augmentation data generator
            train_generator,steps_per_epoch=data_generator_classification(X_train,Y_train,batch_size,augmentationParams)
            hist=model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,epochs=1, verbose=0)
        else:
            hist=model.fit(preprocess(X_train,normalizationParams), Y_train, batch_size=batch_size,epochs=1, verbose=0)
            
        # evaluate on test and train data
        score_test=model.evaluate(preprocess(X_test,normalizationParams),Y_test,verbose=0)
        score_train=np.mean(hist.history['loss'])

        # compute evaluation metrics
        y_test_pred=model.predict(preprocess(X_test,normalizationParams))
        Y_test_pred=multiplyVectorByMatrix(y_test_pred>=binaryThreshold,X_test[:,1][:,np.newaxis])
        evalMetric,_=computeEvalMetric(Y_test,Y_test_pred>=maskThreshold)
        print("eval metric: %.3f" %evalMetric)
        
       
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
    #plt.show()
    
    print ('training completed!')
    train_status='completed!'
    return train_status    


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
    nbepoch=params_train["nbepoch"]
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
    
        print ('epoch: %s / %s,  Current Learning Rate: %.1e' %(epoch,nbepoch,model.optimizer.lr.get_value()))
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

        # evaluation metric        
        Y_pred=model.predict(preprocess(X_test,normalizationParams))>=0.5
        evalMetricPerFold,_=computeEvalMetric(Y_test,Y_pred)
        print('eval metric: %.3f' %evalMetricPerFold)
        

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
    #plt.show()
    
    print ('training completed!')
    train_status='completed!'
    return train_status    


def train_test_evalMetric(data,params_train,model):
    X_train,Y_train,X_test,Y_test=data
    foldnm=params_train['foldnm']  
    pre_train=params_train['pre_train'] 
    batch_size=params_train['batch_size'] 
    augmentation=params_train['augmentation'] 
    weightfolder=params_train['weightfolder'] 
    path2weights=params_train['path2weights'] 
    normalizationParams=params_train["normalizationParams"]
    augmentationParams=params_train["augmentationParams"]
    nbepoch=params_train["nbepoch"]
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
    if params_train['evalMetric']=='evalMetric': 
        best_score = 0
        previous_score = 0
    else:
        best_score = 1e6
        previous_score = 1e6
    patience = 0
    
    # augmentation data generator
    train_generator,steps_per_epoch=data_generator(X_train,Y_train,batch_size,augmentationParams)

    
    for epoch in range(params_train['nbepoch']):
    
        print ('epoch: %s / %s,  Current Learning Rate: %.1e' %(epoch,nbepoch,model.optimizer.lr.get_value()))
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

        # evaluation metric        
        Y_pred=model.predict(preprocess(X_test,normalizationParams))>=0.5
        evalMetricPerFold,_=computeEvalMetric(Y_test,Y_pred)
        print('eval metric: %.3f' %evalMetricPerFold)
        score_test=evalMetricPerFold

        # check for improvement    
        if (score_test>=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)  
            model.save(path2model)
            
        # learning rate schedule
        if score_test<previous_score:
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
    #plt.show()
    
    print ('training completed!')
    train_status='completed!'
    return train_status    

def overlay_contour(img,mask,color=(0,255,0)):
    mask=np.array(mask,"uint8")
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 3)
    return img   


def disp_imgs_2masks_labels(X,Y2,y,y_pred,r=2,c=3):
    assert len(X)==len(y)        
    n=r*c
    plt.figure(figsize=(12,8))
    indices=np.random.randint(len(X),size=n)
    for k,ind in enumerate(indices):
        img=X[ind,0]
        mask=X[ind,1]>=0.5
        mask2=Y2[ind,0]>=0.5
        img=overlay_contour(img,mask,(255,0,255))    
        img=overlay_contour(img,mask2,(0,255,0))# gt    
        h,w=img.shape
        text1=y_pred[ind]
        text2=y[ind]
        plt.subplot(r,c,k+1)
        plt.imshow(img);        
        plt.text(5,h-5,text1,fontsize=12)
        plt.text(50,h-5,text2,fontsize=12)
        plt.title(ind)    
    #plt.show()
    plt.draw()	

def disp_imgs_masks_labels(X,y,r=2,c=3):
    assert len(X)==len(y)        
    n=r*c
    plt.figure(figsize=(12,8))
    indices=np.random.randint(len(X),size=n)
    for k,ind in enumerate(indices):
        img=X[ind,0]
        mask=X[ind,1]>=0.5
        img=overlay_contour(img,mask)    
        h,w=img.shape
        label=y[ind]
        plt.subplot(r,c,k+1)
        plt.imshow(img,cmap="gray");        
        plt.text(5,h-5,label,fontsize=12)
        plt.title(ind)    
    #plt.show()
    plt.draw()	
 
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
    #plt.show()
    plt.draw()

def array_stats(X):
    X=np.asarray(X)
    print ('array shape: ',X.shape, X.dtype)
    #print 'min: %.3f, max:%.3f, avg: %.3f, std:%.3f' %(np.min(X),np.max(X),np.mean(X),np.std(X))
    print ('min: {}, max: {}, avg: {:.3}, std:{:.3}'.format( np.min(X),np.max(X),np.mean(X),np.std(X)))


from utils import utils_config
import os

def load_data_sudoAnnotations(configs,data_type="test"):
    path2allExperiments=configs.path2allExperiments
    agileIterationNum=configs.agileIterationNum
    projectStage=configs.projectStage
    maskThreshold=configs.maskThreshold
    binaryThreshold=configs.binaryThreshold
    
    print("pick an experiment to be used as sudo-annotations:")
    experimentSudo=utils_config.getAnExperiment(path2allExperiments,agileIterationNum,projectStage)
    path2experimentSudo=os.path.join(path2allExperiments,experimentSudo,"predictions")
    path2pickle=glob(path2experimentSudo+"/Y_pred_"+data_type+"*.p")[0]
    data = pickle.load( open( path2pickle, "rb" ) )
    Y_pred=data["Y"]>= maskThreshold
    y_pred=data["y"]
    ids=data["ids"]
    Y_pred=multiplyVectorByMatrix(y_pred,Y_pred,binaryThreshold)
    
    return Y_pred,ids

    

def load_data_classify_prob_bin(configs,data_type="train"):
    
    # loading images and masks
    path2pickle=os.path.join(configs.path2data,data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        X=data["X"]
        X=X.astype("float32")/255.
        Y=data["Y"]
        ids=data["ids"]
    else:
        raise IOError(path2pickle+" does not exist!")

    # loading predictions
    path2pickle=glob(configs.path2data+"Y_pred_"+data_type+"*.p")[0]
    seg_model_version=path2pickle.split("_")[3][:-2] # get seg model version
    configs.seg_model_version=seg_model_version # store model version
    #path2pickle=os.path.join(configs.path2data,"Y_pred_"+data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        Y_pred=data["Y"]
        Y_bin_pred=(Y_pred>=configs.binaryThreshold).astype("float32")
    else:
        raise IOError(path2pickle+" does not exist!")

    # concat images and predictions
    X=np.concatenate((X,Y_pred,Y_bin_pred),axis=1)
    
    # conver masks to labels
    y=np.any(Y,axis=(1,2,3))
    
    return X,y,ids

def load_data_classify_prob_largeMaskThreshold(configs,data_type="train"):
    
    # loading images and masks
    path2pickle=os.path.join(configs.path2data,data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        X=data["X"]
        X=X.astype("float32")/255.
        Y=data["Y"]
        ids=data["ids"]
    else:
        raise IOError(path2pickle+" does not exist!")

    # loading predictions
    path2pickle=glob(configs.path2data+"Y_pred_"+data_type+"*.p")[0]
    baseName=os.path.basename(path2pickle)
    lenOfPrefix=len("Y_pred_"+data_type)
    seg_model_version=baseName[lenOfPrefix+1:-2]# path2pickle.split("_")[3][:-2] # get seg model version
    configs.seg_model_version=seg_model_version # store model version
    #path2pickle=os.path.join(configs.path2data,"Y_pred_"+data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        Y_pred=data["Y"]
        #Y_pred=np.array(Y_pred*255,"uint8") # same range as images
    else:
        raise IOError(path2pickle+" does not exist!")

    # concat images and predictions
    X=np.concatenate((X,Y_pred),axis=1)
    
    # conver masks to labels
    sumY=np.sum(Y,axis=(1,2,3))
    y=(sumY>configs.largeMaskThreshold)
    
    
    return X,y,ids,Y


def load_data_classify_prob(configs,data_type="train"):
    
    # loading images and masks
    path2pickle=os.path.join(configs.path2data,data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        X=data["X"]
        X=X.astype("float32")/255.
        Y=data["Y"]
        ids=data["ids"]
    else:
        raise IOError(path2pickle+" does not exist!")

    # loading predictions
    path2pickle=glob(configs.path2data+"Y_pred_"+data_type+"*.p")[0]
    baseName=os.path.basename(path2pickle)
    lenOfPrefix=len("Y_pred_"+data_type)
    seg_model_version=baseName[lenOfPrefix+1:-2]# path2pickle.split("_")[3][:-2] # get seg model version
    configs.seg_model_version=seg_model_version # store model version
    #path2pickle=os.path.join(configs.path2data,"Y_pred_"+data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        Y_pred=data["Y"]
        #Y_pred=np.array(Y_pred*255,"uint8") # same range as images
    else:
        raise IOError(path2pickle+" does not exist!")

    # concat images and predictions
    X=np.concatenate((X,Y_pred),axis=1)
    
    # conver masks to labels
    y=np.any(Y,axis=(1,2,3))
    
    return X,y,ids,Y
    

def load_data_classification(configs,data_type="train"):
    
    # loading images and masks
    path2pickle=os.path.join(configs.path2data,data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        X=data["X"]
        Y=data["Y"]
        ids=data["ids"]
    else:
        raise IOError(path2pickle+" does not exist!")

    # loading predictions
    path2pickle=glob(configs.path2data+"Y_pred_"+data_type+"*.p")[0]
    seg_model_version=path2pickle.split("_")[3][:-2] # get seg model version
    configs.seg_model_version=seg_model_version # store model version
    #path2pickle=os.path.join(configs.path2data,"Y_pred_"+data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        Y_pred=data["Y"]
        Y_pred=np.array(Y_pred*255,"uint8") # same range as images
    else:
        raise IOError(path2pickle+" does not exist!")

    # concat images and predictions
    X=np.concatenate((X,Y_pred),axis=1)
    
    # conver masks to labels
    y=np.any(Y,axis=(1,2,3))
    
    return X,y,ids
            
def load_pickle(path2pickle):
    data = pickle.load( open( path2pickle, "rb" ) )
    X=data["X"]
    Y=data["Y"]
    ids=data["ids"]
    depths=data["depths"]
    return X,Y.astype("uint8"),ids,depths

def load_data_depth(configs,data_type="train"):
    path2pickle=os.path.join(configs.path2data,data_type+"_depths.p")
    if os.path.exists(path2pickle):
        return load_pickle(path2pickle)
    
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

    # load depth
    path2depth=os.path.join(configs.path2data,"depths.csv")
    depths = pd.read_csv(path2depth)

    data = { "X": X, "Y": Y, "ids": ids, "depths": depths }
    pickle.dump( data, open( path2pickle, "wb" ) )
    
    # load
    return load_pickle(path2pickle)

def load_data_ensemble(path2data,data_type="train"):

    path2pickle=os.path.join(path2data,data_type+".p")
    if os.path.exists(path2pickle):
        data = pickle.load( open( path2pickle, "rb" ) )
        X=data["X"]
        Y=data["Y"]
        ids=data["ids"]
        return X,Y.astype("uint8"),ids
    else:
        raise IOError("data does not exist!")

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

def data_resize(X,Y,h,w):

    n,c,h0,w0=X.shape
    if (h0==h and w0==w):
        return X,Y
        
    X_r=np.zeros((n,c,h,w),dtype=X.dtype)
    if Y is not None:
        Y_r=np.zeros((n,c,h,w),dtype=Y.dtype)
    else:
        Y_r=None
    for i in range(n):
        X_r[i,0]=cv2.resize(X[i,0],(w,h),interpolation = cv2.INTER_CUBIC)
        if Y is not None:
            Y_r[i,0]=cv2.resize(Y[i,0],(w,h),interpolation = cv2.INTER_CUBIC)
    return X_r,Y_r
    



