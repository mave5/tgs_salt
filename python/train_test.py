# train test
import configs
from utils import utils_train_test
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm_notebook
from keras.preprocessing import image
from sklearn.model_selection import ShuffleSplit
from utils import models
import os

#from image import ImageDataGenerator

#%%



# =============================================================================
# load data    
# =============================================================================
X,Y,ids=utils_train_test.load_data(configs,"train")
utils_train_test.array_stats(X)
utils_train_test.array_stats(Y)


utils_train_test.disp_imgs_masks(X,Y)


# =============================================================================
# model definition
# =============================================================================
model = models.model_skip(configs.trainingParams)
model.summary()


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
    
    utils_train_test.array_stats(X_train)
    utils_train_test.array_stats(Y_train)
    utils_train_test.array_stats(X_test)
    utils_train_test.array_stats(Y_test)
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
    model=models.model_skip(trainingParams)
    #model.summary()    
    
    data=X_train,Y_train,X_test,Y_test
    utils_train_test.train_test_model(data,trainingParams,model)
    
    # loading best weights from training session
    if  os.path.exists(path2weights):
        model.load_weights(path2weights)
        print ('weights loaded!')
    else:
        raise IOError('weights does not exist!!!')
    
    score_test=model.evaluate(utils_train_test.preprocess(X_test,configs.normalization_type),Y_test,verbose=0,batch_size=8)
    print ('score_test: %.5f' %(score_test))    
    Y_pred=model.predict(utils_train_test.preprocess(X_test,configs.normalization_type))>=0.5
    dicePerFold,_=utils_train_test.calc_dice(Y_test,Y_pred)
    print('average dice: %.2f' %dicePerFold)
    print ('-' *30)
    # store scores for all folds
    scores_nfolds.append(score_test)

print ('average score for %s folds is %s' %(nFolds,np.mean(scores_nfolds)))