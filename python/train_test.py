# train test
import configs
from utils import utils_train_test
import numpy as np
#import matplotlib.pylab as plt
#import pandas as pd
#from tqdm import tqdm_notebook
#from keras.preprocessing import image
#from sklearn.model_selection import ShuffleSplit
#from utils import models

configs.showModelSummary=True

#%%

# =============================================================================
# load data    
# =============================================================================
X,Y,ids_train=utils_train_test.load_data(configs,"train")
X,Y=utils_train_test.padArrays(X,Y,configs.padSize)
utils_train_test.array_stats(X)
utils_train_test.array_stats(Y)
utils_train_test.disp_imgs_masks(X,Y)


# =============================================================================
# pick nonzero images and masks
# =============================================================================
if configs.nonZeroMasksOnly:
    nzMaskIndices=np.where(np.any(Y,axis=(1,2,3)))[0]
    X=X[nzMaskIndices]
    Y=Y[nzMaskIndices]
    utils_train_test.array_stats(X) 
    utils_train_test.array_stats(Y)
    utils_train_test.disp_imgs_masks(X,Y)
    
# =============================================================================
# train for n-Folds
# =============================================================================
evalMatric_nfolds=utils_train_test.trainNfolds(X,Y,configs)

# =============================================================================
# store/update evaluation metrics in configs
# =============================================================================
utils_train_test.updateRecoredInConfigs(configs.path2configs,"nFoldsMetrics",evalMatric_nfolds)
utils_train_test.updateRecoredInConfigs(configs.path2configs,"avgMetric",np.mean(evalMatric_nfolds))

# =============================================================================
# Obtain and store predictions of Ensemble model for train data
# =============================================================================
X,Y,ids_train=utils_train_test.load_data(configs,"train")
X,Y=utils_train_test.padArrays(X,Y,configs.padSize)
Y_pred=utils_train_test.getOutputAllFolds(X,configs,binaryMask=False)
Y_pred=utils_train_test.unpadArray(Y_pred,configs.padSize)
utils_train_test.storePredictions(configs,Y_pred,"train")


# =============================================================================
# leaderboard data
# =============================================================================
X_leaderboard,_,ids_leaderboard=utils_train_test.load_data(configs,"test")
X_leaderboard,_=utils_train_test.padArrays(X_leaderboard,None,configs.padSize)
utils_train_test.array_stats(X_leaderboard)


# =============================================================================
# get output for leaderboard data
# =============================================================================
Y_leaderboard=utils_train_test.getOutputAllFolds(X_leaderboard,configs,binaryMask=False)
Y_leaderboard=utils_train_test.unpadArray(Y_leaderboard,configs.padSize)

# =============================================================================
# store predictions of Ensemble model for Leaderboard data
# =============================================================================
utils_train_test.storePredictions(configs,Y_leaderboard,"test")

# =============================================================================
# convert outputs to Run Length Dict
# =============================================================================
rlcDict=utils_train_test.converMasksToRunLengthDict(Y_leaderboard>=configs.maskThreshold,ids_leaderboard)    
  

# =============================================================================
# crate submission
# =============================================================================
utils_train_test.createSubmission(rlcDict,configs)




