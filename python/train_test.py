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



#%%

# =============================================================================
# load data    
# =============================================================================
X,Y,ids=utils_train_test.load_data(configs,"train")
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
# leaderboard data
# =============================================================================
X_leaderboard,_,ids_leaderboard=utils_train_test.load_data(configs,"test")
utils_train_test.array_stats(X_leaderboard)


# =============================================================================
# get leaderboard data
# =============================================================================
Y_leaderboard=utils_train_test.getOutputAllFolds(X_leaderboard,configs)


# =============================================================================
# convert outputs to Run Length Dict
# =============================================================================
rlcDict=utils_train_test.converMasksToRunLengthDict(Y_leaderboard,ids_leaderboard)    
  

# =============================================================================
# crate submission
# =============================================================================
utils_train_test.createSubmission(rlcDict,configs)