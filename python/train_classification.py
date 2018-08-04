# train test
import configs_classification as configs
from utils import utils_train_test
import numpy as np


#%% model summary
configs.showModelSummary=True

# =============================================================================
# load data    
# =============================================================================
X,y,ids_train=utils_train_test.load_data_classification(configs,"train")
utils_train_test.array_stats(X)
utils_train_test.array_stats(y)
utils_train_test.disp_imgs_masks_labels(X,y)



# =============================================================================
# train for n-Folds
# =============================================================================
evalMatric_nfolds=utils_train_test.trainNfolds_classification(X,y,configs)

# =============================================================================
# store/update evaluation metrics in configs
# =============================================================================
utils_train_test.updateRecoredInConfigs(configs.path2configs,"nFoldsMetrics",evalMatric_nfolds)
utils_train_test.updateRecoredInConfigs(configs.path2configs,"avgMetric",np.mean(evalMatric_nfolds))
utils_train_test.updateRecoredInConfigs(configs.path2configs,"segModelVersion",configs.seg_model_version,overwrite=True)


# =============================================================================
# leaderboard data
# =============================================================================
X_leaderboard,_,ids_leaderboard=utils_train_test.load_data_classification(configs,"test")
utils_train_test.array_stats(X_leaderboard)


# =============================================================================
# get leaderboard data
# =============================================================================
Y_leaderboard=utils_train_test.getOutputAllFolds_classification(X_leaderboard,configs)

# =============================================================================
# store predictions of Ensemble model for Leaderboard data
# =============================================================================
utils_train_test.storePredictions(configs,Y_leaderboard,"test")

# =============================================================================
# convert outputs to Run Length Dict
# =============================================================================
rlcDict=utils_train_test.converMasksToRunLengthDict(Y_leaderboard,ids_leaderboard)    
  

# =============================================================================
# crate submission
# =============================================================================
utils_train_test.createSubmission(rlcDict,configs)


