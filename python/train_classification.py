# train test
import configs_classification as configs
from utils import utils_train_test
import numpy as np


#%% model summary
configs.showModelSummary=True

# =============================================================================
# load data    
# =============================================================================
#X,y,ids_train,masks=utils_train_test.load_data_classify_prob(configs,"train")
X,y,ids_train,masks=utils_train_test.load_data_classify_prob_largeMaskThreshold(configs,"train")
utils_train_test.array_stats(X)
utils_train_test.array_stats(y)
utils_train_test.disp_imgs_masks_labels(X,y)

# =============================================================================
# train for n-Folds
# =============================================================================
evalMatric_nfolds=utils_train_test.trainNfolds_classification(X,y,configs,masks)


# =============================================================================
# get predictions for training data
# =============================================================================
Y_pred,Y_pred_soft,y_pred=utils_train_test.getOutputAllFolds_classify_prob_ySoft(X,configs)


# =============================================================================
# store predictions for training data
# =============================================================================
utils_train_test.storePredictionsWithIds_classify(configs,Y_pred_soft,y_pred,ids_train,"train")

# =============================================================================
# store/update evaluation metrics in configs
# =============================================================================
utils_train_test.updateRecoredInConfigs(configs.path2configs,"nFoldsMetrics",evalMatric_nfolds)
utils_train_test.updateRecoredInConfigs(configs.path2configs,"avgMetric",np.mean(evalMatric_nfolds))
utils_train_test.updateRecoredInConfigs(configs.path2configs,"segModelVersion",configs.seg_model_version,overwrite=True)

# =============================================================================
# Error Analysis
# =============================================================================
y_temp=y_pred[:,0]>=0.5
sumY=np.sum(Y_pred,axis=(1,2,3))
errorInds=np.where(y!=y_temp)[0]
accuracy=1-len(errorInds)/(len(y)*1.0)
print("accuracy: %.2f" %accuracy)
utils_train_test.disp_imgs_2masks_labels(X[errorInds],masks[errorInds],y[errorInds],y_pred[errorInds])

# =============================================================================
# leaderboard data
# =============================================================================
X_leaderboard,_,ids_leaderboard,_=utils_train_test.load_data_classify_prob(configs,"test")
utils_train_test.array_stats(X_leaderboard)


# =============================================================================
# get predictions for leaderboard data
# =============================================================================
#Y_leaderboard=utils_train_test.getOutputAllFolds_classify_prob(X_leaderboard,configs)
Y_leaderboard,Y_leaderboard_soft,y_leaderboard=utils_train_test.getOutputAllFolds_classify_prob_ySoft(X_leaderboard,configs)


# =============================================================================
# store predictions of Ensemble model for Leaderboard data
# =============================================================================
utils_train_test.storePredictionsWithIds_classify(configs,Y_leaderboard_soft,y_leaderboard,ids_leaderboard,"test")

# =============================================================================
# convert outputs to Run Length Dict
# =============================================================================
rlcDict=utils_train_test.converMasksToRunLengthDict(Y_leaderboard,ids_leaderboard)    
  

# =============================================================================
# crate submission
# =============================================================================
utils_train_test.createSubmission(rlcDict,configs)



