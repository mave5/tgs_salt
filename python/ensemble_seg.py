# train test
#import configs
import os
from utils import utils_train_test

#%%
maskThreshold=0.5

# =============================================================================
# paths
# =============================================================================
path2data="../data/"
path2output="./output/"
path2allExperiments=os.path.join(path2output,'experiments/')
path2Ensemble=os.path.join(path2allExperiments,"0.0.0")
path2predictions=os.path.join(path2Ensemble,"predictions")
if not os.path.exists(path2predictions):
    os.makedirs(path2predictions)
    print(path2predictions +" created!")

# =============================================================================
# list of experiments to be ensembled
# =============================================================================
experiments=["0.2.5","0.2.10"]
Y_pred_ensemble=utils_train_test.getOutputEnsemble(path2allExperiments,experiments,data_type="train")    
utils_train_test.storePredictionsEnsemble(path2predictions,Y_pred_ensemble,"train")
del Y_pred_ensemble

Y_leaderboard=utils_train_test.getOutputEnsemble(path2allExperiments,experiments,data_type="test")    
utils_train_test.storePredictionsEnsemble(path2predictions,Y_leaderboard,"test")

# =============================================================================
# convert outputs to Run Length Dict
# =============================================================================
_,_,ids_leaderboard=utils_train_test.load_data_ensemble(path2data,data_type="test")
rlcDict=utils_train_test.converMasksToRunLengthDict(Y_leaderboard>=maskThreshold,ids_leaderboard)    

# =============================================================================
# crate submission
# =============================================================================
utils_train_test.createSubmissionEnsemble(rlcDict,path2Ensemble)




