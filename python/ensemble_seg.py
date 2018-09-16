# train test
#import configs
import os
from utils import utils_train_test
import sys
from utils import utils_config

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
experiments=["0.3.7","0.3.11","0.5.3","0.6.10","0.6.14","0.6.16"]
experimentsJoin="".join("_"+e for e in experiments)

Y_pred_ensemble=utils_train_test.getOutputEnsemble_evalMetric(path2allExperiments,experiments,data_type="train",path2data=path2data)    
utils_train_test.storePredictionsEnsemble(path2predictions,Y_pred_ensemble,"train"+experimentsJoin)
del Y_pred_ensemble

Y_leaderboard=utils_train_test.getOutputEnsemble_evalMetric(path2allExperiments,experiments,data_type="test",path2data=path2data)    
utils_train_test.storePredictionsEnsemble(path2predictions,Y_leaderboard,"test"+experimentsJoin)


# =============================================================================
# convert outputs to Run Length Dict
# =============================================================================
yesNoDict={"y": "yes",
           "n": "no",
           }
yn=utils_config.getInputFromUser(yesNoDict,"Create submission? ")
if yn is "no":
    sys.exit()
_,_,ids_leaderboard=utils_train_test.load_data_ensemble(path2data,data_type="test")
rlcDict=utils_train_test.converMasksToRunLengthDict(Y_leaderboard>=maskThreshold,ids_leaderboard)    

# =============================================================================
# crate submission
# =============================================================================
utils_train_test.createSubmissionEnsemble(rlcDict,path2Ensemble,info=experimentsJoin)




