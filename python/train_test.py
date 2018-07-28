# train test
import configs
from utils import utils_train_test
import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm_notebook
from keras.preprocessing import image
from sklearn.model_selection import ShuffleSplit
from utils import models

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
# train for n-Folds
# =============================================================================
utils_train_test.trainNfolds(X,Y,configs)



