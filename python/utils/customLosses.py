from keras import losses

from keras import backend as K
_EPSILON = K.epsilon()


class customCategoricalCrossEntropy:
    def __init__(self,nb_batch=8,nb_class=1,nb_sample=16384):
        self.nb_batch=nb_batch
        self.nb_class=nb_class        
        self.nb_sample=nb_sample
        self.__name__="categorigal_crossentropy_nogt"
    
    def __call__(self, y_true, y_pred):        
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        
        # obtain number of pixels and classes
        nb_sample=self.nb_sample
        nb_class=self.nb_class

        # find out class predictions        
        pred=K.argmax(y_pred,axis=-1)
    
        # find out which classes are missing annotations
        y_trueAny=K.any(y_true,axis=1,keepdims=True)
        
        # checking two conditions
        # class c does not have annotation: y_trueAnyRepeat
        # (not) prediction is equal to class c: K.not_equal(pred,cls)
        backGroundWeights=1
        for cls in range(0,nb_class):
            y_trueAnyRepeat=K.repeat_elements(y_trueAny[:,:,cls],nb_sample,axis=1)
            backGroundWeights*=K.not_equal(pred,cls)+y_trueAnyRepeat
        
        loss=backGroundWeights*y_true[:,:,0]*K.log(y_pred[:,:,0])
        for cls in range(1,nb_class):
            loss+=y_true[:,:,cls]*K.log(y_pred[:,:,cls])
            
        return -loss

def focal_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    gamma=2
    out=-y_true*(1-y_pred)**gamma*K.log(y_pred)-(1-y_true)*K.log(1-y_pred)*y_pred**gamma
    
    return out #K.mean(out,axis=(0,1))


def iou_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    smooth=_EPSILON
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f, axis=1, keepdims=True)  
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) -intersection
    iou=(intersection+smooth) / (union+smooth)
    return iou

def compute_precision_at_tensor(iou, threshold):
    tp = iou >= threshold
    fp= iou < threshold
    fn= iou < threshold
    return (tp+_EPSILON) / (tp + fp + fn+_EPSILON)

def computePrecision_tensor(gt, prediction):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    thresholds=K.constant(thresholds)
    iou = iou_tensor(gt, prediction)
    precisions=0.0
    for th in thresholds:
        prc=compute_precision_at_tensor(iou, th)
        precisions+=prc
    return precisions / K.eval(thresholds.shape[0])


def averagePrecision_loss(y_true, y_pred):
    loss=-computePrecision_tensor(y_true, y_pred)
    return K.mean(loss)

def loss_combined(y_true,y_pred):
    loss=losses.binary_crossentropy(y_true,y_pred)-K.log(jacard_coef(y_true,y_pred))
    return loss

def jacard_coef(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)

    smooth=_EPSILON
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    #print K.dtype(y_true_f),K.dtype(y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) 

    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) -intersection+ smooth

    jaccard=K.mean((intersection+smooth) / union)
    return jaccard


def intersectionOverUnion(y_true, y_pred):
    smooth=_EPSILON
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) 
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) - intersection
    return K.mean((intersection+ smooth) / (union+ smooth))

def iou_loss(y_true, y_pred):
    loss=1-jacard_coef(y_true,y_pred)
    return loss

def dice_coef(y_true, y_pred):
    smooth=1
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
