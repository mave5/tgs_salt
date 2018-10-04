

from keras import backend as K
_EPSILON = K.epsilon()


class customLoss:
    def __init__(self,nb_batch=8,nb_class=1,nb_sample=12288):
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

