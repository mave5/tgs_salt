import os
import pandas as pd
import ast
import json



class loadConfigsForExperiment(object):

    def __init__(self,path2allExperiments,experiment):
        
        # experiment
        self.experiment=experiment

        self.path2experiment=os.path.join(path2allExperiments,experiment)
        
        self.path2configs=os.path.join(self.path2experiment,'configs.csv')
        if os.path.exists(self.path2configs):
            configsDF=pd.read_csv(self.path2configs)
            print('-'*50)
            print ('Configs loaded!')
            print('-'*50)
        else:
            raise IOError("configs does not exist!")
    
        self.path2data=configsDF.loc[configsDF['Name']=='path2data','Value'].tolist()[0]        
        
        # dimensions
        img_hwc=configsDF.loc[configsDF['Name']=='img_hwc','Value'].tolist()[0]
        self.img_height,self.img_width,self.img_channel=ast.literal_eval(img_hwc)
        print('h,w,c loaded from configs!')
        print('img_height,img_width,img_channel: %s,%s,%s' %(self.img_height,self.img_width,self.img_channel))    
        print('-'*50)
    
        # normalization
        normalizationParams=configsDF.loc[configsDF['Name']=='normalizationParams','Value'].tolist()[0]
        self.normalizationParams=ast.literal_eval(normalizationParams)# convert string to dict     
        normalization_type=self.normalizationParams['normalization_type']
        print('normalization type loaded from settings!')
        print(json.dumps(self.normalizationParams,indent=4,sort_keys=True))    
        
    
        # hist equalization    
        histeq=configsDF.loc[configsDF['Name']=='histeq','Value'].tolist()[0]        
        if histeq=='True':
            self.histeq=True
        elif histeq=='False':
            self.histeq=False
        else:
            raise IOError('histeq not found!')
            
        print('histeq was loaded from settings!')
        print('histeq is %s' %histeq)
    
        # message
        self.message=configsDF.loc[configsDF['Name']=='message','Value'].tolist()[0]
        print('message loaded from Configs!')
        print (self.message)
    
        # train params
        trainingParams=configsDF.loc[configsDF['Name']=='trainingParams','Value'].tolist()[0]
        trainingParams=ast.literal_eval(trainingParams)    
        trainingParams['path2experiment']=self.path2experiment # we over write weightfolder
        trainingParams['pre_train']=False # we over write pre_train value
        trainingParams['normalizationParams']=normalizationParams
        trainingParams["augmentationParams"]=None
        print('params_train loaded from settings!')
        print('-'*50)
        self.trainingParams=trainingParams
    
        # padding size in case of zero padding
        pre_settings=configsDF.loc[configsDF['Name']=='pre_settings','Value'].tolist()[0]
        pre_settings=ast.literal_eval(pre_settings)    
        self.nFolds=pre_settings["nFolds"]
        self.maskThreshold=pre_settings["maskThreshold"]
        try:
            self.padSize=pre_settings["padSize"]
        except:
            self.padSize=(0,0)
            
        # model type    
        self.model_type=configsDF.loc[configsDF['Name']=='model_type','Value'].tolist()[0]
        print('model type was loaded from settings!')
                    
    