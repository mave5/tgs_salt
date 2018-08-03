import re
import os
#import pandas as pd
import sys
import json
import csv
import numpy as np


def sort_nicely( l ): 
  """ Sort the given list in the way that humans expect. 
  """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  l.sort( key=alphanum_key ) 


def getInputFromUser(dictX,textX=""):
    print('-'*50)
    pythonVersion=sys.version_info
    while True:
        print(json.dumps(dictX,indent=4,sort_keys=True))    
        if pythonVersion[0]==2:        
            key1=raw_input(textX)
        else:
            key1=input(textX)
        if key1 in dictX:
            item=dictX[key1]
            break
    return item


def print_version(*vars):
    for var in vars:
        module = __import__(var)    
        print ('%s: %s' %(var,module.__version__))

def get_version(var):
    module = __import__(var)    
    return module.__version__

# sort based on timeutils_config
def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))


def read_csv(path2settingsTemp):
    with open(path2settingsTemp, mode='r') as settings_file:
        settings_reader = csv.reader(settings_file)
        settings = {rows[1]: rows[2:][0] for rows in settings_reader}
        
    return settings

def getAnExperiment(path2allExperiments,agileIterationNum,projectStage):
    
    # get a list of all existing experiments    
    listOfExperiments=os.listdir(path2allExperiments)
    sort_nicely(listOfExperiments)
    
    # a dict to collect experiments and messages
    listOfExperimentsMessages={}
    for exp in listOfExperiments:
        expi=exp.replace(".", "") # convert version to a sudo number
        path2settingsTemp=os.path.join(path2allExperiments,exp,'configs.csv')
       
        try:
            settingsDFTemp=read_csv(path2settingsTemp)
            messageTemp = settingsDFTemp['message']
            listOfExperimentsMessages[expi]="%s %s" %(exp , messageTemp)
        except:
            pass
    
    # if there is no experiments we start from index 0
    if len(listOfExperiments)==0:
       expi='0' 
       
    # adding a new item to dict if we want to create a new experiment
    # parse version into: project phase, iteration phase, experiment number
    try:
        pj,it,expi=exp.split('.')
    except:
        it=agileIterationNum
    # in case of a new iteration, we start from index 0
    if int(agileIterationNum)>int(it): 
        expi='0'     
    expi=str(int(expi)+1)
    expiVer="0"
    mvTemp=projectStage+"."+agileIterationNum+"."+expi
    listOfExperimentsMessages[expiVer]=mvTemp+" create a new experiment"
    
    experiment=getInputFromUser(listOfExperimentsMessages,'Select an experiment: ')
    experiment=experiment.split(' ')[0]
    print ('%s selected' %experiment)   
    
    return experiment
