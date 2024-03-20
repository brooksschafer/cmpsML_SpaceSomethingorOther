#%% MODULE BEGINS
module_name = 'PA2'
'''
Version: <***>
Description:
<***>
Authors:
<***>
Date Created : <***>
Date Last Updated: <***>
Doc:
<***>
Notes:
<***>
'''
#
#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
    os.chdir(r"C:\Users\brook\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")

#custom imports
#other imports
from copy import deepcopy as dpcpy

from matplotlib import pyplot as plt
#import mne
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pckl
#
#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pathSoIRoot = 'INPUT\\stream'
pathSoi = f'{pathSoIRoot}\\'
soi_file = '1_132_bk_pic.pckl'

#Load SoI object
with open(f'{pathSoi}{soi_file}', 'rb') as fp:
       soi = pckl.load(fp)
#
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here
#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
def main():
    '''Channel Index: 
    M1: 19
    P7: 14
    P3: 6   '''
    sampelFreq = soi['info']['eeg_info']['effective_srate']
    
    m1Stream = soi['series'][19]
    m1tStamp = soi['tStamp']
    m1Label = soi['info']['eeg_info']['channels'][19]['label']

    p7Stream = soi['series'][14]
    p7tStamp = soi['tStamp']
    p7Label = soi['info']['eeg_info']['channels'][14]['label']

    p3Stream = soi['series'][6]
    p3tStamp = soi['tStamp']
    p3Label = soi['info']['eeg_info']['channels'][6]['label']

    #Plot streams
       
#
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
#TEST Code
