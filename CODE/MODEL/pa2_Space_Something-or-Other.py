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
# if __name__ == "__main__":
#     import os
#     os.chdir(r"C:\Users\melof\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther")

#custom imports
#other imports
from copy import deepcopy as dpcpy

from matplotlib import pyplot as plt
# import mne
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

    sampleFreq = soi['info']['eeg_info']['effective_srate']
    
    m1Stream = soi['series'][19]
    m1tStamp = soi['tStamp']
    m1tStamp -= m1tStamp[0]
    m1Label = soi['info']['eeg_info']['channels'][19]['label']

    p7Stream = soi['series'][14]
    p7tStamp = soi['tStamp']
    p7tStamp -= p7tStamp[0]
    p7Label = soi['info']['eeg_info']['channels'][14]['label']

    p3Stream = soi['series'][6]
    p3tStamp = soi['tStamp']
    p3tStamp -= p3tStamp[0]
    p3Label = soi['info']['eeg_info']['channels'][6]['label']

    # Plot streams
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(m1tStamp, m1Stream)
    plt.title(m1Label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(p7tStamp, p7Stream)
    plt.title(p7Label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(p3tStamp, p3Stream)
    plt.title(p3Label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
    
       
#
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
#TEST Code
