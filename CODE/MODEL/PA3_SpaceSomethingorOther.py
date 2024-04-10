#%% MODULE BEGINS
module_name = 'PA3'
'''
Version: <***>
Description:
<***>
Authors: Brooks Schafer, Melinda McElveen
Date Created : <***>
Date Last Updated: <***>
Doc:
<***>
Notes:
<***>
'''
#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
    os.chdir(r"C:\Users\brook\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")
    # os.chdir(r"C:\Users\melof\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")

#custom imports
#other imports
from copy import deepcopy as dpcpy

from matplotlib import pyplot as plt
import scipy.signal as signal
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pckl
from scipy.stats import kurtosis, skew
import time
#
#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
chLabel = input("Enter a channel label(ex, M1):").upper()
#
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here
def getIndex(chLabel):
    pathSoIRoot = 'INPUT\\DataSmall\\sb1\\se1'
    pathSoi = f'{pathSoIRoot}\\'
    soi_file = '1_1_bk_pic.pckl'
    #Load SoI objectM1
    with open(f'{pathSoi}{soi_file}', 'rb') as fp:
        soi = pckl.load(fp)

    #Finding channel index
    for i in range(len(soi['info']['eeg_info']['channels'])):
        if soi['info']['eeg_info']['channels'][i]['label'][0] == chLabel:
            chIndex = i
            print("Index found!\n")

    return chIndex
    
def applyFilters(stream, sampleFreq):
    #Apply notch filter
    notch_freq = [60, 120, 180, 240]
    for freq in notch_freq:
        b_notch, a_notch = signal.iirnotch(w0=freq, Q=50, fs=sampleFreq)
        stream = signal.filtfilt(b_notch, a_notch, stream)
    
    #Apply impedance filter
    impedance = [124, 126]
    b_imp, a_imp = signal.butter(N=4, Wn=[impedance[0] / (sampleFreq/2), impedance[1] / (sampleFreq / 2)], btype='bandstop')
    stream = signal.filtfilt(b_imp, a_imp, stream)
    
    #Apply bandpass filter
    bandpass = [0.5, 32]
    b_bandpass, a_bandpass = signal.butter(N=4, Wn=[bandpass[0] / (sampleFreq/2), bandpass[1] / (sampleFreq/2)], btype='bandpass')
    stream = signal.filtfilt(b_bandpass, a_bandpass, stream)

    #Apply rereferencing
    stream -= np.mean(stream)

    return stream

def getFeatures(stream, streamID, sb, se, window_size=100, overlap=0.25):
    num_samples = len(stream)
    window_size = min(window_size, num_samples)
    step_size = int(window_size * (1 - overlap))
    
    #Generating window stats
    windowStats = []
    for start in range(0, num_samples, step_size):
        end = start + window_size
        window = stream[start:end]
        windowStats.append({
            'sb':       sb,
            'se':       se,
            'streamID': streamID,
            'windowID': len(windowStats),
            'mean':     np.mean(window),
            'std':      np.std(window),
            'kur':      kurtosis(window),
            'skew':     skew(window)
        })
    windowStats = pd.DataFrame(windowStats)
    
    #Generating features from window stats
    newFeatures = []
    newFeatures.append({
        'sb':       sb,
        'se':       se,
        'streamID': streamID,
        'f0':       np.mean(windowStats['mean']),
        'f1':       np.std(windowStats['mean']),
        'f2':       np.mean(windowStats['std']),
        'f3':       np.std(windowStats['std']),
        'f4':       np.mean(windowStats['kur']),
        'f5':       np.std(windowStats['kur']),
        'f6':       np.mean(windowStats['skew']),
        'f7':       np.std(windowStats['skew']),
        'class':    0
    })
    newFeatures = pd.DataFrame(newFeatures)
    return newFeatures
    
def visualize(features, chLabel):
    plt.figure(figsize=(12,12))
    plt.suptitle(chLabel + " Features", fontsize=18)
    for i, f1, f2 in zip(range(1,5), ['f0', 'f2', 'f4', 'f6'], ['f1', 'f3', 'f5', 'f7']):
        plt.subplot(2, 2, i)
        plt.scatter(features[features['class'] == 0][f1], features[features['class'] == 0][f2], c='g', label='sb1(class 0)', alpha=0.75)
        plt.scatter(features[features['class'] == 1][f1], features[features['class'] == 1][f2], c='b', label='sb2(class 1)', alpha=0.75)
        plt.title(f1 + ' vs ' + f2, fontsize=16)
        plt.xlabel(f1, fontsize=15)
        plt.ylabel(f2, fontsize=15)
        plt.legend()
    plt.tight_layout()
    plt.show()
#
#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
def main():
    features = pd.DataFrame(columns=['sb', 'se', 'streamID', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'class'])
    chIndex = getIndex(chLabel)
    
    streamID = 0
    for sb in ['sb1', 'sb2']:
        for se in ['se1', 'se2']:
            pathSoIRoot = 'INPUT\\DataSmall\\' + sb + '\\' + se
            files = os.listdir(pathSoIRoot)
            for file in files:
                pathSoi = f'{pathSoIRoot}\\'
                soi_file = file
                #Load SoI object
                with open(f'{pathSoi}{soi_file}', 'rb') as fp:
                    soi = pckl.load(fp)
                
                #Apply filters
                filteredStream = applyFilters(soi['series'][chIndex], soi['info']['eeg_info']['effective_srate'])
                #Get features
                newFeatures = getFeatures(filteredStream, streamID, sb, se)
                #Add features to DF
                features = pd.concat([features, newFeatures], ignore_index=True)
                
                streamID += 1
    features.loc[features['sb'] == 'sb2', 'class'] = 1
    
    #Visualize features
    visualize(features, chLabel)
    #Split train/val and test data
    trainVal = features.loc[features['se'] == 'se1']
    test = features.loc[features['se'] == 'se2']
    #Output csv files
    trainVal.to_csv('OUTPUT\\TrainValidateData.csv')
    test.to_csv('OUTPUT\\TestData.csv')
#             
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
#