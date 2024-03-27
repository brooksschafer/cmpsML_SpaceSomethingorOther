#%% MODULE BEGINS
module_name = 'PA2'
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
#
#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
    os.chdir(r"C:\Users\brook\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")

#custom imports
#other imports
from copy import deepcopy as dpcpy

from matplotlib import pyplot as plt
import scipy.signal as signal
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

#Reading desired channel labels
ch1Labl, ch2Labl, ch3Labl = input("Enter 3 channel labels (ex: P7 P3 Pz):").split()
#
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here
def getIndex(ch1Labl, ch2Labl, ch3Labl):
    #Iterating through SoI channels to find needed channel indexes
    for i in range(len(soi['info']['eeg_info']['channels'])):
        if soi['info']['eeg_info']['channels'][i]['label'][0] == ch1Labl:
            ch1Indx = i
            print("Index1 found!\n")
        elif soi['info']['eeg_info']['channels'][i]['label'][0] == ch2Labl:
            ch2Indx = i
            print("Index2 found!\n")
        elif soi['info']['eeg_info']['channels'][i]['label'][0] == ch3Labl:
            ch3Indx = i
            print("Index3 found!\n")
    
    return ch1Indx, ch2Indx, ch3Indx

def getData(ch1Indx, ch2Indx, ch3Indx):
    #Reading timestamp and sample frequency
    tStamp = soi['tStamp']
    tStamp -= tStamp[0]
    sampleFreq = soi['info']['eeg_info']['effective_srate']

    #Reading streams, timestamp and labels into dictionaries
    ch1 = {'stream': soi['series'][ch1Indx], 'tStamp': tStamp, 'label': soi['info']['eeg_info']['channels'][ch1Indx]['label']}
    ch2 = {'stream': soi['series'][ch2Indx], 'tStamp': tStamp, 'label': soi['info']['eeg_info']['channels'][ch2Indx]['label']}
    ch3 = {'stream': soi['series'][ch3Indx], 'tStamp': tStamp, 'label': soi['info']['eeg_info']['channels'][ch3Indx]['label']}
    
    return ch1, ch2, ch3, sampleFreq

def plotData(ch1, ch2, ch3):
    plt.figure(figsize=(12, 6))
    for i, channel in zip(range(1,4), [ch1, ch2, ch3]):
        plt.subplot(3, 1, i)
        plt.plot(channel['tStamp'], channel['stream'])
        plt.title(channel['label'])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()  

def applyFilters(ch1, ch2, ch3, sampleFreq):
    #Apply notch filter
    notch_freq = [60, 120, 180, 240]
    for freq in notch_freq:
        b_notch, a_notch = signal.iirnotch(w0=freq, Q=50, fs=sampleFreq)
        ch1Stream_notched = signal.filtfilt(b_notch, a_notch, ch1['stream'])
        ch2Stream_notched = signal.filtfilt(b_notch, a_notch, ch2['stream'])
        ch3Stream_notched = signal.filtfilt(b_notch, a_notch, ch3['stream'])
    
    #Apply impedance filter
    impedance = [124, 126]
    b_imp, a_imp = signal.butter(N=4, Wn=[impedance[0] / (sampleFreq/2), impedance[1] / (sampleFreq / 2)], btype='bandstop')
    ch1Stream_impeded = signal.filtfilt(b_imp, a_imp, ch1Stream_notched)
    ch2Stream_impeded = signal.filtfilt(b_imp, a_imp, ch2Stream_notched)
    ch3Stream_impeded = signal.filtfilt(b_imp, a_imp, ch3Stream_notched)
    
    #Apply bandpass filter
    bandpass = [0.5, 32]
    b_bandpass, a_bandpass = signal.butter(N=4, Wn=[bandpass[0] / (sampleFreq/2), bandpass[1] / (sampleFreq/2)], btype='bandpass')
    ch1Stream_bandpass = signal.filtfilt(b_bandpass, a_bandpass, ch1Stream_impeded)
    ch2Stream_bandpass = signal.filtfilt(b_bandpass, a_bandpass, ch2Stream_impeded)
    ch3Stream_bandpass = signal.filtfilt(b_bandpass, a_bandpass, ch3Stream_impeded)
    filteredData = [ch1Stream_bandpass, ch2Stream_bandpass, ch3Stream_bandpass]

    return filteredData

def plotFilteredData(ch1, ch2, ch3, filteredData):
    plt.figure(figsize=(14, 10))
    chIndex = 0
    pltIndex = 0
    for channel in [ch1, ch2, ch3]:
        #Plotting original data
        pltIndex += 1
        plt.subplot(3, 2, pltIndex)
        plt.plot(channel['tStamp'], channel['stream'])
        plt.title(channel['label'][0] + ' Original')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        #Plotting filtered data
        pltIndex += 1
        plt.subplot(3, 2, pltIndex)
        plt.plot(channel['tStamp'], filteredData[chIndex])
        plt.title(channel['label'][0] + ' Filtered')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        chIndex += 1

    plt.tight_layout()
    plt.show() 
#    
#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
def main():
    ch1Indx, ch2Indx, ch3Indx = getIndex(ch1Labl, ch2Labl, ch3Labl)
    ch1, ch2, ch3, sampleFreq = getData(ch1Indx, ch2Indx, ch3Indx)
    plotData(ch1, ch2, ch3)

    filteredData = applyFilters(ch1, ch2, ch3, sampleFreq)
    #Rereferencing filtered data
    for channel in filteredData:
        channel -= np.mean(channel)
    plotFilteredData(ch1, ch2, ch3, filteredData)
#
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
#