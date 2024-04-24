#%% MODULE BEGINS
module_name = 'PA4'
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
    base_path = r"C:\Users\brook\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE"
    path = os.path.join(base_path, r"C:\Users\melof\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")
    # os.chdir(r"C:\Users\brook\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")
    # os.chdir(r"C:\Users\melof\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")

#custom imports
#other imports
from copy import deepcopy as dpcpy

from matplotlib import pyplot as plt
import scipy.signal as signal
import numpy as np
#import mne
import csv
import pandas as pd
import seaborn as sns
import pickle as pckl
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
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
    pathSoIRoot = os.path.join(path, 'INPUT','DataSmall','sb1','se1')
    # pathSoIRoot = 'INPUT\\DataSmall\\sb1\\se1'
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
    plt.suptitle(chLabel + " Features\n", fontsize=18)
    for i, f1, f2 in zip(range(1,5), ['f0', 'f2', 'f4', 'f6'], ['f1', 'f3', 'f5', 'f7']):
        plt.subplot(2, 2, i)
        unique_f1 = sorted(features[f1].unique())
        plt.hist(features[features['class'] == 0][f1], bins=unique_f1, weights=features[features['class'] == 0][f2], color='g', label='sb1(class 0)', alpha=0.75)
        plt.hist(features[features['class'] == 1][f1], bins=unique_f1, weights=features[features['class'] == 1][f2], color='b', label='sb2(class 1)', alpha=0.75)
        plt.title(f1 + ' vs ' + f2, fontsize=16, loc='left')
        plt.xlabel(f1, fontsize=15)
        plt.ylabel(f2, fontsize=15)
        plt.legend()
    plt.tight_layout()
    plt.show()
    
def three_tier_test(features, chLabel):
    train_val, test = train_test_split(features, test_size=0.2, random_state=None)
    train, val = train_test_split(train_val, test_size=.25, random_state=None)
    
    X_train, Y_train = train.drop(columns=['class']), train['class'].astype('int')
    X_val, Y_val = val.drop(columns=['class']), val['class'].astype('int')
    X_test, Y_test = test.drop(columns=['class']), test['class'].astype('int')
    
    model = MLPClassifier(hidden_layer_sizes=100, max_iter=500, activation='logistic').fit(X_train, Y_train)
    plt.plot(model.loss_curve_, label='Training', c='b', alpha=0.75)
    model.fit(X_val, Y_val)
    plt.plot(model.loss_curve_, label='Validation', c='g', alpha=0.75)
    plt.title(chLabel + ' Epoch Error Curve')
    plt.xlabel('Epoch')
    plt.ylabel("Error")
    plt.legend()

    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)
    
    return {'ValTruth': Y_val, 'ValPrediction': val_predictions, 'TestTruth': Y_test, 'TestPrediction': test_predictions} 

def plotPerformance(modelResults, chLabel):
    #Calculating and plotting performance measures for val and test data
    plt.figure(figsize=(10,8))
    plt.title(chLabel + ' Performance Measures', fontsize=16)
    barWidth = 0.3
    for dataset in ['Val', 'Test']:
        precision =   precision_score(modelResults[dataset+'Truth'], modelResults[dataset+'Prediction'])
        recall =      recall_score(modelResults[dataset+'Truth'], modelResults[dataset+'Prediction'])
        f1 =          f1_score(modelResults[dataset+'Truth'], modelResults[dataset+'Prediction'])
        accuracy =    accuracy_score(modelResults[dataset+'Truth'], modelResults[dataset+'Prediction'])
        specificity = recall_score(modelResults[dataset+'Truth'], modelResults[dataset+'Prediction'], pos_label=0)
        auc =         roc_auc_score(modelResults[dataset+'Truth'], modelResults[dataset+'Prediction'])       
        performanceMeasures = {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy, 'Specificity': specificity, 'AUC': auc}
        if dataset == 'Val':
            plt.bar(performanceMeasures.keys(), performanceMeasures.values(), -barWidth, align='edge', color='g', label=dataset)
        else:
            plt.bar(performanceMeasures.keys(), performanceMeasures.values(), +barWidth, align='edge', color='b', label=dataset)  
    #plt.xaxis.set_tick_params(labelsize='medium')
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
            pathSoIRoot = os.path.join(path, 'INPUT','DataSmall',sb,se)
            # pathSoIRoot = 'INPUT\\DataSmall\\' + sb + '\\' + se
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
    
    #Trying out getting rid of sb se from columns
    features = pd.get_dummies(features, columns=['sb','se'])
    
    #Visualize features
    visualize(features, chLabel)
    #Split train/val and test data
    trainVal = features.loc[features['se_se1'] == 'se1']
    test = features.loc[features['se_se2'] == 'se2']
    #Output csv files
    trainValfilepath = os.path.join(path, 'OUTPUT', 'TrainValidateData.csv')
    testDatafilepath = os.path.join(path, 'OUTPUT', 'TestData.csv')
    trainVal.to_csv(trainValfilepath)
    test.to_csv(testDatafilepath)
    
    #Apply model and 3-tier testing scheme
    modelResults = three_tier_test(features, chLabel)
    #Get performance measures from model
    plotPerformance(modelResults, chLabel)
#             
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
#