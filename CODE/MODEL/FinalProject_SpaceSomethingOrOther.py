#%% MODULE BEGINS
module_name = 'Final_Project'
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
    #os.chdir(r"C:\Users\melof\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")

#custom imports
#other imports
from copy import deepcopy as dpcpy

from matplotlib import pyplot as plt
import scipy.signal as signal
import numpy as np
import csv
import pandas as pd
import seaborn as sns
import pickle as pckl
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf 
from tensorflow.keras.losses import MeanSquaredError 
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, LayerNormalization, Permute
from tensorflow.keras import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import ConfusionMatrixDisplay

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
            print("Channel found!\n")

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


def get_metrics(y_true, y_pred): 
    return {'Precision' : precision_score(y_true, y_pred), 
            'Recall' : recall_score(y_true, y_pred), 
            'F1' : f1_score(y_true, y_pred), 
            'Accuracy' : accuracy_score(y_true, y_pred), 
            'Specificity' : recall_score(y_true, y_pred, pos_label = 0),
            'AUC' : roc_auc_score(y_true, y_pred)} 
      
    
def plotFeatures(features, chLabel):
    plt.figure(figsize=(12,12))
    plt.suptitle(chLabel + " Features\n", fontsize=18)
    for i, f1, f2 in zip(range(1,5), ['f0', 'f2', 'f4', 'f6'], ['f1', 'f3', 'f5', 'f7']):
        plt.subplot(2, 2, i)
        plt.scatter(features[features['class'] == 0][f1], features[features['class'] == 0][f2], c='g', label='sb1(class 0)', alpha=0.6)
        plt.scatter(features[features['class'] == 1][f1], features[features['class'] == 1][f2], c='b', label='sb2(class 1)', alpha=0.6)
        plt.title(f1 + ' vs ' + f2, fontsize=16, loc='left')
        plt.xlabel(f1, fontsize=15)
        plt.ylabel(f2, fontsize=15)
        plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_model_run(y_true, y_pred, additional_plot, chLabel, model_type, figsize = (12,6)):
    model_metrics = get_metrics(y_true, y_pred)
    fig = plt.gcf()
    ax_0 = fig.add_subplot(1, 3, 2) if additional_plot is not None else fig.add_subplot(1,2,1)#subplots(1,3, figsize = figsize, tight_layout = True)
    ax_1 = fig.add_subplot(1, 3, 3) if additional_plot is not None else fig.add_subplot(1,2,2)
    ax_0.bar(model_metrics.keys(), model_metrics.values())
    ax_0.title.set_text('Performance Measures')
    ax_0.tick_params(axis='x', labelrotation = 30)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax = ax_1, colorbar=False)
    ax_1.title.set_text('Confusion Matrix')
    
    fig.suptitle(f'{model_type} - {chLabel}')
    return fig
    

def knn(trainVal, test):
    #Split train/val data
    train, val = train_test_split(trainVal, test_size=.25, random_state=None)

    k_values = list(range(1, len(val) + 1))
    accuracies = []

    #Find best k-value using train/val data
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train.drop(columns='class'), train['class'])

        test_predictions = model.predict(val.drop(columns='class'))

        accuracy = accuracy_score(val['class'], test_predictions)
        accuracies.append(accuracy)

    #Plot k-value vs accuracy
    # kvalueplot = plt.axes()
    fig = plt.figure(tight_layout = True)
    kvalueplot = fig.add_subplot(1,3,1)
    kvalueplot.scatter(k_values, accuracies, marker='o')
    kvalueplot.title.set_text('Accuracy vs. k Value')
    kvalueplot.set_xlabel('k Value')
    kvalueplot.set_ylabel('Accuracy')
    # plt.show

    #Get k-value from best model
    best_k = k_values[(accuracies.index(max(accuracies)))]
    print("Best k-value = ", best_k)

    #Run best model against test data
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(train.drop(columns='class'), train['class'])
    test_predictions = best_model.predict(test.drop(columns='class'))

    return test['class'], test_predictions, kvalueplot

def dtree(trainVal, test):
    best_accuracy = 0
    
    #Using k-fold cross validation k=5
    for k_fold in range(1,6):
        #Shuffle data
        trainVal.sample(frac=1)
        #Split data
        train, val = train_test_split(trainVal, test_size= .25)
        
        #Fit model with training data
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(train.drop(columns='class'), train['class'])
        
        #Test model against validation data
        test_predictions = model.predict(val.drop(columns='class'))

        #Get accuracy of model
        accuracy = accuracy_score(val['class'], test_predictions)

        #Comparing accuracies to find the best model
        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_model = model

    #Run best model against test data
    test_predictions = best_model.predict(test.drop(columns='class'))

    #Get results from test data
    fig = plt.figure(tight_layout = True)

    return test['class'], test_predictions, None

def ann(trainVal, test):
    trainVal = trainVal.copy()
    test = test.copy()
    train, val = train_test_split(trainVal, test_size = .25) 
    train_y = train.pop('class').values.astype(np.float32)
    test_y = test.pop('class').values.astype(np.float32)
    val_y = val.pop('class').values.astype(np.float32)

    model = Sequential()
    model.add(Dense(128, activation = 'gelu'))
    model.add(Dropout(.2))
    model.add(LayerNormalization())
    model.add(Dense(32, activation = 'gelu'))
    model.add(Dropout(.1))
    model.add(LayerNormalization())
    model.add(Dense(64, activation = 'gelu'))
    model.add(LayerNormalization())
    model.add(Dropout(.1))
    model.add(Dense(2, activation = 'softmax'))
    
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = .8, patience = 2, min_lr = 1e-6)
    early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 20, restore_best_weights = True)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), 
                  loss = 'sparse_categorical_crossentropy'#tf.nn.sparse_softmax_cross_entropy_with_logits(),
                  )#metrics= []])
    history = model.fit(train.to_numpy(), train_y,
                        batch_size = 4, 
                        epochs = 1000,
                        validation_data = [val.to_numpy(), val_y],
                        callbacks = [reduce_lr, early_stop])
    
    y_pred = np.argmax(np.array(model.predict(test)), axis = -1)


    loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig = plt.figure(tight_layout = True)
    ann_plot = fig.add_subplot(1, 3, 1)
    ann_plot.plot(loss, label= 'Training Loss', c = 'b', alpha = .75)
    ann_plot.plot(val_loss, label = 'Validation Loss', c = 'g', alpha = .75)
    ann_plot.title.set_text('Epoch Error Curve')
    ann_plot.set_xlabel('Epoch')
    ann_plot.set_ylabel("Error")
    ann_plot.legend()
    return test_y, y_pred, ann_plot

def svm(trainVal, test):
    best_accuracy = 0

    #Using k-fold cross validation k=5
    for k_fold in range(1,6):
        #Shuffle data
        trainVal.sample(frac=1)

        #Splitting into data and labels
        X_trainVal = trainVal.iloc[:, :-1]
        y_trainVal = trainVal.iloc[:, -1]

        #Splitting trainVal into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.25, random_state=42)
        
        #Initialize and train the SVM model
        svm_model = SVC()
        svm_model.fit(X_train, y_train)

        #Predict on the validation set
        y_pred_val = svm_model.predict(X_val)

        #Get accuracy of model
        val_accuracy = accuracy_score(y_val, y_pred_val)

        #Comparing accuracies to find the best model
        if(val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_model = svm_model

    fig = plt.figure()
    
    #Run best model against test data
    test_predictions = best_model.predict(test.drop(columns='class'))

    #Get results from test data
 
    return test['class'], test_predictions, None

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
    plotFeatures(features, chLabel)

    #Split train/val and test data
    trainVal = features.loc[features['se'] == 'se1']
    test = features.loc[features['se'] == 'se2']

    #Output CSV files to input folder to be used in models
    trainVal.to_csv('INPUT\\'+chLabel+'TrainValidateData.csv',index=False)
    test.to_csv('INPUT\\'+chLabel+'TestData.csv',index=False)

    #Reading features from input CSV files
    trainVal_path = 'INPUT\\'+chLabel+'TrainValidateData.csv'
    test_path = 'INPUT\\'+chLabel+'TestData.csv'
    trainVal_data = pd.read_csv(trainVal_path) 
    test_data = pd.read_csv(test_path)

    #Dropping columns no longer needed
    trainVal_data = trainVal_data.drop(columns=['sb','se','streamID'])
    test_data = test_data.drop(columns=['sb','se','streamID'])

    #Running KNN model
    y_true, y_pred, kvalueplot = knn(trainVal_data, test_data)
    fig = plot_model_run(y_true, y_pred, kvalueplot, chLabel, 'KNN')
    plt.show()
    print(chLabel, ':', 'KNN Confusion Matrix\n', confusion_matrix(y_true, y_pred))

    #Running DTree model
    y_true, y_pred, _ = dtree(trainVal_data, test_data)
    plot_model_run(y_true, y_pred, None, 'DTree', chLabel)
    print(chLabel, ':', 'DTree Confusion Matrix\n', confusion_matrix(y_true, y_pred))

    #Running ANN model
    # ann_performance, ann_cMatrix = ann(trainVal_data, test_data)
    y_true, y_pred, ann_plot = ann(trainVal_data, test_data)
    fig = plot_model_run(y_true, y_pred, ann_plot, 'ANN', chLabel)
    plt.show()
    print(chLabel, ':', 'ANN Confusion Matrix\n', confusion_matrix(y_true, y_pred))
    
    #Running SVM
    y_true, y_pred, _ = svm(trainVal_data, test_data)
    fig = plot_model_run(y_true, y_pred, None, 'SVM', chLabel)
    plt.show()
    print(chLabel, ':', 'SVM Confusion Matrix\n', confusion_matrix(y_true, y_pred))
#             
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
#