#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''if __name__ == "__main__":
import os
#os.chdir("./../..")
#
#custom imports
#other imports
from copy import deepcopy as dpcpy

from matplotlib import pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import seaborn as sns'''
import pickle as pckl

#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
subject = 'sb1'
session = 'se1'
pathSoIRoot = 'INPUT\\stream'
pathSoi = f'{pathSoIRoot}\\'
soi_file = '1_132_bk_pic.pckl'

#Load SoI object
with open(f'{pathSoi}{soi_file}', 'rb') as fp:
       soi = pckl.load(fp)

#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here
def main():
    pass
#
#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
'''if __name__ == "__main__":
print(f"\"{module_name}\" module begins.")'''
#TEST Code
