# Import necessary libraries
import numpy as np
import pandas as pd
import time
import argparse

from dataUtils import read_dataset, data_preprocess

# for plots
import matplotlib.pyplot as plt
import seaborn as sns



DATA_FOLDER = 'data'
MODELS_FOLDER = 'models'
OUTPUT_FOLDER = 'output'


parser = argparse.ArgumentParser(description="Utility functions for storing and loading machine learning models.")
parser.add_argument("--model", type=str, required=True, choices=["RF", "XGB", "SVR", "NN"],
                    help="Specify the machine learning model (RF, XGB, SVR, NN).")
parser.add_argument("--task", type=str, required=True, choices=["finetune", "train", "evaluate"],
                    help="Specify the task (finetune, train, evaluate).")
args = parser.parse_args()




def main(): 
    df_01_09_init, df_02_09_init =  read_dataset()
    df_01_09 = data_preprocess(df_01_09_init)



    if args.task == 'train':
        
    return
    



if __name__=="__main__": 
    main()



