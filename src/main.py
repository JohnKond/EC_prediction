# Import necessary libraries
import numpy as np
import pandas as pd
import time
import argparse
from dataUtils import read_dataset, data_preprocess, data_split
from modelFinetune import model_finetune
from modelTrain import train_model_pipeline

# for plots
import matplotlib.pyplot as plt
import seaborn as sns



OUTPUT_FOLDER = 'output'


parser = argparse.ArgumentParser(description="Utility functions for storing and loading machine learning models.")
parser.add_argument("--model", type=str, required=True, choices=["RF", "XGB", "SVR", "NN"],
                    help="Specify the machine learning model (RF, XGB, NN).")
parser.add_argument("--geo_info", action="store_true",
                    help="Include the coordinates information (X,Y) in model training.")
parser.add_argument("--task", type=str, required=True, choices=["finetune", "train", "evaluate"],
                    help="Specify the task (finetune, train, evaluate).")
parser.add_argument("--test", action="store_true",
                    help="Test run.")
args = parser.parse_args()




def main(): 

    df_01_09_init, df_02_09_init =  read_dataset()
    df_01_09 = data_preprocess(df_01_09_init)
    X_train, X_test, y_train, y_test = data_split(df_01_09, geo_info=args.geo_info, test=args.test)
    
    if args.task == 'train':
        print(f'- Train the {args.model} model on 01-09 dataset..')
        train_model_pipeline(args.model, X_train, X_test, y_train, y_test, df_01_09.columns.tolist())
    if args.task == 'finetune':
        print(f'- Finetune a {args.model} model on 01-09 dataset..')
        model_finetune(args.model, X_train, y_train)
    if args.task == 'evaluate':
        print(f'- Evaluate the {args.model} model on 02-09 dataset..')

    return
    

if __name__ == "__main__": 
    main()



