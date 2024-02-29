"""
dataUtils.py

This file provides utility functions for handling data operations, such as reading and preprocessing datasets.

Functions:
- read_dataset()
- data_preprocessing(df)
- data_split(df, geo_info=False)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FOLDER = 'data'

def read_dataset():
    """
    Read two datasets from CSV files.

    Returns:
    - pd.DataFrame: The first dataset (df_01_09).
    - pd.DataFrame: The second dataset (df_02_09).
    """

    print('- Reading dataset..')
    df_01_09 = pd.read_csv(rf"../{DATA_FOLDER}/1_FINAL_DATASET_AFTER_KRIGING_INTERPOLATION.csv")
    df_02_09 = pd.read_csv(rf"../{DATA_FOLDER}/DATA_02-09-23.csv")
    return df_01_09, df_02_09

def data_preprocess(df):
    """
    Perform data preprocessing by filling missing values with the mean.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The preprocessed dataframe.
    """
    # Fill missing values with the mean
    new_df = df.fillna(df.mean())
    return new_df

def data_split(df, geo_info=False, test=False):
    """
    Split the dataset into features and target variables, and further split them into training and testing sets.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - geo_info (bool): If True, includes geo information(X,Y) in the features.

    Returns:
    - pd.DataFrame: Training features.
    - pd.DataFrame: Testing features.
    - pd.Series: Training target.
    - pd.Series: Testing target.
    """

    if test:
        df = df[:50]

    if not geo_info:
        features = df.drop(['id','EC','X','Y'], axis=1)
    else: 
        features = df.drop(['id','EC'], axis=1)
    target = df['EC']

    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
