# Import necessary libraries
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Import models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


# Import the NN libraries
import tensorflow as tf
from tensorflow import keras
from keras import Sequential,layers
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# for plots
import matplotlib.pyplot as plt
import seaborn as sns