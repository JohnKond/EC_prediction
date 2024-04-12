#####################################################################
#   Author      : Ioannis Kontogiorgakis                            #
#   File        : storeLoadUtils.py                                 #
#   Description : This file provides utility functions for storing  # 
#                 and loading machine learning models, as well as   # 
#                 handling model parameters.                        #
#####################################################################


# Import necessary libraries
import os
import ast
import joblib


PARAMS_FOLDER = '../params'
MODELS_FOLDER = '../models'
FEATURES_FILE = '../features.txt'


def store_params(model, params):
    """
    Store model parameters in a text file.

    Parameters:
    - model (str): The model name.
    - params (str): The parameters to be stored.

    Returns:
    - None
    """
    file_path = f'{PARAMS_FOLDER}/{model}_best_params.txt'
    
    # Write the text to the file
    with open(file_path, 'w') as file:
        file.write(params)

    print(f"{model} best params has been stored in {file_path}")


def load_params(model):
    """
    Load model parameters from a text file.

    Parameters:
    - model (str): The model name.

    Returns:
    - dict or False: The loaded parameters as a dictionary if successful, False otherwise.
    """
    # Specify the file path
    file_path = f'{PARAMS_FOLDER}/{model}_best_params.txt'

    # Check if the file exists
    # if os.path.exists(file_path):
        # print(f"{model} parameters loaded.")
    # else:
        # print(f"{model} parameters do not exist.")
        # return False
    
    if not os.path.exists(file_path):
        return False


    with open(file_path, 'r') as file:
        file_content = file.read()

        try:
            # Convert string to dictionary using ast.literal_eval
            params_dict = ast.literal_eval(file_content)
            if not isinstance(params_dict, dict):
                print(f"The file {file_path} does not contain a valid dictionary.")
        except (SyntaxError, ValueError):
            print(f"The file {file_path} contains invalid syntax.")
            return False

    return params_dict


def store_model(model, model_name):
    """
    Save a machine learning model to a pickle file.

    Parameters:
    - model: The model to be saved.
    - model_name (str): The name of the model.

    Returns:
    - None
    """
    file_path = f'{MODELS_FOLDER}/{model_name}_model.pkl'

    try:
        # Save the model to the specified file path
        joblib.dump(model, file_path)
        print(f"Model saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(model_name):
    """
    Load a machine learning model from a pickle file.

    Parameters:
    - model_name (str): The name of the model.

    Returns:
    - The loaded model or None if unsuccessful.
    """
    file_path = f'{MODELS_FOLDER}/{model_name}_model.pkl'


    if not os.path.exists(file_path):
        return False
    else:

        try:
            # Load the model from the specified file path
            loaded_model = joblib.load(file_path)
            print(f"Model loaded successfully from {file_path}")
            return loaded_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        

def store_features(features):
    # Write features to a text file
    with open(FEATURES_FILE, "w") as file:
        for feature in features:
            file.write(feature + "\n")

    print(f"- Features have been sorted accoring their significance and stored {FEATURES_FILE}.")
    return


def load_features():
    # Read features from the text file into a list
    with open(FEATURES_FILE, "r") as file:
        features_list = [line.strip() for line in file]
    return features_list
