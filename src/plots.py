#################################################################
#   Author      : Ioannis Kontogiorgakis                        #
#   File        : plots.py                                      #
#   Description : This file is responsible for producing the    #
#                 plots for model visualization and evaluation. # 
#################################################################


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_pred(model_name, y_pred, y_test,rmse, mse, r2):
  """
  Plot the actual vs. predicted values and display evaluation metrics.

  Parameters:
  - model_name (str): The name of the regression model.
  - y_pred (Series): The predicted target values.
  - y_test (Series): The true target values.
  - rmse (float64): The root mean squared error.
  - mse (float64): The mean squared error.
  - r2 (float64): The R-squared score.

  Returns:
  None
  """
  # Create a DataFrame with actual and predicted values
  results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

  # Plot the predicted vs. actual values
  plt.figure(figsize=(12, 4))
  sns.scatterplot(x='Actual', y='Predicted', data=results_df, color='limegreen')
  plt.title(f'{model_name} : Actual vs. Predicted EC Values')
  plt.xlabel('Actual Values')
  plt.ylabel('Predicted Values')

  # Add a diagonal line for reference
  plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
  plt.text(0.1, 0.8, f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR-squared: {r2:.4f}', transform=plt.gca().transAxes)


  plt.legend(loc='lower right')
  plt.show()



def residuals_test(y_pred, y_test):
  """
  Plot residuals against predicted values and the distribution of residuals.

  Parameters:
  - y_pred (Series): The predicted target values.
  - y_test (Series): The true target values.

  Returns:
  None
  """
  # Calculate residuals
  residuals = y_test - y_pred

  # Create a figure with two subplots side by side
  fig, axs = plt.subplots(1, 2, figsize=(10, 4))

  # Plot residuals against predicted values in the first subplot
  sns.scatterplot(x=y_pred, y=residuals, ax=axs[0])
  axs[0].set_title('Residual Plot')
  axs[0].set_xlabel('Predicted Values')
  axs[0].set_ylabel('Residuals')
  axs[0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual Line')
  axs[0].legend()

  # Plot the distribution of residuals in the second subplot
  sns.histplot(residuals, kde=True, ax=axs[1])
  axs[1].set_title('Distribution of Residuals')
  axs[1].set_xlabel('Residuals')
  axs[1].set_ylabel('Frequency')

  # Adjust layout to prevent overlap
  plt.tight_layout()
  plt.show()