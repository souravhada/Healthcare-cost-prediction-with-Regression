# Healthcare Cost Prediction with Regression

## Project Overview
This project focuses on predicting healthcare costs using a regression model. By analyzing a dataset containing various features such as age, sex, BMI, number of children, smoker status, and region, we aim to predict individual medical costs billed by health insurance. The model is built using TensorFlow and Keras, showcasing the power of neural networks in handling regression tasks.

## Features and Labels
The dataset includes the following features:
- Age: Age of the primary beneficiary
- Sex: Insurance contractor gender, female or male
- BMI: Body mass index
- Children: Number of children covered by health insurance / Number of dependents
- Smoker: Smoking status
- Region: The beneficiary's residential area in the US

The label or target variable is:
- Expenses: Individual medical costs billed by health insurance

## Requirements
- Python 3.8 or later
- TensorFlow 2.x
- Pandas
- Numpy
- Matplotlib (for plotting)

## Setup Instructions
1. Clone the repository to your local machine.
   ```bash
   git clone <repository-url>

## Model Architecture
The final model is a neural network that incorporates regularization to mitigate overfitting, structured as follows:
- An input normalization layer that scales the features for optimal neural network processing.
- Two dense hidden layers, each with 64 units and ReLU activation functions, with L2 regularization applied to mitigate overfitting by penalizing large weights.
- A final output layer with a single unit for the regression output, predicting healthcare costs.

To further prevent overfitting, early stopping monitors validation loss and halts training if no improvement is observed after a specified number of epochs, restoring the best model weights observed during training.

## Compile and Training Details
- **Optimizer**: Adam, chosen for its adaptive learning rate capabilities, making it suitable for this regression task.
- **Loss Function**: Mean Absolute Error (MAE), which provides a straightforward interpretation in terms of average error magnitude.
- **Metrics**: MAE and Mean Squared Error (MSE), to evaluate the model's performance and error distribution.

## Results
After training, the model achieves a Mean Absolute Error (MAE) on the test set. This performance indicates the model's capability to predict healthcare costs with a reasonable degree of accuracy. Ongoing tuning and experimentation may lead to further improvements.

## Acknowledgments
This project was developed as part of the FreeCodeCamp's machine learning curriculum. The dataset used for this project is sourced from the "Machine Learning with Python" course by FreeCodeCamp.
