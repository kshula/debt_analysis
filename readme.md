# Debt Analysis and Prediction with Machine learning

This project is a Streamlit web application designed for debt service analysis and prediction. It includes features for evaluating lender classification using K-Nearest Neighbors (KNN) and predicting debt service using Random Forest and Gradient Boosting models.

## Features

### 1. Lender Classification with KNN

The application includes a functionality to perform lender classification based on debt characteristics using the K-Nearest Neighbors (KNN) algorithm. The key steps involved are:

- Loading and preprocessing the debt data.
- Applying KNN to classify lenders based on debt stock and interest arrears.
- Visualizing the classification results using interactive plots.

### 2. Debt Service Prediction

The application offers debt service prediction using two machine learning models: Random Forest and Gradient Boosting. The prediction process involves:

- Splitting the dataset into training and testing sets.
- Training the models on the training data.
- Evaluating model performance using accuracy and R2 score.
- Generating future predictions for debt service over specified periods.

## How to Use

1. **Installation**

   Ensure you have Python installed. Clone this repository and navigate to the project directory.

   ```bash
   git clone https://github.com/kshula/debt_analysis.git
   cd debt
   ```


    Install the required Python packages using pip and the provided requirements.txt file.

```bash
Copy code
pip install -r requirements.txt
```
## Running the Application
Start the Streamlit web app by running the following command in your terminal.
```bash
Copy code
streamlit run main.py
```
This will launch the web application in your default web browser.
## Navigation
Home: Displays an overview of debt service over time.
Model Accuracy: Evaluates model performance on debt service data.
Predictions: Generates future predictions for debt service using selected models.
Debt Analysis: Machine learning KNN Analysis

## File Structure
main.py: Main Python script containing Streamlit application code.
data/: Directory containing dataset files used by the application.
requirements.txt: List of Python packages required for the project.

## Contributors
Kampamba Shula