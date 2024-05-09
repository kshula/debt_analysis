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
   git clone https://github.com/your/repository.git
   cd repository
