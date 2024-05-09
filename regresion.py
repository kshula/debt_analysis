import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# Load the dataset
@st.cache_data  # Cache data loading for improved performance
def load_data(filename):
    return pd.read_csv(filename)

# Function to train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}
    models = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'Accuracy': accuracy, 'R2 Score': r2, 'Predictions': y_pred}

    return results

# Function to plot debt service predictions over time
def plot_predictions_over_time(predictions, periods):
    fig = go.Figure()

    for name, preds in predictions.items():
        fig.add_trace(go.Scatter(
            x=periods,
            y=preds,
            mode='lines',
            name=name
        ))

    fig.update_layout(
        title="Debt Service Predictions Over Time",
        xaxis_title="Period",
        yaxis_title="Debt Service"
    )

    return fig

# Main function to run the Streamlit app
def main():
    st.title("Debt Service Forecasting and Analysis")

    # Load data
    data = load_data('data\\data_1.csv')

    # Splitting features and target variable
    X = data.drop('Debt_service', axis=1)
    y = data['Debt_service']

    # Splitting data into train and test sets (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Display results
    st.subheader("Model Evaluation Results:")
    for name, scores in results.items():
        st.write(f"Model: {name}")
        st.write(f"Accuracy: {scores['Accuracy']:.4f}")
        st.write(f"R2 Score: {scores['R2 Score']:.4f}")
        st.write("=" * 30)

    # Analysis Page
    st.sidebar.title("Analysis")

    # Allow user to select forecast periods
    periods = st.sidebar.slider("Select Forecast Periods", min_value=1, max_value=50, value=10)

    # Get debt service predictions for selected periods
    predictions = {}
    for name, model_results in results.items():
        predictions[name] = model_results['Predictions'][:periods]

    # Plot debt service predictions over time
    if predictions:
        st.subheader(f"Debt Service Predictions for Next {periods} Periods:")
        periods_range = range(1, periods + 1)
        fig = plot_predictions_over_time(predictions, periods_range)
        st.plotly_chart(fig)

# Run the Streamlit app
if __name__ == "__main__":
    main()
