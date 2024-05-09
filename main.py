import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def plot_predictions(predictions, future_periods):
    fig = go.Figure()

    for model_name, preds in predictions.items():
        fig.add_trace(go.Scatter(
            x=list(future_periods),
            y=preds,
            mode='lines+markers',
            name=model_name
        ))

    fig.update_layout(
        title=f"Debt Service Predictions for Next {len(future_periods)} Periods",
        xaxis_title="Period",
        yaxis_title="Debt Service"
    )

    return fig

# Load data and process numeric columns
@st.cache_data  # Cache data loading for improved performance
def load_debt_data(filename):
    df = pd.read_csv(filename)
    # Process numeric columns
    df['Debt Stock***'] = df['Debt Stock***'].str.replace(',', '').astype(float)
    df['Interest Arrears'] = df['Interest Arrears'].str.replace(',', '').astype(float)
    df.fillna(0, inplace=True)  # Replace NaN with zero
    return df


def load_data(filename):
    return pd.read_csv(filename)

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
        results[name] = {'Accuracy': accuracy, 'R2 Score': r2}

    return results

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "Model Accuracy", "Predictions", "Debt Analysis"))

    if page == "Home":
        st.title("Home")
        st.write("Welcome to Debt Service Forecasting and Analysis!")

        # Load data
        data = pd.read_csv('data/data_1.csv')

        # Check if 'Year' column exists in the dataset
        if 'year' not in data.columns:
            st.write("Error: 'Year' column not found in the dataset.")
            return

        # Plot debt service over time
        fig = px.line(data, x='year', y='Debt_service', title='Debt Service Over Time')
        fig.update_xaxes(title='year')
        fig.update_yaxes(title='Debt Service')

        # Display the plot
        st.plotly_chart(fig)

    elif page == "Model Accuracy":
        st.title("Analysis")
        st.subheader("Model Evaluation Results:")

        # Load data
        data = load_data('data/data_1.csv')

        # Splitting features and target variable
        X = data.drop('Debt_service', axis=1)
        y = data['Debt_service']

        # Splitting data into train and test sets (80:20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate models
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

        # Display evaluation results
        for name, scores in results.items():
            st.write(f"Model: {name}")
            st.write(f"Accuracy: {scores['Accuracy']:.4f}")
            st.write(f"R2 Score: {scores['R2 Score']:.4f}")
            st.write("=" * 30)

    elif page == "Predictions":
        st.title("Predictions")
        st.write("Select Forecast Periods:")
        periods = st.slider("Periods", min_value=1, max_value=7, value=3)

        # Load data
        data = load_data('data/data_1.csv')

        # Splitting features and target variable
        X = data.drop('Debt_service', axis=1)
        y = data['Debt_service']

        # Train models on entire dataset
        models = {
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor()
        }

        # Make predictions for selected periods
        future_periods = list(range(1, periods + 1))
        predictions = {}

        for name, model in models.items():
            # Train model on entire dataset
            model.fit(X, y)

            # Generate predictions for the specified future periods
            future_predictions = []
            current_data_point = X.tail(1).copy()

            for _ in future_periods:
                # Make prediction for the next period using the current data point
                prediction = model.predict(current_data_point)[0]
                future_predictions.append(prediction)

                # Update data point for the next period if needed (e.g., shift time step)
                # Example: current_data_point['column_name'] = new_value

            # Store predictions for the model
            predictions[name] = future_predictions

        # Plot debt service predictions over time
        if predictions:
            st.subheader(f"Debt Service Predictions for Next {periods} Periods:")
            fig = plot_predictions(predictions, future_periods)
            st.plotly_chart(fig)
    
    elif page == "Debt Analysis":
        df = load_debt_data('data/data_debt.csv')  # Use '/' instead of '\\'

        # Select features and target variable for k-NN
        X = df[['Debt Stock***', 'Interest Arrears']]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit k-NN model to find neighbors
        n_neighbors = 5  # Number of neighbors for k-NN
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(X_scaled)

        # Generate a Plotly graph of k-NN neighbors
        st.subheader(f"Plot of {n_neighbors} Nearest Neighbors")
        query_point = np.array([[np.mean(X['Debt Stock***']), np.mean(X['Interest Arrears'])]])
        query_point_scaled = scaler.transform(query_point)

        # Find nearest neighbors of the query point
        distances, indices = knn.kneighbors(query_point_scaled)

        # Create a scatter plot of the data and highlight the nearest neighbors
        fig = go.Figure()

        # Scatter plot of all data points
        fig.add_trace(go.Scatter(
            x=X['Debt Stock***'],
            y=X['Interest Arrears'],
            mode='markers',
            marker=dict(color='blue', size=8),
            text=df['Lender'],  # Set lenders' names as tooltips
            name='All Data Points'
        ))

        # Highlight nearest neighbors
        nearest_neighbors_x = X['Debt Stock***'].iloc[indices[0]].values
        nearest_neighbors_y = X['Interest Arrears'].iloc[indices[0]].values
        nearest_neighbors_lenders = df['Lender'].iloc[indices[0]].values
        fig.add_trace(go.Scatter(
            x=nearest_neighbors_x,
            y=nearest_neighbors_y,
            mode='markers',
            marker=dict(color='red', size=12),
            text=nearest_neighbors_lenders,  # Set nearest neighbors' lenders as tooltips
            name='Nearest Neighbors'
        ))

        # Highlight query point
        fig.add_trace(go.Scatter(
            x=[query_point[0][0]],
            y=[query_point[0][1]],
            mode='markers',
            marker=dict(color='green', size=15),
            text=['Query Point'],  # Set query point label
            name='Query Point'
        ))

        # Update layout and display the Plotly graph
        fig.update_layout(
            title=f'{n_neighbors} Nearest Neighbors of Query Point',
            xaxis_title='Debt Stock',
            yaxis_title='Interest Arrears',
            hovermode='closest'  # Display tooltip for nearest point on hover
        )

        st.plotly_chart(fig)

        # Display the cleaned DataFrame on the homepage
        st.subheader("Cleaned Data")
        st.write(df)


        

# Run the Streamlit app
if __name__ == "__main__":
    main()
