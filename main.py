import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load data and process numeric columns
@st.cache_data  # Cache data loading for improved performance
def load_data(filename):
    df = pd.read_csv(filename)
    # Process numeric columns
    df['Debt Stock***'] = df['Debt Stock***'].str.replace(',', '').astype(float)
    df['Interest Arrears'] = df['Interest Arrears'].str.replace(',', '').astype(float)
    df.fillna(0, inplace=True)  # Replace NaN with zero
    return df

# Main function to run the Streamlit app
def main():
    st.title("Debt Stock and Interest Arrears Analysis")

    # Load data
    df = load_data('data/data_debt.csv')  # Use '/' instead of '\\'

    # Display the cleaned DataFrame on the homepage
    st.subheader("Cleaned Data")
    st.write(df)

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

# Run the Streamlit app
if __name__ == "__main__":
    main()
