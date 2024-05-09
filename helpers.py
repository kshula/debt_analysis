import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go

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
        results[name] = {'Accuracy': accuracy, 'R2 Score': r2, 'Predictions': y_pred}

    return results


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