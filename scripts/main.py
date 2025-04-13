import pandas as pd
import numpy as np




from data_loader import DataLoader
from gru_model import GRUPredictor
from gru_processor import GRUPreprocessor
from neural_prophet_model import NeuralProphetPredictor
from plotter import Plotter


if __name__ == "__main__":
    # Load and preprocess data
    loader = DataLoader()
    df = loader.fetch_data()

    pre = GRUPreprocessor()
    x, y, scaled = pre.transform(df)
    x_reshaped = pre.reshape_input(x)

    # GRU
    gru_model = GRUPredictor(input_shape=(x_reshaped.shape[1], 1))
    gru_model.train(x_reshaped, y)
    gru_model.save("./models/gru_model.keras")
    last_sequence = x[-1]
    predictions_scaled = gru_model.predict_next_days(last_sequence, days=21)
    gru_predictions = pre.inverse_transform(predictions_scaled)

    # NeuralProphet
    prophet_model = NeuralProphetPredictor()
    prophet_result = prophet_model.train_predict(df, future_days=21)

    # Plot
    future_dates = pd.date_range(
        start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=21, freq="B"
    )
    Plotter.plot_predictions(
        history_dates=df["Date"],
        history_values=df["Close"],
        future_dates=future_dates,
        gru_preds=gru_predictions,
        prophet_preds=prophet_result["yhat1"].values,
    )
