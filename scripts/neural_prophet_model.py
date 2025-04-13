import logging
from scripts.data_loader import DataLoader
from neuralprophet import NeuralProphet

class NeuralProphetPredictor:
    def __init__(self):
        self.model = NeuralProphet()

    def train_predict(self, df, future_days=21):
        df_prophet = df.rename(columns={"Date": "ds", "Close": "y"})
        self.model.fit(df_prophet, freq="B")
        future = self.model.make_future_dataframe(df_prophet, periods=future_days)
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat1"]].tail(future_days)

if __name__ == "__main__":    

    loader = DataLoader()
    df = loader.fetch_data()

    predictor = NeuralProphetPredictor()
    forecast = predictor.train_predict(df, future_days=21)
    logging.info(forecast)
