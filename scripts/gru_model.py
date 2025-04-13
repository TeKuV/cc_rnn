import os
import numpy as np
from data_loader import DataLoader
from gru_processor import GRUPreprocessor
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Sequential, load_model


class GRUPredictor:
    def __init__(self, input_shape=None, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = Sequential()
            self.model.add(
                GRU(units=50, return_sequences=False, input_shape=input_shape)
            )
            self.model.add(Dense(1))
            self.model.compile(optimizer="adam", loss="mean_squared_error")

    def train(
        self,
        x_train,
        y_train,
        epochs=20,
        batch_size=32,
        return_history=False,
        progress_callback=None,
    ):
        from tensorflow.keras.callbacks import Callback

        class StreamlitProgressCallback(Callback):
            def __init__(self, total_epochs, bar):
                self.total_epochs = total_epochs
                self.bar = bar

            def on_epoch_end(self, epoch, logs=None):
                progress = int((epoch + 1) / self.total_epochs * 100)
                self.bar.progress(
                    progress,
                    text=f"Epoch {epoch+1}/{self.total_epochs} - Loss: {logs['loss']:.4f}",
                )

        callbacks = []
        if progress_callback:
            callbacks.append(StreamlitProgressCallback(epochs, progress_callback))

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        return history if return_history else None

    def predict_next_days(self, last_sequence, days=21):
        predictions = []
        current_sequence = last_sequence
        for _ in range(days):
            prediction = self.model.predict(
                current_sequence.reshape(1, -1, 1), verbose=0
            )
            predictions.append(prediction[0, 0])
            current_sequence = np.append(current_sequence[1:], prediction).reshape(
                -1, 1
            )
        return np.array(predictions).reshape(-1, 1)

    def save(self, path="./models/gru_model.keras"):
        self.model.save(path)


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.fetch_data()

    pre = GRUPreprocessor()
    x, y, scaled = pre.transform(df)
    x_reshaped = pre.reshape_input(x)

    gru_model = GRUPredictor(input_shape=(x_reshaped.shape[1], 1))
    gru_model.train(x_reshaped, y)
    gru_model.save("./models/gru_model.keras")
    last_sequence = x[-1]
    predictions_scaled = gru_model.predict_next_days(last_sequence, days=21)
    gru_predictions = pre.inverse_transform(predictions_scaled)
