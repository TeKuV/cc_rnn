import numpy as np
from sklearn.preprocessing import MinMaxScaler

class GRUPreprocessor:
    def __init__(self, sequence_length=60):
        self.scaler = MinMaxScaler()
        self.sequence_length = sequence_length

    def transform(self, data):
        scaled = self.scaler.fit_transform(data[["Close"]])
        x, y = [], []
        for i in range(self.sequence_length, len(scaled)):
            x.append(scaled[i - self.sequence_length : i, 0])
            y.append(scaled[i, 0])
        return np.array(x), np.array(y), scaled

    def reshape_input(self, x):
        return x.reshape((x.shape[0], x.shape[1], 1))

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)
