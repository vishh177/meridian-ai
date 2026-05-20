import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# ─── 1. THE NEURAL NETWORK ARCHITECTURE ───
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ─── 2. PREPARE THE DATA ───
def prepare_data(prices, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.FloatTensor(X_train).unsqueeze(-1)
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_train = torch.FloatTensor(y_train).unsqueeze(-1)
    y_test = torch.FloatTensor(y_test).unsqueeze(-1)

    return X_train, X_test, y_train, y_test, scaler

# ─── 3. TRAIN THE MODEL ───
def train_model(model, X_train, y_train, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.6f}")

    return model

# ─── 4. PREDICT NEXT 7 DAYS ───
def predict_next_days(model, scaler, prices, sequence_length=60, days=7):
    model.eval()
    scaled = scaler.transform(prices.reshape(-1, 1))
    last_sequence = scaled[-sequence_length:]
    predictions = []

    current_sequence = last_sequence.copy()

    with torch.no_grad():
        for _ in range(days):
            x = torch.FloatTensor(current_sequence).unsqueeze(0)
            pred = model(x)
            pred_value = pred.item()
            predictions.append(pred_value)
            current_sequence = np.append(current_sequence[1:], [[pred_value]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten().tolist()

# ─── 5. EVALUATE ACCURACY ───
def evaluate_model(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        actual = y_test.numpy()

    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(actual)

    mae = np.mean(np.abs(predictions - actual))
    direction_correct = np.sum(
        np.sign(np.diff(predictions.flatten())) == np.sign(np.diff(actual.flatten()))
    )
    direction_accuracy = direction_correct / (len(actual) - 1) * 100

    return {
        "mae": round(float(mae), 2),
        "direction_accuracy": round(float(direction_accuracy), 1)
    }

# ─── 6. MAIN FUNCTION ───
def run_lstm_prediction(ticker):
    print(f"Training LSTM for {ticker}...")

    stock = yf.Ticker(ticker)
    history = stock.history(period="2y")
    prices = history["Close"].values

    X_train, X_test, y_train, y_test, scaler = prepare_data(prices)
    model = LSTMModel()
    model = train_model(model, X_train, y_train, epochs=50)
    evaluation = evaluate_model(model, X_test, y_test, scaler)
    next_7_days = predict_next_days(model, scaler, prices)

    current_price = prices[-1]
    predicted_7d = next_7_days[-1]
    predicted_change = ((predicted_7d - current_price) / current_price) * 100

    if predicted_change > 2:
        signal = "BULLISH"
    elif predicted_change < -2:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "current_price": round(float(current_price), 2),
        "predicted_7d_price": round(float(predicted_7d), 2),
        "predicted_change_pct": round(float(predicted_change), 2),
        "signal": signal,
        "next_7_days": [round(p, 2) for p in next_7_days],
        "direction_accuracy": evaluation["direction_accuracy"],
        "mae": evaluation["mae"]
    }