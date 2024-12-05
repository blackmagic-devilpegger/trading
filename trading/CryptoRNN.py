import krakenex
import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class CryptoRNN:
    def __init__(self, api_key, seq_length=10):
        """
        Initialisiert die Klasse mit API-Key und Sequenzlänge.
        """
        self.api = krakenex.API()
        self.api.key = api_key
        self.seq_length = seq_length        # Anzahl der Zeitpunkte in einer Eingabesequenz
        self.df = None                      # Dataframe für die historischen Daten
        self.X = None                       # Eingabedaten für das Modell
        self.y = None                       # Zielwerte für das modell
        self.model = None

    def fetch_data(self, pair="XXBTZUSD", interval=60, days=30):
        """
        Ruft historische Daten von der Kraken-API ab und speichert sie in einem DataFrame.
        """
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': pair,                   #Handelspaar z.B Bitcoin zu USd
            'interval': interval,           #Zeitinterval in Minuten
            'since': int(time.time()) - 60 * 60 * 24 * days  # Startzeit (aktuelle Zeit minus days)
        }

        response = requests.get(url, params=params)
        data = response.json()

        if len(data['error']) == 0:
            #Daten in einen dataframe laden
            ohlc = data['result'][pair]
            self.df = pd.DataFrame(
                ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            )
            self.df['time'] = pd.to_datetime(self.df['time'], unit='s')

            # Konvertiere relevante Spalten in numerische Werte und entferne fehlerhafte Zeilen
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df = self.df.dropna()  # Entferne Zeilen mit NaN-Werten

            print("Daten erfolgreich abgerufen:")
            print(self.df.head())
        else:
            raise Exception(f"API-Fehler: {data['error']}")

    def prepare_data(self, validation_split=0.2):
        """
        Bereitet die Daten für das RNN vor: Normalisierung und Sequenzierung.
        """
        if self.df is None:
            raise Exception("Keine Daten verfügbar. Rufe zuerst fetch_data() auf.")

        # Relevante Spalten auswählen und normalisieren
        ohlc_data = self.df[['open', 'high', 'low', 'close']].astype(float)
        normalized_data = (ohlc_data - ohlc_data.mean()) / ohlc_data.std()

        # Sequenzen und Zielwerte erstellen
        X, y = [], []
        for i in range(len(normalized_data) - self.seq_length):
            X.append(normalized_data.iloc[i:i+self.seq_length].values)
            y.append(normalized_data.iloc[i+self.seq_length]['close'])

        # Konvertiere die Listen in numpy-Arrays und dann in Tensoren
        X = np.array(X)  # Konvertiere die Liste in ein numpy-Array
        y = np.array(y)  # Konvertiere die Liste in ein numpy-Array

        # Split in Trainings- und Validierungsdaten
        split_idx = int(len(X) * (1 - validation_split))
        self.X_train, self.X_val = torch.tensor(X[:split_idx], dtype=torch.float32), torch.tensor(X[split_idx:],
                                                                                                  dtype=torch.float32)
        self.y_train, self.y_val = torch.tensor(y[:split_idx], dtype=torch.float32), torch.tensor(y[split_idx:],
                                                                                                  dtype=torch.float32)

        print(f"Trainingsdaten: {len(self.X_train)} Sequenzen, Validierungsdaten: {len(self.X_val)} Sequenzen")

    def create_model(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Erstellt das RNN-Modell.
        """
        class RNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(RNNModel, self).__init__()
                self.num_layers = num_layers  # Anzahl der Schichten speichern
                self.hidden_size = hidden_size  # Versteckte Einheiten speichern
                self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)        #RNN-Layer
                self.fc = nn.Linear(hidden_size, output_size)                       #Fully Connected Layer

            def forward(self, x):
                h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)          #Initialisiere den versteckten Zustand
                out, _ = self.rnn(x, h_0)                                           #RNN Berechnung
                out = self.fc(out[:, -1, :])                                        #Ausgabe des letzten Zeitpunkts
                return out

        self.model = RNNModel(input_size, hidden_size, output_size)
        print("Modell erstellt.")

    def train_model(self, num_epochs=50, batch_size=32, learning_rate=0.0001, patience=10):
        """
        Trainiert das RNN-Modell mit Early Stopping .
        """
        if self.model is None:
            raise Exception("Kein Modell verfügbar. Rufe create_model() auf.")

        criterion = nn.MSELoss()                                                    #Verlustfunktion (Mean-Squared Error)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)           #Adam-Optimierer

        # Listen zur Speicherung der Verluste
        self.train_loss_values = []
        self.val_loss_values = []

        # Early Stopping Variablen
        best_val_loss = float("inf")  # Start mit einem sehr hohen Verlustwert
        epochs_no_improve = 0  # Zählt die Epochen ohne Verbesserung

        for epoch in range(num_epochs):
            epoch_loss = 0

            #Training
            self.model.train()
            for i in range(0, len(self.X_train), batch_size):
                X_batch = self.X_train[i:i+batch_size]
                y_batch = self.y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)                        #Berechne den Verlust
                loss.backward()                                                     #BAckpropagation
                optimizer.step()

                # Addiere den Batch-Verlust zum epoch_loss,um den Gesamtverlust für die aktuelle Epoche zu berechnen
                epoch_loss += loss.item()

            # Validierung
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(self.X_val), batch_size):
                    X_val_batch = self.X_val[i:i + batch_size]
                    y_val_batch = self.y_val[i:i + batch_size]
                    outputs = self.model(X_val_batch)
                    val_loss += criterion(outputs.squeeze(), y_val_batch).item()

            # Speichere die Verluste
            train_loss_epoch = epoch_loss / len(self.X_train)
            val_loss_epoch = val_loss / len(self.X_val)
            self.train_loss_values.append(epoch_loss / len(self.X_train))
            self.val_loss_values.append(val_loss / len(self.X_val))

            # Ausgabe der Verluste
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss / len(self.X_train):.6f}, Validation Loss: {val_loss / len(self.X_val):.6f}")

            # Überprüfung für Early Stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                epochs_no_improve = 0  # Zurücksetzen, da es eine Verbesserung gab
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(
                    f"Frühes Stoppen nach {epoch + 1} Epochen. Validation Loss hat sich {patience} Epochen nicht verbessert.")
                break


    def predict(self, input_data):
        """
        Gibt Vorhersagen für Eingabedaten zurück.
        """
        if self.model is None:
            raise Exception("Kein Modell verfügbar. Rufe create_model() auf.")

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_data)
        return predictions


if __name__ == "__main__":
    # Instanziiere die Klasse mit deinem API-Key
    crypto_rnn = CryptoRNN(api_key='w3if4ZjPEKdgCVsj7J/KVRgkSKhAhYBcJJrrp8gXTfrRdlylAVafK85F')

    # 1. Daten abrufen
    crypto_rnn.fetch_data()

    # 2. Daten vorbereiten
    crypto_rnn.prepare_data()

    # 3. Modell erstellen
    crypto_rnn.create_model(input_size=4, hidden_size=50, output_size=1)

    # 4. Modell trainieren
    crypto_rnn.train_model(num_epochs=50, batch_size=32, learning_rate=0.0005)

    ####Visualisierung des Trainings-und Validierungsverlustes
    import matplotlib.pyplot as plt

    # Beispiel: Verlustwerte während des Trainings
    train_loss_values = crypto_rnn.train_loss_values
    val_loss_values =  crypto_rnn.val_loss_values

    # Plot
    plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label="Train Loss")
    plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

    # 5. Vorhersagen treffen
    test_input = crypto_rnn.X_val[:5]  # Beispiel: Erste 5 Sequenzen
    predictions = crypto_rnn.predict(test_input)
    print("Vorhersagen:", predictions)

    # Rücknormalisierung der Vorhersagen
    # Mittelwert und Standardabweichung der Originaldaten berechnen
    ohlc_mean = crypto_rnn.df[['open', 'high', 'low', 'close']].mean()
    ohlc_std = crypto_rnn.df[['open', 'high', 'low', 'close']].std()


    # Rücknormalisieren der Vorhersagen
    predictions_original_scale = predictions * ohlc_std['close'] + ohlc_mean['close']
    predictions_original_scale = predictions_original_scale.detach().numpy()  # Konvertierung in numpy-Array
    print("Rücknormierte Vorhersagen (im Originalmaßstab):", predictions_original_scale)

    # Mean Absolute Deviation for predictions
    mad_predictions = np.mean(
        np.abs(predictions_original_scale - crypto_rnn.df['close'].iloc[-len(predictions):].values))
    print(f"Mean Absolute Deviation (MAD) of Predictions: {mad_predictions:.4f} USD")

    # Mean Absolute Deviation for average price
    average_price = crypto_rnn.df['close'].mean()
    mad_average = np.mean(np.abs(crypto_rnn.df['close'] - average_price))
    print(f"Mean Absolute Deviation (MAD) of Average Price: {mad_average:.4f} USD")

    # Anzahl der Datenpunkte, die wir visualisieren wollen (angenommen, wir haben mindestens 50 Datenpunkte)
    n = min(50, len(predictions_original_scale))

    # Bereite die tatsächlichen Werte (Validierungsdaten) und die Vorhersagen vor
    actual_prices = crypto_rnn.df['close'].iloc[-(len(crypto_rnn.y_val)):].values[:n]
    predicted_prices = predictions_original_scale[:n]

    # Visualisierung der Vorhersagen
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Actual Prices", color='blue', alpha=0.7)
    plt.plot(predicted_prices, label="Predicted Prices", color='red', linestyle='--', alpha=0.7)
    plt.title("Bitcoin Price Prediction")
    plt.xlabel("Time Steps")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()


    # Handelsstrategien
    # Einfache Preisprognose
    current_price = crypto_rnn.df.iloc[-1]['close']  # Aktueller Schlusskurs
    predicted_price = predictions_original_scale[0].item()  # Vorhergesagter nächster Schlusskurs

    margin = 0.02 * current_price  # Sicherheitsmarge von 2,0 %
    if predicted_price > current_price + margin:
        print("Strategie-Einfache_Preisprognose: Kaufen (Preis wird steigen).")
    elif predicted_price < current_price - margin:
        print("Strategie-Einfache_Preisprognose: Verkaufen (Preis wird fallen).")
    else:
        print("Strategie-Einfache_Preisprognose: Keine Aktion (Unsicherheit zu groß).")

    # Threshold-basierter Handel
    #threshold = 0.02  # 2% Schwelle
    #elif (current_price - predicted_price) / current_price > threshold:
        #print("Strategie-Treshold: Verkaufen (hohe Wahrscheinlichkeit, dass der Preis fällt).")
    #else:
        #print("Strategie-Treshold: Keine Aktion (Unklarer Markttrend).")