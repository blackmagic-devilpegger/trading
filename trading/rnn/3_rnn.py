
import krakenex
import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# EXPERIMENT
# +DROPOUT-LAYER hinzugefügt : dropout_prob=0.1
#Erwartung: Bessere Generalisierung, Reduzierung von Überanpassung (Overfitting)
#Ergebnis: Validierungsverluste sinken gleichmäßig und sind stabil, ohne plötzliche anstiege.

# +KLASSIFIKATIONSMODELL(Random Forest) hinzugefügt  1= Preis steigt, -1 = Preis fällt, 0 = Preis bleibt konstant(mit Sicherheitsmarge)
#Erwartung: Ergänzung der regression durch klare Kauf-/Verkauf-/Keine-Aktion-Entscheidungen
#Ergebnis: Sinnvollere vorhersagen für die Handelsstrategien durch Klassifikation + Sicherheitsmarge

# +Berechnung des R²-Score hinzugefügt : während des Trainings und der Validierung
#Erwartung: Bewertungsmöglichkeit zur Genauigkeit der regression : Gibt Auskunft darüber wie gut die Vorhersagen mit den tatsächlichen Werten übereinstimmen.
#Ergebnis: ein stabiler R²-Wert der bis zu 0.957 ansteigt, zeigt hohe Modellgenauigkeit, aber train_loss-werte und valid_loss-Werte zeigen ab und zu steigeungen an.


# +REGRESSION hinzugefügt : sagt den zukünftigen schlusskurs (close price) vorher.
#Erwartung: Präzisere numerische Vorheresage treffen
#Ergebnis: Die Regression zeigte eine gute Anpassung (R²-Wert 0.957 am Ende des Trainings),
#          jedoch bleibt die MAD der Vorhersagen höher als die des Durchschnittspreises, was Verbesserungspotenzial zeigt.

# +MEAN/STD Normalisierung (z-Score) hinzugefügt
# Erwartung: Stabilisierung des Trainingsprozesses durch Standardisierung der Eingabedaten (bessere Skalierung).
# Ergebnis: Die Normalisierung führte zu stabileren Verlustwerten während des Trainings.
# Die Training- und Validation-Loss-Werte zeigen über die Epochen eine gleichmäßige Abnahme mit leichten Schwankungen, was auf eine effektive Standardisierung hinweist.

# +VISUALISIERUNG der OHLC-Daten hinzugefügt (open, high, low, close) inn einem separaten diagramm
#Erwartung: Bessere Nachvollziehbarkeit der Daten
#Ergebnis: Die OHLC-Daten zeigen Trends und Schwankungen im Marktverlauf.
# Der finale Validation Loss sank auf 0.000830, und der R²-Wert stieg auf 0.957464, was auf eine gute Anpassung des Modells hinweist.
# Die MAD der Vorhersagen (5578.1725 USD) war jedoch höher als die MAD des Durchschnittspreises (3009.8923 USD), was weitere Optimierungsmöglichkeiten aufzeigt.

# +RÜCKNORMALISIERUNG der Vohersagen hinzugefügt: Vorhersagen wurden in den ursprünglichen Maßstab zurückkonventiert.
#Erwartung: rücknormalisierung soll realistische Preiswerte liefern, die mit den tatsächlichen Daten(Schlusskursen) vergleichbar sind.
#Ergebnis: MAD der Vorhersagen (of Prediction) ist höher als MAD des Durchschnittspreises. Vorhersagen des Modells weisen größere Schwankungen auf als der Durchschnittspreis.
# Das Modell erfasst Trends , muss aber noch optimiert werden.

# => Optimierungsbdearf

class CryptoRNN:

    def __init__(self, api_key, seq_length=10):
        """
        Initialisiert die Klasse mit API-Key und Sequenzlänge.
        """
        self.api = krakenex.API()
        self.api.key = api_key
        self.seq_length = seq_length        # Anzahl der Zeitpunkte in einer Eingabesequenz
        self.df = None                      # Speichert die OHLC-Daten
        self.X = None                       # Eingabedaten für das Modell
        self.y = None                       # Zielwerte für das modell
        self.regression_model = None
        self.model = None

    def fetch_data(self, pair="XXBTZUSD", interval=60, days=30):
        """
        Ruft historische Daten von der Kraken-API ab und speichert sie in einem DataFrame.
        """
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': pair,                                                   # Handelspaar z.B Bitcoin zu USd
            'interval': interval,            # Zeitinterval in Minuten
            'since': int(time.time()) - 60 * 60 * 24 * days  # Startzeit (aktuelle Zeit minus days)
        }

        response = requests.get(url, params=params)
        data = response.json()

        if len(data['error']) == 0:
            # Daten in einen dataframe laden
            ohlc = data['result'][pair]
            self.df = pd.DataFrame(
                ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            )
            self.df['time'] = pd.to_datetime(self.df['time'], unit='s')

            # Konvertiere relevante Spalten in numerische Werte und entferne fehlerhafte Zeilen
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df = self.df.dropna()                                                  # Entferne Zeilen mit NaN-Werten

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

        # Relevante Spalten auswählen
        ohlc_data = self.df[['open', 'high', 'low', 'close']].astype(float)
        # und normalisieren
        normalized_data = (ohlc_data - ohlc_data.mean()) / ohlc_data.std()

        # Initialisierung der Zielvariablen und der Sicherheitsmarge
        X, y_reg, y_class = [], [], []
        margin = 0.02                                                                       # Sicherheitsmarge von 2 %

        # Sequenzen und Zielwerte erstellen
        for i in range(len(normalized_data) - self.seq_length -1):
            X.append(normalized_data.iloc[i:i+self.seq_length].values)
            y_reg.append(normalized_data.iloc[i+self.seq_length]['close'])

            # Klassifikationslabel erstellen
            price_diff = self.df['close'].iloc[i + self.seq_length + 1] - self.df['close'].iloc[i + self.seq_length]
            if price_diff > margin:
                y_class.append(1)  # Steigt
            elif price_diff < -margin:
                y_class.append(-1)  # Fällt
            else:
                y_class.append(0)  # Konstant

        # Konvertiere die Listen in numpy-Arrays und dann in Tensoren
        X = np.array(X)                                                     # Konvertiere die Liste in ein numpy-Array
        y_reg = np.array(y_reg)                                             # Konvertiere die Liste in ein numpy-Array
        y_class = np.array(y_class)


        # Split in Trainings- und Validierungsdaten
        split_idx = int(len(X) * (1 - validation_split))
        self.X_train, self.X_val = torch.tensor(X[:split_idx], dtype=torch.float32), torch.tensor(X[split_idx:],
                                                                                                  dtype=torch.float32)
        self.y_train_reg, self.y_val_reg = torch.tensor(y_reg[:split_idx], dtype=torch.float32), torch.tensor(y_reg[split_idx:],
                                                                                                  dtype=torch.float32)

        self.X_train_class = X[:split_idx]
        self.X_val_class = X[split_idx:]
        self.y_train_class = y_class[:split_idx]
        self.y_val_class = y_class[split_idx:]

    def create_model(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Erstellt das RNN-Modell.
        """
        class RNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.1):
                super(RNNModel, self).__init__()
                self.num_layers = num_layers  # Anzahl der Schichten speichern
                self.hidden_size = hidden_size  # Versteckte Einheiten speichern
                self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)        #RNN-Layer
                self.dropout = nn.Dropout(dropout_prob)                                           #Dropout-Layer für Regularisierung
                self.fc = nn.Linear(hidden_size, output_size)                                     #Fully Connected Layer

            def forward(self, x):
                h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)     #Initialisiere den versteckten Zustand
                out, _ = self.rnn(x, h_0)                                           #RNN Berechnung
                out = out[:, -1, :]                                                 # Zugriff auf den letzten Zeitschritt (Batchgröße, Hidden_Size)
                out = self.dropout(out)                                             #Dropout nach dem letzten Zeit-Schritt
                out = self.fc(out)                                                  #Ausgabe des letzten Zeitpunkts
                return out

        self.model = RNNModel(input_size, hidden_size, output_size)
        print("Modell mit Dropout erstellt.")

    def train_model(self, num_epochs=50, batch_size=32, learning_rate=0.0001, patience=10):
        """
        Trainiert das RNN-Modell mit Early Stopping.

        """
        if self.model is None:
            raise Exception("Kein Modell verfügbar. Rufe create_model() auf.")

        criterion = nn.MSELoss()                                                    #Verlustfunktion (Mean-Squared Error)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)           #Adam-Optimierer

        # Listen zur Speicherung der Verluste
        self.train_loss_values = []
        self.val_loss_values = []

        #Early Stopping Variablen
        best_val_loss = float("inf")  # Start mit einem sehr hohen Verlustwert
        epochs_no_improve = 0  # Zählt die Epochen ohne Verbesserung

        for epoch in range(num_epochs):
            epoch_loss = 0

            #Training
            self.model.train()
            for i in range(0, len(self.X_train), batch_size):
                X_batch = self.X_train[i:i+batch_size]
                y_batch = self.y_train_reg[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)                        #Berechne den Verlust
                loss.backward()                                                     #Backpropagation
                optimizer.step()

                # Addiere den Batch-Verlust zum epoch_loss,um den Gesamtverlust für die aktuelle Epoche zu berechnen
                epoch_loss += loss.item()

            # Validierung
            self.model.eval()
            val_loss = 0
            y_val_pred = []
            with torch.no_grad():
                for i in range(0, len(self.X_val), batch_size):
                    X_val_batch = self.X_val[i:i + batch_size]
                    y_val_batch = self.y_val_reg[i:i + batch_size]
                    outputs = self.model(X_val_batch)
                    val_loss += criterion(outputs.squeeze(), y_val_batch).item()
                    y_val_pred.extend(outputs.squeeze().tolist())

            # Speichere die Verluste
            train_loss_epoch = epoch_loss / len(self.X_train)
            val_loss_epoch = val_loss / len(self.X_val)
            self.train_loss_values.append(epoch_loss / len(self.X_train))
            self.val_loss_values.append(val_loss / len(self.X_val))

            # Berechne R²-Score
            y_val_actual = self.y_val_reg.numpy()
            r2 = r2_score(y_val_actual, y_val_pred)

            # Ausgabe der Verluste und R²-Score
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss / len(self.X_train):.6f}, Validation Loss: {val_loss / len(self.X_val):.6f}, R²: {r2:.6f}")

            # Überprüfung für Early Stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                epochs_no_improve = 0  # Zurücksetzen, da es eine Verbesserung gab
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(f"Frühes Stoppen nach {epoch + 1} Epochen. Validation Loss hat sich {patience} Epochen nicht verbessert.")
                break





    def train_classification_model(self):
        """
        Trainiert ein Klassifikationsmodell.
        """
        from sklearn.ensemble import RandomForestClassifier
        self.classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classification_model.fit(self.X_train_class.reshape(len(self.X_train_class), -1), self.y_train_class)
        print("Klassifikationsmodell trainiert.")

    def predict_classification(self, input_data):
        """
        Gibt Klassifikationsvorhersagen zurück.
        """
        prediction = self.classification_model.predict(input_data.reshape(1, -1))
        return prediction[0]


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

    # Visualisierung der ursprünglichen OHLC-Daten
    plt.figure(figsize=(12, 6))
    plt.plot(crypto_rnn.df['time'], crypto_rnn.df['close'], label="Close Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("OHLC Data")
    plt.legend()
    plt.show()


    # 2. Daten vorbereiten
    crypto_rnn.prepare_data()

    # 3. Modell erstellen
    crypto_rnn.create_model(input_size=4, hidden_size=50, output_size=1)

    # 4. RegressionsModell trainieren
    crypto_rnn.train_model(num_epochs=50, batch_size=32, learning_rate=0.0005)


    # 5. Trainiere das Klassifikationsmodell
    crypto_rnn.train_classification_model()


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

    # 6. Vorhersagen treffen
    test_input = crypto_rnn.X_val[:5]  # Beispiel: Erste 5 Sequenzen
    predictions = crypto_rnn.predict(test_input)
    print("Vorhersagen:", predictions)

    # Rücknormalisierung der Vorhersagen
    # Mittelwert und Standardabweichung der Originaldaten berechnen
    ohlc_mean = crypto_rnn.df[['open', 'high', 'low', 'close']].mean()
    ohlc_std = crypto_rnn.df[['open', 'high', 'low', 'close']].std()


    # Rücknormalisieren der Vorhersagen
    predictions_original_scale = predictions * ohlc_std['close'] + ohlc_mean['close']
    predictions_original_scale = predictions_original_scale.detach().numpy()            # Konvertierung in numpy-Array
    print("Rücknormierte Vorhersagen (im Originalmaßstab):", predictions_original_scale)

    # Mean Absolute Deviation for predictions
    mad_predictions = np.mean(
        np.abs(predictions_original_scale - crypto_rnn.df['close'].iloc[-len(predictions):].values))
    print(f"Mean Absolute Deviation (MAD) of Predictions: {mad_predictions:.4f} USD")

    # Mean Absolute Deviation for average price
    average_price = crypto_rnn.df['close'].mean()
    mad_average = np.mean(np.abs(crypto_rnn.df['close'] - average_price))
    print(f"Mean Absolute Deviation (MAD) of Average Price: {mad_average:.4f} USD")

    # Zusätzliche Metriken: MSE und MAE
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # Tatsächliche Werte aus den Validierungsdaten
    actual_prices = crypto_rnn.df['close'].iloc[-(len(crypto_rnn.y_val_reg)):].values[:len(predictions_original_scale)]

    # Berechnung von MSE und MAE
    mse = mean_squared_error(actual_prices, predictions_original_scale)
    mae = mean_absolute_error(actual_prices, predictions_original_scale)

    print(f"Mean Squared Error (MSE): {mse:.4f} USD")
    print(f"Mean Absolute Error (MAE): {mae:.4f} USD")

    # Anzahl der Datenpunkte, die wir visualisieren wollen
    n = min(50, len(predictions_original_scale))

    # Bereite die tatsächlichen Werte (Validierungsdaten) und die Vorhersagen vor
    actual_prices = crypto_rnn.df['close'].iloc[-(len(crypto_rnn.y_val_reg)):].values[:n]
    predicted_prices = predictions_original_scale[:n]

    # Visualisierung der Vorhersagen
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Actual Prices", color='blue', alpha=0.7, marker='o', markersize=8, linewidth=2)
    plt.plot(predicted_prices, label="Predicted Prices", color='red', linestyle='--', alpha=0.7,  marker='o', markersize=8, linewidth=2)
    plt.title("Bitcoin Price Prediction")
    plt.xlabel("Time Steps")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()


    # 7. Handelsstrategien
    # Klassifikationsvorhersage für die nächste Stunde
    example_input = crypto_rnn.X_val_class[0]  # Beispiel: Erstes Validierungsbeispiel
    prediction = crypto_rnn.predict_classification(example_input)

    if prediction == 1:
        print("Strategie: Kaufen (Preis wird steigen).")
    elif prediction == -1:
        print("Strategie: Verkaufen (Preis wird fallen).")
    else:
        print("Strategie: Keine Aktion (Unsicherheit zu groß).")

