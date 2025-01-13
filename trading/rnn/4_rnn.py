import krakenex
import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler
import matplotlib.pyplot as plt

# EXPERIMENT
# NORMALISIERUNG angepasst:MIin/MAxScaler zur Skalierung der Daten verwendet
#Erwartung: Werte werden in einem festen Bereich ( z.b 0,1) sklaiert (wodurch das MOdell stabiler trainiert werden kann)was in vorherigen Experimenten bessere Ergebnisse erzielt
#Ergebnis: MinMax-Normalisierung führte zu stabileren Verlustwerten während des Trainings, erzielte jedoch schlechtere Vorhersagen (MAD: 94048.1768 USD).

# RÜCKNORMALISIERUNG angepasst: mit MInMaxScaler
#Erwartung: Rücknormalisierung mit MInMax soll eine konsistente Skalierung in Übereinstimmung mit MinMaxscaler gewährleisten.
#Ergebnis: Diese Rücknormalisierung erzeugte ebenfalls unrealistische Vorhersagen(stark negative werte)
# Kann auf eine fehlerhafte skalierung hindeuten.

#VERLUSTWERTE: VAlid_loss sank auf 0.000047 , jedoch führte die Vorhersage zu einer signifikant höheren MAD (94048.1768)!!!!!!

#VORHERSAGEN &  MAD: Rücknormalisierte Vorhersagen wie [-5.516801, -5.516799] waren stark unrealistisch.
# Die MAD (94048.1768 USD) war signifikant höher, was die schlechteste Vorhersage bisher darstellt.

# => Evtl. ein Fehler unterlaufen ?Nichts gefunden.

class CryptoRNN:

    def __init__(self, api_key, seq_length=10):
        """
        Initialisiert die Klasse mit API-Key und Sequenzlänge.
        """
        self.api = krakenex.API()
        self.api.key = api_key
        self.seq_length = seq_length                            # Anzahl der Zeitpunkte in einer Eingabesequenz
        self.df = None                                          # Dataframe für die historischen Daten
        self.X = None                                           # Eingabedaten für das Modell
        self.y = None                                           # Zielwerte für das modell
        self.regression_model = None
        self.model = None
        self.scaler = MinMaxScaler()                            # MinMaxScaler für Normalisierung
        self.ohlc_max = None                                    # Max-Werte für Rücknormalisierung
        self.ohlc_min = None                                    # Min-Werte für Rücknormalisierung
    def fetch_data(self, pair="XXBTZUSD", interval=60, days=30):
        """
        Ruft historische Daten von der Kraken-API ab und speichert sie in einem DataFrame.
        """
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': pair,                                       # Handelspaar: Bitcoin zu USd
            'interval': interval,                               # Zeitinterval in Minuten
            'since': int(time.time()) - 60 * 60 * 24 * days     # Startzeit (aktuelle Zeit minus days)
        }

        # API-Anfrage
        response = requests.get(url, params=params)
        data = response.json()

        if len(data['error']) == 0:                             # Überprüfung auf Fehler in der API-Antwort
            # Daten in einen dataframe laden
            ohlc = data['result'][pair]
            self.df = pd.DataFrame(
                ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            )
            # Konvertiere die Zeitspalte in ein Datetime-Format
            self.df['time'] = pd.to_datetime(self.df['time'], unit='s')

            # Konvertiere relevante Spalten in numerische Werte und entferne fehlerhafte Zeilen
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df = self.df.dropna()                          # Entferne Zeilen mit NaN-Werten

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

        # Speichern der Max- und Min-Werte für Rücknormalisierung
        self.ohlc_max = ohlc_data.max()
        self.ohlc_min = ohlc_data.min()

        # Normalisierung der Daten mit MinMaxScaler
        normalized_data = pd.DataFrame(self.scaler.fit_transform(ohlc_data), columns=ohlc_data.columns)

        # Initialisierung der Sequenzen und Zielvariablen
        X, y_reg, y_class = [], [], []
        margin = 0.02                                                   # Sicherheitsmarge von 2 %

        # Erstellung der Eingabesequenzen und Zielwerte
        for i in range(len(normalized_data) - self.seq_length -1):
            X.append(normalized_data.iloc[i:i+self.seq_length].values)
            y_reg.append(normalized_data.iloc[i+self.seq_length]['close'])

            # Klassifikationslabel erstellen basierend auf Preisänderung
            price_diff = self.df['close'].iloc[i + self.seq_length + 1] - self.df['close'].iloc[i + self.seq_length]
            if price_diff > margin:
                y_class.append(1)  # Steigt
            elif price_diff < -margin:
                y_class.append(-1)  # Fällt
            else:
                y_class.append(0)  # Konstant

        # Konvertiere die Listen in numpy-Arrays und dann in Tensoren
        X = np.array(X)
        y_reg = np.array(y_reg)
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
                self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)        # RNN-Layer
                self.dropout = nn.Dropout(dropout_prob)                                           # Dropout-Layer für Regularisierung
                self.fc = nn.Linear(hidden_size, output_size)                                     # Fully Connected Layer

            def forward(self, x):
                h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)     # Initialisiere den versteckten Zustand
                out, _ = self.rnn(x, h_0)                                           # RNN Berechnung
                out = out[:, -1, :]                                                 # Nimmt nur den letzten Zeitschritt(Batchgröße, Hidden_Size)
                out = self.dropout(out)                                             # Dropout nach dem letzten Zeit-Schritt
                out = self.fc(out)                                                  # Ausgabe des letzten Zeitpunkts
                return out

        self.model = RNNModel(input_size, hidden_size, output_size)
        print("Modell mit Dropout erstellt.")

    def train_model(self, num_epochs=50, batch_size=32, learning_rate=0.0001, patience=10):
        """
        Trainiert das RNN-Modell mit Early Stopping.

        """
        # Überprüfung, ob ein Modell existiert
        if self.model is None:
            raise Exception("Kein Modell verfügbar. Rufe create_model() auf.")

        criterion = nn.MSELoss()                                                    # Verlustfunktion (Mean-Squared Error) für Regressionsprobleme
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)           # Adam-Optimierer mit spezifiziertem Lernrate

        # Listen zur Speicherung der Verlustwerte
        self.train_loss_values = []
        self.val_loss_values = []

        #Early Stopping Variablen
        best_val_loss = float("inf")                                                # Setze initial besten Validierungsverlust auf unendlich
        epochs_no_improve = 0                                                       # Zählt die Epochen ohne Verbesserung

        # Trainingsschleife über die spezifizierten Epochen
        for epoch in range(num_epochs):
            epoch_loss = 0                                                          # Gesamtverlust der aktuellen Epoche initialisieren

            #Training
            self.model.train()
            # Batchweise Training
            for i in range(0, len(self.X_train), batch_size):
                # Erstellen eines Batches aus den Trainingsdaten
                X_batch = self.X_train[i:i+batch_size]
                y_batch = self.y_train_reg[i:i+batch_size]

                optimizer.zero_grad()                                               # Rücksetzen der Gradienten
                outputs = self.model(X_batch)                                       # Vorhersage mit dem Modell
                loss = criterion(outputs.squeeze(), y_batch)                        #Berechne den Verlust
                loss.backward()                                                     #BAckpropagation(Gradientenberechnung)
                optimizer.step()

                # Addiere den Batch-Verlust zum epoch_loss,um den Gesamtverlust für die aktuelle Epoche zu berechnen
                epoch_loss += loss.item()

            # Validierung: Evaluierung des Modells ohne Gradientenberechnung
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

            # Berechne R²-Score für die Validierungsdaten
            y_val_actual = self.y_val_reg.numpy()
            r2 = r2_score(y_val_actual, y_val_pred)

            # Ausgabe der Ergebnisse pro Epoche
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss / len(self.X_train):.6f}, Validation Loss: {val_loss / len(self.X_val):.6f}, R²: {r2:.6f}")

            # Überprüfung für Early Stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                epochs_no_improve = 0                           # Zurücksetzen des Zählers bei Verbesserung
            else:
                epochs_no_improve += 1                          # Erhöhung des Zählers bei keiner Verbesserung

            if epochs_no_improve == patience:                   # Abbruchkriterium bei fehlender Verbesserung
                print(f"Frühes Stoppen nach {epoch + 1} Epochen. Validation Loss hat sich {patience} Epochen nicht verbessert.")
                break


    def train_classification_model(self):
        """
        Trainiert ein Klassifikationsmodell.
        """
        from sklearn.ensemble import RandomForestClassifier
        # Random Forest Classifier mit 100 Entscheidungsbäumen
        self.classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classification_model.fit(self.X_train_class.reshape(len(self.X_train_class), -1), self.y_train_class)
        print("Klassifikationsmodell trainiert.")

    def predict_classification(self, input_data):
        """
        Gibt Klassifikationsvorhersagen zurück.
        """
        # Klassifikationsvorhersage (1: Kaufen, -1: Verkaufen, 0: Keine Aktion)
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
            predictions = self.model(input_data) # Berechnung der Vorhersagen mit dem Modell
        return predictions


if __name__ == "__main__":
    # Instanziiere die Klasse mit deinem API-Key
    crypto_rnn = CryptoRNN(api_key='w3if4ZjPEKdgCVsj7J/KVRgkSKhAhYBcJJrrp8gXTfrRdlylAVafK85F')

    # 1. Daten abrufen
    # Ruft historische Daten von der Kraken-API ab und speichert sie in einem DataFrame.
    crypto_rnn.fetch_data()

    # 2. Daten vorbereiten
    # Normalisiert die Daten und erstellt Sequenzen sowie Zielwerte für das Modell.
    crypto_rnn.prepare_data()

    # 3. Modell erstellen
    crypto_rnn.create_model(input_size=4, hidden_size=50, output_size=1)

    # 4. RegressionsModell trainieren
    # Trainiert das RNN-Modell mit den vorbereiteten Daten und speichert Trainings- und Validierungsverluste.
    crypto_rnn.train_model(num_epochs=50, batch_size=32, learning_rate=0.0005)


    # 5. Trainiere das Klassifikationsmodell
    # Trainiert einen Random Forest Classifier für die Klassifikation von Preisänderungen.
    crypto_rnn.train_classification_model()


    ####Visualisierung des Trainings-und Validierungsverlustes
    #Extrahiert die Verlustwerte für Training und Validierung
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
    # Nimmt eine Beispiel-Eingabe aus den Validierungsdaten
    test_input = crypto_rnn.X_val[:5]                                                # Hier Beispiel: Erste 5 Sequenzen
    predictions = crypto_rnn.predict(test_input).detach().numpy().flatten()          # Berechnet Vorhersagen mit dem trainierten Modell

    # Rücknormalisierung der Vorhersagen mit MInMaxScaller (nur für die Spalte "close")
    close_min = crypto_rnn.scaler.min_[3]                                       # Index 3 für "close"
    close_range = crypto_rnn.scaler.scale_[3]                                   # Skalenfaktor für "close"
    predictions_original_scale = predictions * close_range + close_min
    print("Vorhersagen (rücknormalisiert):", predictions_original_scale)


    # Mean Absolute Deviation for predictions
    # Berechnet die mittlere absolute Abweichung zwischen den Vorhersagen und den tatsächlichen Werten
    mad_predictions = np.mean(
        np.abs(predictions_original_scale - crypto_rnn.df['close'].iloc[-len(predictions):].values))
    print(f"Mean Absolute Deviation (MAD) of Predictions: {mad_predictions:.4f} USD")

    # Mean Absolute Deviation for average price
    # Berechnet die mittlere absolute Abweichung der tatsächlichen Preise vom Durchschnittspreis
    average_price = crypto_rnn.df['close'].mean()
    mad_average = np.mean(np.abs(crypto_rnn.df['close'] - average_price))
    print(f"Mean Absolute Deviation (MAD) of Average Price: {mad_average:.4f} USD")

    # Anzahl der Datenpunkte, die visualisiert werden sollen
    n = min(50, len(predictions_original_scale))

    # Bereitet die tatsächlichen Werte (Validierungsdaten) und die Vorhersagen vor
    actual_prices = crypto_rnn.df['close'].iloc[-(len(crypto_rnn.y_val_reg)):].values[:n]
    predicted_prices = predictions_original_scale[:n]

    # Visualisierung der Vorhersagen
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
    example_input = crypto_rnn.X_val_class[0]                       # Beispiel: Erstes Validierungsbeispiel
    prediction = crypto_rnn.predict_classification(example_input)

    # Ausgabe der Strategie basierend auf der Klassifikationsvorhersage
    if prediction == 1:
        print("Strategie: Kaufen (Preis wird steigen).")
    elif prediction == -1:
        print("Strategie: Verkaufen (Preis wird fallen).")
    else:
        print("Strategie: Keine Aktion (Unsicherheit zu groß).")