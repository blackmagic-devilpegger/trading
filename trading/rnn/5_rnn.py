import krakenex
import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# EXPERIMENT

# +FEATURE-ENGINEERING hinzugefügt: RSI, MA, Volatilität
# Erwartung: Verbesserte Analyse der Markttrends (Trend und Momentum) durch zusätzliche Features wie Relative Strength Index (RSI), gleitende Durchschnitte (MA 20, MA 50) und Volatilität.
# Ergebnis: Validation Loss sank kontinuierlich von 0.0816 (Epoche 1) auf 0.0188 (Epoche 22).
#           Visualisierungen zeigten klare Markttrends und Schwankungen, die Handelsstrategien unterstützen.

# +Z-SCORE-NORMALISIERUNG hinzugefügt
# Erwartung: Stabilisierung des Trainingsprozesses durch Standardisierung der Eingabedaten.
# Ergebnis: Trainings- und Validierungsverluste waren konsistenter ohne plötzliche hohe Anstiege.
#           Beispiel: Training Loss sank bis auf 0.0015 (Epoche 29), stieg jedoch in späteren Epochen leicht an.


# +ERWEITERTE VISUALISIERUNGEN hinzugefügt: RSI, MA, Volatilität
# Erwartung: Bessere Nachvollziehbarkeit der Daten durch Plots von RSI, MA und Volatilität.
# Ergebnis: Visualisierungen zeigten deutliche Trends und Schwankungen im Markt, was die Analyse und Entscheidungsfindung erleichterte.

# EVALUATION erweitert: Berechnung von MAD und R²-Score
# Erwartung: Präzisere Bewertung der Modellleistung.
# Ergebnis: R²-Score stieg bis auf 0.952 (Epoche 22), was eine gute Übereinstimmung zeigt.
#           Mean Absolute Deviation (MAD) der Vorhersagen war mit 7619.68 USD weiterhin höher als die MAD des Durchschnittspreises (2862.31 USD).
#           Trotz hoher Genauigkeit des R² bleibt Verbesserungspotential bei der Vorhersagegenauigkeit.

# +RÜCKNORMALISIERUNG der Vorhersagen hinzugefügt
# Erwartung: Rücknormalisierung soll realistische Preise liefern, die mit den tatsächlichen Schlusskursen vergleichbar sind.
# Ergebnis: Rücknormierte Vorhersagen (z. B. 101758.85 USD bis 101579.64 USD) waren realistisch.
#           Die weiterhin hohe Mean Absolute Deviation (MAD) zeigt Optimierungsbedarf bei der Modellfeinjustierung.

# VALIDATION LOSS UND TRAIN LOSS
# - Der Training Loss sank von 0.0337 (Epoche 1) auf 0.0015 (Epoche 29), was auf eine effektive Anpassung des Modells hinweist.
# - Der Validation Loss sank bis auf 0.0188 (Epoche 22), zeigte jedoch in späteren Epochen leichte Schwankungen und Anstiege (z. B. bis 0.0208 in Epoche 25).
# - Diese Schwankungen könnten auf beginnendes Overfitting hindeuten, da das Modell nach Epoche 22 keine signifikanten Verbesserungen mehr erzielte.
# - Die insgesamt niedrigen Validation Loss Werte zeigen dennoch, dass das Modell die Daten weitgehend gut generalisiert hat.

# FAZIT:
# RNN_5 zeigt klare Verbesserungen im Vergleich zu RNN_4:
# - Verbesserte Stabilität und Genauigkeit durch Feature-Engineering und Z-Score-Normalisierung.
# - Effizientes Training durch frühzeitiges Stoppen.
# - Präzisere Evaluation mit zusätzlichen Metriken.
# - Leichte Schwankungen im Valid_Loss und Train_loss, als auch die weiterhin hohe Mean Absolute Deviation (MAD) der Vorhersagen zeigen Optimierungspotential,
#   insbesondere bei der Feinjustierung des Modells.

# ==> Die Werte von MAD schwanken ständing, bei jedem Ausführen, da man immer mit aktuellen Daten arbeitet. Die smacht eine Vorhersage mit einem RNN schwer.

def calculate_rsi(data, window=14):
    """
    Berechnet den Relative Strength Index (RSI).
    """
    delta = data.diff()                                                 # Differenzen zwischen den Datenpunkten berechnen
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()    # Positive Gewinne
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()   # Negative Verluste
    rs = gain / loss                                                    # Verhältnis von Gewinnen zu Verlusten
    rsi = 100 - (100 / (1 + rs))                                        # Berechnung des RSI
    return rsi

# Feature-Berechnungsfunktionen
def calculate_moving_average(data, window):
    """
    Berechnet den gleitenden Durchschnitt.
    """
    return data.rolling(window=window).mean()                              # Rolling Mean berechnen

def calculate_volatility(data, window=20):
    """
    Berechnet die Volatilität basierend auf logarithmischen Renditen.
    """
    log_returns = np.log(data / data.shift(1))                              # Logarithmische Renditen berechnen
    volatility = log_returns.rolling(window=window).std() * np.sqrt(window) # Standardabweichung der Renditen
    return volatility


class CryptoRNN:

    def __init__(self, api_key, seq_length=20):
        """
        Initialisiert die Klasse mit API-Key und Sequenzlänge.
        """
        self.api = krakenex.API()
        self.api.key = api_key
        self.seq_length = seq_length                            # Sequenzlänge für das RNN (Anzahl der Zeitpunkte)
        self.df = None                                          # Dataframe zur speicherung der abgerufenen Daten
        self.X_train_class, self.X_val_class = None, None       # Trainings- und Validierungsdaten (Klassifikation)
        self.y_train_class, self.y_val_class = None, None       # Labels für Klassifikation
        self.classification_model = None
        self.regression_model = None
        self.train_loss_values = []                             # Liste zur Speicherung der Trainingsverluste
        self.val_loss_values = []                               # Liste zur Speicherung der Validierungsverluste

    def fetch_data(self, pair="XXBTZUSD", interval=60, days=30):
        """
        Ruft historische Daten von der Kraken-API ab und speichert sie in einem DataFrame.
        """
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': pair,                                       # Handelspaar z.B Bitcoin zu USd
            'interval': interval,                               # Zeitinterval in Minuten
            'since': int(time.time()) - 60 * 60 * 24 * days     # Startzeit (aktuelle Zeit minus days)
        }

        response = requests.get(url, params=params)             # API-Abfrage senden
        data = response.json()                                  # Antwort in JSON konvertieren

        if len(data['error']) == 0:
            # Daten in einen dataframe laden
            ohlc = data['result'][pair]
            self.df = pd.DataFrame(
                ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            )
            self.df['time'] = pd.to_datetime(self.df['time'], unit='s')       # Zeit in Datetime-Format umwandeln

            # Konvertiere relevante Spalten in numerische Werte und entferne fehlerhafte Zeilen
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')             # Fehlerhafte Werte als NaN behandeln
            self.df = self.df.dropna()

            # Berechnung der neuen Features
            # Berechnung der RSI (Relative Strength Index)
            self.df['rsi'] = calculate_rsi(self.df['close'])

            # Berechnung der gleitenden Durchschnitte (20-Tage und 50-Tage)
            self.df['ma_20'] = calculate_moving_average(self.df['close'], window=20)
            self.df['ma_50'] = calculate_moving_average(self.df['close'], window=50)

            # Berechnung der Volatilität
            self.df['volatility'] = calculate_volatility(self.df['close'], window=20)

            # Entferne NaN-Werte, die durch die Berechnungen entstehen könnten
            self.df = self.df.dropna()

            print("Daten erfolgreich abgerufen:")
            print(self.df.head())                                                               # Ausgabe der ersten Zeilen des DataFrames

            # Plot der berechneten Features
            plt.figure(figsize=(12, 6))
            plt.plot(self.df['time'], self.df['close'], label='Close Price')
            plt.plot(self.df['time'], self.df['rsi'], label='RSI', linestyle='--')
            plt.plot(self.df['time'], self.df['ma_20'], label='MA 20', linestyle='-.')
            plt.plot(self.df['time'], self.df['volatility'], label='Volatility', linestyle=':')
            plt.legend()
            plt.title("Features Visualization")
            plt.show()
        else:
            raise Exception(f"API-Fehler: {data['error']}")

    def prepare_data(self, validation_split=0.2):
        """
        Bereitet die Daten für das RNN vor: Normalisierung und Sequenzierung.
        """
        if self.df is None:
            raise Exception("Keine Daten verfügbar. Rufe zuerst fetch_data() auf.")

        # Relevante Spalten auswählen und normalisieren
        relevant_columns = ['open', 'high', 'low', 'close', 'rsi', 'ma_20', 'ma_50', 'volatility']
        normalized_data = (self.df[relevant_columns] - self.df[relevant_columns].mean()) / self.df[
            relevant_columns].std()

        # Initialisierung der Zielvariablen und der Sicherheitsmarge
        X, y_reg, y_class = [], [], []
        margin = 0.02                                                           # Sicherheitsmarge von 2 %

        # Sequenzen und Zielwerte erstellen
        for i in range(len(normalized_data) - self.seq_length -1):
            X.append(normalized_data.iloc[i:i+self.seq_length].values)          # Eingabesequenz
            y_reg.append(normalized_data.iloc[i+self.seq_length]['close'])      # Zielwert (Regression)

            # Klassifikationslabel erstellen
            price_diff = self.df['close'].iloc[i + self.seq_length + 1] - self.df['close'].iloc[i + self.seq_length]
            if price_diff > margin:
                y_class.append(1)  # Steigt
            elif price_diff < -margin:
                y_class.append(-1)  # Fällt
            else:
                y_class.append(0)  # Konstant

        # Konvertiere die Listen in numpy-Arrays und dann in Tensoren
        X = np.array(X)                                                             # Konvertiere die Liste in ein numpy-Array
        y_reg = np.array(y_reg)                                                     # Konvertiere die Liste in ein numpy-Array
        y_class = np.array(y_class)


        # Split in Trainings- und Validierungsdaten
        split_idx = int(len(X) * (1 - validation_split))
        self.X_train, self.X_val = torch.tensor(X[:split_idx], dtype=torch.float32), torch.tensor(X[split_idx:],
                                                                                                  dtype=torch.float32)
        self.y_train_reg, self.y_val_reg = torch.tensor(y_reg[:split_idx], dtype=torch.float32), torch.tensor(y_reg[split_idx:],
                                                                                                  dtype=torch.float32)

        # Daten für Klassifikation speichern
        self.X_train_class = X[:split_idx]
        self.X_val_class = X[split_idx:]
        self.y_train_class = y_class[:split_idx]
        self.y_val_class = y_class[split_idx:]

    def create_regression_model(self, input_size, hidden_size, output_size, num_layers=2):
        """
        Erstellt das RNN-Modell.
        """
        class RNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
                super(RNNModel, self).__init__()
                self.num_layers = num_layers  # Anzahl der Schichten speichern
                self.hidden_size = hidden_size  # Versteckte Einheiten speichern
                self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)        # RNN-Layer
                self.dropout = nn.Dropout(dropout_prob)                                                    # Dropout-Layer für Regularisierung
                self.fc = nn.Linear(hidden_size, output_size)                                              # Fully Connected Layer

            def forward(self, x):
                h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)                            # Initialisiere den versteckten Zustand
                out, _ = self.rnn(x, h_0)                                                                  #RNN Berechnung
                out = out[:, -1, :]                                                                        # Zugriff auf den letzten Zeitschritt (Batchgröße, Hidden_Size)
                out = self.dropout(out)                                                                    # Dropout nach dem letzten Zeit-Schritt
                out = self.fc(out)                                                                         # Ausgabe des letzten Zeitpunkts
                return out

        self.regression_model = RNNModel(input_size, hidden_size, output_size)
        print("Regressionsmodell erstellt.")



    def train_regression_model(self, num_epochs=50, batch_size=16, learning_rate=0.00001, patience=10):
        """
        Trainiert das RNN-Modell mit Early Stopping.

        """
        if self.regression_model is None:
            raise Exception("Kein Regressionsmodell verfügbar. Rufe create_regression_model() auf.")

        criterion = nn.MSELoss()                                                               # Verlustfunktion (Mean-Squared Error)
        optimizer = optim.Adam(self.regression_model.parameters(), lr=learning_rate)           # Adam-Optimierer


        # Listen zur Speicherung der Verluste
        self.train_loss_values = []
        self.val_loss_values = []

        #Early Stopping Variablen
        best_val_loss = float("inf")                                                    # Start mit einem sehr hohen Verlustwert
        epochs_no_improve = 0                                                           # Zählt die Epochen ohne Verbesserung

        for epoch in range(num_epochs):
            epoch_loss = 0

            #Training
            self.regression_model.train()
            for i in range(0, len(self.X_train), batch_size):
                X_batch = self.X_train[i:i+batch_size]
                y_batch = self.y_train_reg[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.regression_model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)                        # Berechne den Verlust
                loss.backward()                                                     # Backpropagation
                optimizer.step()

                # Addiere den Batch-Verlust zum epoch_loss,um den Gesamtverlust für die aktuelle Epoche zu berechnen
                epoch_loss += loss.item()

            # Validierung
            self.regression_model.eval()
            val_loss = 0
            y_val_pred = []
            y_val_pred_classification = []
            y_val_pred_regression = []  # Initialisiere die Liste vor der Schleife

            with torch.no_grad():
                for i in range(len(self.X_val)):
                    # Regressionsvorhersage
                    X_val_instance = self.X_val[i].unsqueeze(0)                                     # Einzelne Sequenz
                    y_val_instance = self.y_val_reg[i]
                    outputs = self.regression_model(X_val_instance)
                    val_loss += criterion(outputs.squeeze(), y_val_instance).item()
                    y_val_pred_regression.append(outputs.item())



            # Speichere die Verluste
            train_loss_epoch = epoch_loss / len(self.X_train)
            val_loss_epoch = val_loss / len(self.X_val)
            self.train_loss_values.append(train_loss_epoch)
            self.val_loss_values.append(val_loss_epoch)

            # Berechne R²-Score für Regression
            y_val_actual_regression = self.y_val_reg.numpy()
            r2 = r2_score(y_val_actual_regression, y_val_pred_regression)

            # Ausgabe der Verluste und R²-Score
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss_epoch:.6f}, "f" Validation Loss: {val_loss_epoch:.6f}, R²: {r2:.6f}")


            # Überprüfung für Early Stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                epochs_no_improve = 0                               # Zurücksetzen, wenn es eine Verbesserung gab
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(f"Frühes Stoppen nach {epoch + 1} Epochen. Validation Loss hat sich {patience} Epochen nicht verbessert.")
                break

    def create_classification_model(self):
        """
        Erstellt das Klassifikationsmodell.
        """
        from sklearn.ensemble import RandomForestClassifier
        self.classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Klassifikationsmodell erstellt.")


    def train_classification_model(self):
        """
        Trainiert ein Klassifikationsmodell.
        """
        from sklearn.ensemble import RandomForestClassifier
        if self.classification_model is None:
            raise Exception("Kein Klassifikationsmodell verfügbar. Rufe create_classification_model() auf.")

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
        if self.regression_model is None:
            raise Exception("Kein Regressionsmodell verfügbar. Rufe create_regression_model() auf.")

        self.regression_model.eval()
        with torch.no_grad():
            predictions = self.regression_model(input_data)
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

    # 3. Regressionsmodell erstellen und trainieren
    crypto_rnn.create_regression_model(input_size=8, hidden_size=128, output_size=1, num_layers=2)
    crypto_rnn.train_regression_model(num_epochs=50, batch_size=16, learning_rate=0.0005)

    # 4. Klassifikationsmodell erstellen und trainieren
    crypto_rnn.create_classification_model()
    crypto_rnn.train_classification_model()

    # Beispiel: Klassifikationsvorhersage für die nächste Stunde
    example_input = crypto_rnn.X_val_class[0]                           # Beispiel: Erstes Validierungsbeispiel
    prediction = crypto_rnn.predict_classification(example_input)

    if prediction == 1:
        print("Strategie: Kaufen (Preis wird steigen).")
    elif prediction == -1:
        print("Strategie: Verkaufen (Preis wird fallen).")
    else:
        print("Strategie: Keine Aktion (Unsicherheit zu groß).")

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
    test_input = crypto_rnn.X_val[:5]                                   # Beispiel: Erste 5 Sequenzen
    predictions = crypto_rnn.predict(test_input)
    print("Vorhersagen:", predictions)

    # Rücknormalisierung der Vorhersagen
    # Mittelwert und Standardabweichung der Originaldaten berechnen
    ohlc_mean = crypto_rnn.df[['open', 'high', 'low', 'close']].mean()
    ohlc_std = crypto_rnn.df[['open', 'high', 'low', 'close']].std()


    # Rücknormalisieren der Vorhersagen
    predictions_original_scale = predictions * ohlc_std['close'] + ohlc_mean['close']
    predictions_original_scale = predictions_original_scale.detach().numpy()                        # Konvertierung in numpy-Array
    print("Rücknormierte Vorhersagen (im Originalmaßstab):", predictions_original_scale)

    # Berechnung der tatsächlichen Werte für den Vergleich
    actual_prices = crypto_rnn.df['close'].iloc[-len(predictions):].values

    # Mean Absolute Deviation for predictions
    mad_predictions = np.mean(np.abs(predictions_original_scale - actual_prices))
    print(f"Mean Absolute Deviation (MAD) of Predictions: {mad_predictions:.4f} USD")

    # Mean Absolute Error for predictions
    mae_predictions = mean_absolute_error(actual_prices, predictions_original_scale)
    print(f"Mean Absolute Error (MAE) of Predictions: {mae_predictions:.4f} USD")

    # Mean Squared Error for predictions
    mse_predictions = np.mean((predictions_original_scale - actual_prices) ** 2)
    print(f"Mean Squared Error (MSE) of Predictions: {mse_predictions:.4f} USD")

    # Mean Absolute Deviation for average price
    average_price = crypto_rnn.df['close'].mean()
    mad_average = np.mean(np.abs(crypto_rnn.df['close'] - average_price))
    print(f"Mean Absolute Deviation (MAD) of Average Price: {mad_average:.4f} USD")

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
    example_input = crypto_rnn.X_val_class[0]                           # Beispiel: Erstes Validierungsbeispiel
    prediction = crypto_rnn.predict_classification(example_input)

    if prediction == 1:
        print("Strategie: Kaufen (Preis wird steigen).")
    elif prediction == -1:
        print("Strategie: Verkaufen (Preis wird fallen).")
    else:
        print("Strategie: Keine Aktion (Unsicherheit zu groß).")


