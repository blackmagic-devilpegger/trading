import krakenex
import requests
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import time


# EXPERIMENT:    VERSUCH   BORUTA-WRAPPER -> Boruta-Feature-Auswahl für KLASSIFIKATION
#
#Erwartung:  Identifikation relevanter Features zur Klassifikation (y_class: 1 = Preis steigt, -1 = Preis fällt).
#Ergebnis:   Die berechneten Korrelationen der Features mit y_class sind nahezu 0. D.h. Keine lineare Beziehung zw. den Features und der Zielvariable
#         Bsp: rsi = 0.0087, volatility= -0.0050
#Boruta-Ergebnis:  NICHTS AUSGEWÄHLT, da kein signifikanter Einfluss der Features auf die Klassifikation festgestellt.

# ANALYSE UND SCHLUSSFOLGERUNGEN:
# -NUTZUNG DER REGRESSION-FEATURES FÜR EIN CNN:
# Die Features, die in der Regression von Boruta als relevant identifiziert wurden (z. B. `open`, `close`, `rsi`, `ma_20`), könnten  für ein CNN genutzt werden.
# 1. Warum CNN?
#    - CNNs können nicht-lineare und lokale Muster in Zeitreihen erkennen, auch wenn die Korrelation der Features gering ist.
#    - Sie verarbeiten Eingabesequenzen, was die Nutzung von Trends und Bewegungsmustern ermöglicht.
# 2. Idee:
#    - Verwendung der Regression-Features als Eingabe für ein CNN zur Klassifikation (`y_class`: Preis steigt, fällt).
#    - Evtl. Transformation der Features in Sequenzen (z. B. Zeitfenster von 10 Perioden).
# 3. Erwartung:
#    - Verbesserung der Preisvorhersage, durch verwendung passender Features und ignorierung unpassender.


def run_boruta(X, y):
    """
    Führt Boruta Feature-Selektion durch.

    Parameter:
    - X: Eingabefeatures (DataFrame)
    - y: Zielvariable (Series)

    Rückgabe:
    - Liste der ausgewählten Features
    """
    # Random Forest als Basis für Boruta
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)

    # BorutaPy-Objekt initialisieren
    boruta = BorutaPy(estimator=rf, n_estimators='auto', random_state=42)

    # Boruta trainieren
    boruta.fit(X.values, y.values)

    # Ausgewählte Features extrahieren
    selected_features = X.columns[boruta.support_]
    print(f"Ausgewählte Features durch Boruta: {selected_features.tolist()}")

    return selected_features


def calculate_rsi(data, window=14):
    """
    Berechnet den Relative Strength Index (RSI).
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_moving_average(data, window):
    """
    Berechnet den gleitenden Durchschnitt.
    """
    return data.rolling(window=window).mean()


def calculate_volatility(data, window=20):
    """
    Berechnet die Volatilität basierend auf logarithmischen Renditen.
    """

    # 1. Berechnung der logarithmischen Renditen:
    # Logarithmische Renditen zeigen die prozentuale Änderung zwischen zwei aufeinanderfolgenden Werten auf
    # sie sind symmetrischer als einfache Renditen sind.
    log_returns = np.log(data / data.shift(1))

    # 2. Berechnung der rollierenden Standardabweichung:
    # Die Standardabweichung misst die Streuung der logarithmischen Renditen innerhalb eines Fensters.
    # Dies hilft dabei, die kurzfristige Schwankung oder Volatilität der Renditen zu quantifizieren.
    volatility = log_returns.rolling(window=window).std() * np.sqrt(window)

    # 3. Rückgabe der berechneten Volatilität:
    # Die berechnete Volatilität gibt an, wie stark die Preise innerhalb eines bestimmten Fensters schwanken.
    return volatility


class CryptoFeatureSelector:
    def __init__(self, api_key, seq_length=20):
        """
        Initialisiert die Klasse mit API-Key und Sequenzlänge.
        """
        self.api = krakenex.API()
        self.api.key = api_key
        self.seq_length = seq_length                                # Anzahl der Zeitpunkte in einer Eingabesequenz
        self.df = None                                              # DataFrame für die historischen Daten
        self.selected_features_class = None                         # Speichert ausgewählte Features für Klassifikation

    def fetch_data(self, pair="XXBTZUSD", interval=60, days=30):
        """
        Ruft historische Daten von der Kraken-API ab und speichert sie in einem DataFrame.
        """
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': pair,
            'interval': interval,
            'since': int(time.time()) - 60 * 60 * 24 * days
        }

        response = requests.get(url, params=params)
        data = response.json()

        if len(data['error']) == 0:
            ohlc = data['result'][pair]
            self.df = pd.DataFrame(
                ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            )
            self.df['time'] = pd.to_datetime(self.df['time'], unit='s')

            # Konvertiere relevante Spalten in numerische Werte
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df = self.df.dropna()

            # Zusätzliche Features berechnen
            self.df['rsi'] = calculate_rsi(self.df['close'])
            self.df['ma_20'] = calculate_moving_average(self.df['close'], 20)
            self.df['ma_50'] = calculate_moving_average(self.df['close'], 50)
            self.df['volatility'] = calculate_volatility(self.df['close'], 20)

            # Entferne Zeilen mit NaN-Werten
            self.df = self.df.dropna()

        else:
            raise Exception(f"API-Fehler: {data['error']}")

    def create_class_labels(self):
        """
        Erstellt Klassifikationslabels basierend auf Preisänderungen.
        """
        margin = 0.02
        labels = []
        for i in range(len(self.df) - self.seq_length - 1):
            price_diff = self.df['close'].iloc[i + self.seq_length + 1] - self.df['close'].iloc[i + self.seq_length]
            if price_diff > margin:
                labels.append(1)
            elif price_diff < -margin:
                labels.append(-1)
            else:
                labels.append(0)
        return pd.Series(labels)

    def run_feature_selection(self):
        """
        Führt Boruta-Feature-Auswahl für Klassifikation durch.
        """
        if self.df is None:
            raise Exception("Keine Daten verfügbar. Rufe zuerst fetch_data() auf.")

        # Eingabefeatures und Zielvariablen definieren
        X = self.df[['open', 'high', 'low', 'close', 'rsi', 'ma_20', 'ma_50', 'volatility']]
        y_class = self.create_class_labels()

        # Debugging: Überprüfe die Verteilung der Zielvariable
        print("\nVerteilung von y_class:")
        print(y_class.value_counts())

        # Entfernen der Klasse 0 (stabile Preise), da diese stark unterrepräsentiert ist
        y_class = y_class[y_class != 0]
        X = X.iloc[:len(y_class)]               # Kürze X, um die Länge von y_class anzupassen

        # Korrelation der Features mit der bereinigten Zielvariable y_class berechnen
        print("\nKorrelation der Features mit y_class:")
        print(X.corrwith(y_class))

        # Boruta für Klassifikation
        print("\nBoruta-Feature-Auswahl für Klassifikation:")
        self.selected_features_class = run_boruta(X, y_class)

        print("\nFinale Ergebnisse:")
        print(f"Klassifikation Features: {self.selected_features_class.tolist()}")


if __name__ == "__main__":
    # Instanziiere die Klasse mit deinem API-Key
    api_key = 'w3if4ZjPEKdgCVsj7J/KVRgkSKhAhYBcJJrrp8gXTfrRdlylAVafK85F'
    feature_selector = CryptoFeatureSelector(api_key=api_key)

    # 1. Daten abrufen
    feature_selector.fetch_data()

    # 2. Boruta-Feature-Auswahl ausführen
    feature_selector.run_feature_selection()
