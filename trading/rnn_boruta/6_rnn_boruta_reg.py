import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor



# EXPERIMENT:    BORUTA-WRAPPER -> Boruta-Feature-Auswahl für REGRESSION DURCHGEFÜHRT
#1. Verwendung von BorutaPy
#2. Automatische Integration von BorutaPy mit Random Forest regressor
#3. Output der ausgewählten Features
#
# Erwartung: Auswahl der wichtigsten Features für die Zielvariable `close`, um die Modellkomplexität zu reduzieren und die Vorhersagegenauigkeit zu erhöhen.
# Ergebnis: Boruta identifizierte folgende relevante Features:
#           - `open`: Eröffnungswert der Periode.
#           - `high`: Höchstwert der jeweiligen Periode.
#           - `low`: Tiefstwert der jeweiligen Periode.
#           - `close`: Schlusswert der Periode (stark korreliert mit sich selbst).
#           - `rsi`: Relative Strength Index zur Bewertung von Preisbewegungen.
#           - `ma_20`: Gleitender Durchschnitt der Schlusskurse über die letzten 20 Perioden.
#      Nicht relevante Features (`ma_50`, `volatility`) wurden als weniger wichtig eingestuft.

# +DATENBEREINIGUNG und FEATURE-BERECHNUNG vor Boruta durchgeführt
# Erwartung: Erstellung eines sauberen und robusten Datensatzes mit zusätzlichen Features zur Verbesserung der Modellbasis.
# Ergebnis: Zusätzliche Features (`rsi`, `ma_20`, `ma_50`, `volatility`) wurden erfolgreich berechnet.
#           Daten wurden bereinigt, NaN-Werte entfernt und in ein geeignetes Format gebracht.

# WEITERE GEPLANTE SCHRITE:

# -BORUTA-FEATURE-AUSWAHL für Klassifikation geplant
# Ziel: Entwicklung eines separaten Klassifikationsmodells zur Vorhersage von Preisbewegungen (z. B. Steigt/Fällt/Konstant) durch Auswahl der wichtigsten Features.
# Erwartung: Verbesserung der Klassifikationsgenauigkeit durch Reduktion auf relevante Features für die Klassifikationsaufgabe.

# -ANWENDUNG DER AUSGEWÄHLTEN FEATURES im Modell
# Erwartung: Verbesserung der Modellleistung durch Reduktion auf relevante Features.
# Ergebnis: Das Modell wird effizienter trainiert, da irrelevante Features entfernt wurden.
#           Die ausgewählten Features dienen als Eingabe für das Regressionsmodell.
#
#-CNN-Erstellen und trainiernen
# evtl. Ergebnisse mit Baseline-modellen vergleichen
# evtl. hyperparameter optimieren
# Ergebnisse visualisierne und interpretieren

class RegressionFeatureSelector:
    def __init__(self, data):
        """
        Initialisiert den Feature-Selector für Regression.
        :param data: Pandas DataFrame mit den Eingabefeatures und Zielvariable.
        """
        self.data = data                       # Datensatz, der Features und Zielvariable enthält
        self.selected_features = None          # Variable zur Speicherung der ausgewählten Features


    def run_boruta(self):
        """
        Führt Boruta-Feature-Auswahl für Regression durch.
        """
        print("\nStarte Boruta-Feature-Auswahl für Regression...")

        # Eingabefeatures und Zielvariable definieren
        X = self.data[['open', 'high', 'low', 'close', 'rsi', 'ma_20', 'ma_50', 'volatility']]
        y = self.data['close']

        # Debugging: Form und Statistik der Daten anzeigen
        print(f"Shape von X: {X.shape}, Shape von y: {y.shape}")        # Dimensionen der Features und Zielvariable
        print(f"Statistiken der Zielvariable y:\n{y.describe()}")       # Statistische Zusammenfassung der Zielvariable

        # Boruta-Algorithmus ausführen
        rf = RandomForestRegressor(n_jobs=-1, random_state=42)          # Initialisiere Random Forest als Basis für Boruta
        boruta = BorutaPy(estimator=rf, n_estimators='auto', random_state=42)# Boruta-Objekt erstellen
        boruta.fit(X.values, y.values)                                  # Boruta-Feature-Auswahl durchführen

        # Ausgewählte Features speichern
        self.selected_features = X.columns[boruta.support_].tolist()    # Features, die Boruta ausgewählt hat
        print(f"Ausgewählte Features für Regression: {self.selected_features}")
        return self.selected_features


if __name__ == "__main__":
    # API-basierte Datenerfassung (echte Daten)
    import krakenex
    import requests
    import time

    def fetch_data(pair="XXBTZUSD", interval=60, days=30):
        """
        Ruft historische Daten von der Kraken-API ab und bereitet sie vor.
        :param pair: Handelspaar (z. B. Bitcoin/USD)
        :param interval: Zeitintervall in Minuten
        :param days: Anzahl der Tage, für die Daten abgerufen werden sollen
        :return: Pandas DataFrame mit OHLC-Daten und berechneten Features
        """
        api = krakenex.API()
        api.key = 'w3if4ZjPEKdgCVsj7J/KVRgkSKhAhYBcJJrrp8gXTfrRdlylAVafK85F'

        url = "https://api.kraken.com/0/public/OHLC"                        # API-Endpunkt für historische OHLC-Daten
        params = {
            'pair': pair,
            'interval': interval,
            'since': int(time.time()) - 60 * 60 * 24 * days,                # Startzeit basierend auf der Anzahl der Tage
        }

        response = requests.get(url, params=params)                         # Anfrage an die API senden
        data = response.json()                                              # Antwort in JSON konvertieren

        if len(data['error']) == 0:                                         # Prüfen, ob keine Fehler vorliegen
            ohlc = data['result'][pair]
            df = pd.DataFrame(
                ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            )
            df['time'] = pd.to_datetime(df['time'], unit='s')               # Zeitstempel in Datetime-Format konvertieren

            # Konvertiere Spalten zu numerischen Werten
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
                df[col] = pd.to_numeric(df[col], errors='coerce')           # Fehlerhafte Werte in NaN umwandeln
            df = df.dropna()

            # Zusätzliche Features berechnen
            df['rsi'] = calculate_rsi(df['close'])                              # Relative Strength Index berechnen
            df['ma_20'] = calculate_moving_average(df['close'], 20)     # 20-Tage gleitender Durchschnitt
            df['ma_50'] = calculate_moving_average(df['close'], 50)     # 50-Tage gleitender Durchschnitt
            df['volatility'] = calculate_volatility(df['close'], 20)    # Volatilität basierend auf logarithmischen Renditen

            # Entferne Zeilen mit NaN-Werten
            return df.dropna()
        else:
            raise Exception(f"API-Fehler: {data['error']}")

    def calculate_rsi(data, window=14):
        """
        Berechnet den RSI.
        """
        delta = data.diff()                                                     # Differenzen zwischen den Schlusskursen berechnen
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()        # Durchschnittliche Gewinne
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()       # Durchschnittliche Verluste
        rs = gain / loss                                                        # Verhältnis von Gewinnen zu Verlusten
        return 100 - (100 / (1 + rs))                                           # RSI-Wert berechnen

    def calculate_moving_average(data, window):
        """
        Berechnet gleitenden Durchschnitt.
        :param data: Pandas Series
        :param window: Fenstergröße für den gleitenden Durchschnitt
        :return: Gleitender Durchschnitt
        """
        return data.rolling(window=window).mean()                               # Durchschnitt der letzten 'window'-Werte

    def calculate_volatility(data, window=20):
        """
        Berechnet Volatilität.
        :param data: Pandas Series mit Schlusskursen
        :param window: Fenstergröße für die Volatilitätsberechnung
        :return: Volatilität
        """
        log_returns = np.log(data / data.shift(1))                              # Logarithmische Renditen berechnen
        return log_returns.rolling(window=window).std() * np.sqrt(window)       # Standardabweichung der Renditen

    # Daten abrufen
    df = fetch_data()

    # Regression Feature-Selection
    reg_selector = RegressionFeatureSelector(data=df)                           # Initialisiere den Feature-Selector mit den Daten
    reg_features = reg_selector.run_boruta()                                    # Boruta-Feature-Auswahl durchführen

    # Endergebnis ausgeben
    print("\nFinale Ergebnisse:")
    print(f"Regression Features: {reg_features}")


        # BORUTA WÄHLT FOLGENDE FEATURES:
        # open  = Eröffnungswert der Periode
        # high  = Höchstwert der jeweiligen Periode
        # low   = Tiefstwert der jeweiligen Periode
        # close = Schlusswer der jeweiligen Periode (Zielvariable ist stark korreliert mit diesen Features)
        # rsi   = Relative Strength Index, ein technischer Indikator zur Bewertung von Preisbewegungen
        # ma_20 = Gleitender Durchschnitt der Schlusskurse über die letzten 20 Perioden, wird verwendet, um kurzfristige Trends zu identifizieren und Marktbewegungen zu glätten.
