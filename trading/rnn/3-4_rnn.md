
##########Experiment 3: Analyse der Rücknormalisierung (MinMaxScaler), und Auswirkung auf Vorhersagen

Short Description:
Untersucht die Auswirkungen der Normalisierung und Rücknormalisierung auf die Vorhersagequalität eines RNN-Modells. Die Normalisierung der Daten erfolgt mit einem MinMaxScaler, um die Eingabewerte in einen festgelegten Bereich (z. B. [0, 1]) zu skalieren. Ziel ist es,
die Stabilität des Modells zu erhöhen. Es wird außerdem überprüft, ob die Rücknormalisierung korrekt implementiert ist und realistische Vorhersagen liefert.

-->Data Acquisition
Quelle: Kraken API
Handelspaar: Bitcoin zu USD (XXBTZUSD)
Zeitraum: 30 Tage (1-Monats-Daten)
Intervall: 1 Stunde (60 Minuten)
Merkmale (Features):
Open, High, Low, Close (OHLC)
Volume Weighted Average Price (VWAP)
Volume
Trade Count
-Datenbereinigung:
Konvertierung der numerischen Werte.
Entfernung von NaN-Werten.

Features
-->Die folgenden Features wurden verwendet:
Open: Eröffnungspreis.
High: Höchstpreis.
Low: Tiefstpreis.
Close: Schlusskurs (Zielwert für die Vorhersage).
Die Daten wurden mit einem MinMaxScaler normalisiert, um die Werte in einem festen Bereich zu skalieren.

-->Target
Ziel: Aufbauen einer verbesserten Vorhersage des Schlusskurses (Close) für die nächste Zeiteinheit basierend auf historischen Sequenzen.

-->Modeling Architecture
Modell: Recurrent Neural Network (RNN)
Architektur:
Input Layer: 4 Merkmale (Open, High, Low, Close).
Hidden Layer: 50 Neuronen (RNN-Schicht).
Output Layer: 1 Neuron (Vorhersage des nächsten Schlusskurses).
Optimierungen: MinMaxScaler

-->Performance Criteria
Metrik: Mean Absolute Deviation (MAD)
        MAD of Predictions:  94,048.18 USD
        MAD of Average Price (Baseline): 2,989.54 USD

Trainings- und Validierungsverluste:
        Training Loss nach Epoche 50:   0.000097
        Validation Loss nach Epoche 50: 0.000037

-->Baseline
Durchschnittspreis: Der Mittelwert der Schlusskurse wurde als einfache Baseline verwendet.
Leistung der Baseline:
    MAD (Mean Absolute Deviation): 2989.54 USD
Modell im Vergleich zur Baseline:
    Das Modell erreichte eine MAD von 94,048.18 USD, deutlich schlechter als die Baseline.
    Trotz eines niedrigen Validation Loss (0.000037) führten fehlerhafte Skalierungs- und Rücknormalisierungsprozesse
    zu unrealistischen Vorhersagen (z. B. -5.5167 USD).

-->Results
Modellauswertung:
RNN_3:
MSE: 2,799,020.35 USD
MAD der Vorhersagen: 7,329.17 USD
MAD des Durchschnittspreises: 2,942.54 USD
MAE: 1,331.25 USD
R²-Wert: 0.8576

RNN_4:
MSE: 8,726,656,275.85 USD
MAD der Vorhersagen: 93,415.52 USD
MAD des Durchschnittspreises: 2,942.53 USD
MAE: 93,415.52 USD
R²-Wert: 0.8335

Modellbewertung und Vergleich (RNN_4 gegenüber RNN_3):
RNN_4 zeigt einen deutlichen Anstieg des MSE sowie die höchste Abweichung der Vorhersagen (MAD: 93,415.52 USD) und stark erhöhte absolute Fehler (MAE)
im Vergleich zu RNN_3. Trotz eines R²-Werts von 0.8335 bleibt die Modellanpassung schlechter als bei RNN_3. Insgesamt liefert RNN_3 stabilere und
präzisere Ergebnisse, während RNN_4 durch fehlerhafte Skalierung und unrealistische Rücknormalisierungen deutliche Schwächen aufweist.
Eine Optimierung der Normalisierung ist erforderlich, um die Leistung von RNN_4 zu verbessern.

Fazit:
Das Modell RNN_3 liefert insgesamt bessere Ergebnisse mit niedrigeren Fehlerwerten (MSE, MAD, MAE) und einem soliden R²-Wert.

