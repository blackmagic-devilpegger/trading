
##########Experiment 1: Hyperparameteranpassung
Short Description
In diesem Experiment wurde ein Recurrent Neural Network (RNN) entwickelt, um den Schlusskurs von Bitcoin basierend auf historischen Daten vorherzusagen. 
Das Modell wurde mit verschiedenen Optimierungen und Metriken evaluiert, um die Vorhersagegenauigkeit und Robustheit der Handelsstrategien zu verbessern.

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
Die Daten wurden normalisiert, um die Eingabewerte an das Modell anzupassen.

-->Target
Ziel: Aufbauen einer Vorhersage des Schlusskurses (Close) für die nächste Zeiteinheit basierend auf historischen Sequenzen.

-->Modeling Architecture
Modell: Recurrent Neural Network (RNN)
Architektur:
Input Layer: 4 Merkmale (Open, High, Low, Close).
Hidden Layer: 50 Neuronen (RNN-Schicht).
Output Layer: 1 Neuron (Vorhersage des nächsten Schlusskurses).
Optimierungen:
Lernrate: Reduziert auf 0.0005, um Overfitting zu vermeiden.
Early Stopping: Geduld (patience) auf 10 Epochen erhöht.

-->Performance Criteria
Metrik: Mean Absolute Deviation (MAD)
        MAD of Predictions: 7146.50 USD
        MAD of Average Price (Baseline): 2992.80 USD

Trainings- und Validierungsverluste:
        Training Loss nach Epoche 50:   0.000826
        Validation Loss nach Epoche 50: 0.000636

-->Baseline
Durchschnittspreis: Der Mittelwert der Schlusskurse wurde als einfache Baseline verwendet.
Leistung der Baseline:
    MAD (Mean Absolute Deviation): 2992.80 USD
Modell im Vergleich zur Baseline:
    Das Modell konnte Schwankungen besser erfassen, hatte jedoch einen höheren MAD-Wert (7146.50 USD), was auf Verbesserungspotenziale hindeutet.

-->Results
Modellauswertung:
RNN_1:
Model MSE: 50,540,841.85 USD
Model MAE: 7047.94 USD
Model MAD: 7047.94 USD

RNN_2:
Model MSE: 1,566,547.71 USD
Model MAE: 939.28 USD
Model MAD: 6901.52 USD

Modellbewertung und Vergleich (RNN_2 gegenüber RNN_1):
 -Modellverbesserung (zu RNN_1): Das Modell RNN_2 zeigt eine deutliche Verbesserung der Genauigkeit mit signifikant niedrigeren MSE- und MAE-Werten im Vergleich zu RNN_1.
 -Baseline-Konsistenz: Die MAD-Werte von RNN_2 nähern sich der Baseline an, jedoch bleibt das Modell in seiner Performance auf demselben Niveau, 
    ohne die Baseline signifikant zu übertreffen.
-Leistungslücke:
Der große Unterschied zwischen den Modellen (RNN_1 vs. RNN_2) unterstreicht die Fortschritte bei der Optimierung. Dennoch zeigt RNN_2, dass weitere
    Anpassungen erforderlich sind, um die Baseline nachhaltig zu übertreffen.

Fazit:
RNN_2 bietet erhebliche Verbesserungen gegenüber RNN_1 und weist auf eine stabilere Modellperformance hin. Dennoch bleibt die Herausforderung bestehen, 
eine höhere Vorhersagegenauigkeit zu erreichen, die die Baseline nachhaltig übertrifft.



