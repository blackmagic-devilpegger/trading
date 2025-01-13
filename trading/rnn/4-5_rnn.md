##########Experiment 4: Analyse der Erweiterung durch Feature-Engineering und Z-Score-Normalisierung

Short Description:
Dieses Experiment untersucht die Integration zusätzlicher Features wie Relative Strength Index (RSI), gleitende Durchschnitte (MA 20, MA 50) und Volatilität. Ziel war es, die Modellleistung durch erweiterte Marktanalysen zu verbessern.
Zusätzlich wurde Z-Score-Normalisierung verwendet, um die Eingabedaten zu stabilisieren und den Trainingsprozess zu optimieren.

-->Data Acquisition
Quelle: Kraken API
Handelspaar: Bitcoin zu USD (XXBTZUSD)
Zeitraum: 30 Tage (1-Monats-Daten)
Intervall: 1 Stunde (60 Minuten)
Merkmale (Features):
Open, High, Low, Close (OHLC)
RSI: Relative Strength Index zur Trendanalyse.
MA 20, MA 50: Kurz- und langfristige gleitende Durchschnitte.
Volatilität: Berechnet auf Basis logarithmischer Renditen.
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
Input Layer:  8 Merkmale (Open, High, Low, Close, RSI, MA 20, MA 50, Volatilität).
Hidden Layer: 128 Neuronen ( 2RNN-Schichten).
Output Layer: 1 Neuron (Vorhersage des nächsten Schlusskurses).
Optimierungen: Dropout 0.2 zur Regularisierung

-->Performance Criteria
Metrik: Mean Absolute Deviation (MAD)
        MAD of Predictions:  5,662.93 USD
        MAD of Average Price (Baseline): 2,802.36 USD

Trainings- und Validierungsverluste:
        Training Loss nach Epoche 22:   0.001885
        Validation Loss nach Epoche 22:  0.022561

-->Baseline
Durchschnittspreis: Der Durchschnittspreis wurde als einfache Baseline verwendet.
    MAD (Mean Absolute Deviation): 2,802.36 USD
Modell im Vergleich zur Baseline:
Das Modell erreichte eine MAD von 5,662.93 USD, was über der Baseline liegt, jedoch zeigt der kontinuierlich sinkende Validation Loss eine verbesserte Generalisierung im Vergleich zu vorherigen Experimenten.

-->Results
Modellauswertung:
RNN_4:
MSE: 8,726,656,275.85 USD
MAD der Vorhersagen: 93,415.52 USD
MAD des Durchschnittspreises: 2,942.53 USD
MAE: 93,415.52 USD
R²-Wert: 0.8335

RNN_5:
MSE: 12,719,740.40 USD
MAD der Vorhersagen: 3,527.29 USD
MAD des Durchschnittspreises: 2,675.38 USD
MAE: 3,527.29 USD
R²-Wert: 0.739

Modellbewertung und Vergleich (RNN_5 gegenüber RNN_4):

RNN_5 zeigt im Vergleich zu RNN_4 eine deutliche Verbesserung, insbesondere bei den Fehlerwerten. Der MSE von RNN_5 ist realistischer,
auch wenn er noch relativ hoch bleibt. Die mittlere absolute Abweichung (MAD) von RNN_5 liegt bei 3,527.29 USD und ist damit signifikant
niedriger als die von RNN_4 (93,415.52 USD), was auf eine wesentlich höhere Vorhersagegenauigkeit hinweist.
Auch die absoluten Fehler (MAE) von RNN_5 sind deutlich geringer, was die Qualität der Vorhersagen unterstreicht.
Obwohl der R²-Wert bei RNN_5 etwas niedriger ist als bei RNN_4, überzeugt das Modell durch realistischere und konsistentere Ergebnisse

Fazit:
Das Modell RNN_5 überzeigt deutlich  ggü dem Modell RNN_4 durch stabilere und realistischere Vorhersagen, unterstützt durch
verbessertes Feature-Engineering und präzisere Rücknormalisierung. Die weiterhin hohen Fehlerwerte zeigen jedoch Optimierungsbedarf
