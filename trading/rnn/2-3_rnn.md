##########Experiment 2: Erweiterung durch zusätzliche Optimierungen und Klassifikation

Short Description
RNN-Modell weiterentwickelt, um sowohl die Regression des Schlusskurses als auch Klassifikationsaufgaben (Preissteigerung, -senkung, -stabilität) zu integrieren.
Verschiedene Optimierungen und Regularisierungsmethoden wurden hinzugefügt, um die Generalisierungsfähigkeit zu verbessern und robustere Handelsstrategien zu entwickeln.

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
-Normalisierung:
   Mittelwert-Standardabweichungs-Normalisierung (z-Score) zur Stabilisierung des Trainingsprozesses.

-->Target
Ziel: Die Klassifikation teilt die Daten in drei Kategorien ein:
1 bedeutet, dass der Preis steigt, -1 zeigt einen fallenden Preis an, und 0 steht für einen konstanten Preis,
wobei eine Sicherheitsmarge von ±2 % berücksichtigt wird.


-->Modeling Architecture
Modell: Recurrent Neural Network (RNN) + Dropoutlayer
Architektur:
Input Layer: 4 Merkmale (Open, High, Low, Close).
Hidden Layer: 50 Neuronen (RNN-Schicht).
Dropout Layer: 10 % zur Reduzierung von Overfitting
Output Layer: 1 Neuron (Vorhersage des nächsten Schlusskurses).
Optimierungen: Mean/std
Integration von Klassifikationsmodellen zur Entscheidungsunterstützung.


-->Performance Criteria
Metrik: Mean Absolute Deviation (MAD)
            MAD of Predictions: 7198.65 USD
            MAD of Average Price (Baseline): 2989.27 USD
        R²-Score: Bis zu 0.957, zeigt hohe Übereinstimmung der Regression mit tatsächlichen Werten.
Trainings- und Validierungsverluste:
        Training Loss nach Epoche 50:   0.000849
        Validation Loss nach Epoche 50: 0.000675

-->Baseline
Durchschnittspreis: Der Mittelwert der Schlusskurse wurde als einfache Baseline verwendet.
Leistung der Baseline:
    MAD (Mean Absolute Deviation): 2989.27 USD
Modell im Vergleich zur Baseline:
    Das Modell hat zwar einen höheren MAD-Wert als die Baseline, erfasst jedoch komplexere Schwankungen und Trends.

-->Results
-Trainings- und Validierungsverluste: Beide Verluste nehmen stetig ab, ohne plötzliche Anstiege.
                                      Finaler Validation Loss: 0.000675.

-Vorhersagegenauigkeit: Rücknormalisierte Vorhersagen (z. B. [102120.31, 102045.63, 102141.09]]) stimmen teilweise mit den tatsächlichen Preisen überein,
    MAD der Vorhersagen zeigt jedoch Verbesserungspotenzial:(7198.65 USD)
-Handelsstrategien: Klassifikation ergänzt Regression mit klaren Kauf-/Verkauf-/Keine-Aktion-Entscheidungen.

-->Results
Modellauswertung:
RNN_2:
MSE: 1,566,547.71 USD
MAD der Vorhersagen: 6,901.52 USD
MAD des Durchschnittspreises: 2,942.34 USD
MAE: 939.28 USD

RNN_3:
MSE: 2,799,020.35 USD
MAD der Vorhersagen: 7,329.17 USD
MAD des Durchschnittspreises: 2,942.54 USD
MAE: 1,331.25 USD
R²-Wert: 0.8576 (zeigt, dass das Modell eine gute Anpassung hat, jedoch Optimierungspotenzial besteht)

Modellbewertung und Vergleich (RNN_3 gegenüber RNN_2):
Das Modell RNN_2 zeigt insgesamt bessere Metriken mit einem niedrigeren MSE und MAD sowie einem kleineren MAE im Vergleich zu RNN_3.
Dies deutet auf präzisere und stabilere Vorhersagen hin. RNN_3 weist hingegen eine höhere Abweichung (MAD) und größere Fehler (MAE) auf,
was auf stärkere Schwankungen in den Vorhersagen hindeutet.

Ein Vorteil von RNN_3 ist die Ergänzung um den R²-Wert, der mit 0.8576 eine solide Anpassung des Modells an die Daten zeigt.
Dennoch schneidet RNN_3 in den anderen Metriken schlechter ab, was weiteren Optimierungsbedarf, insbesondere zur Reduktion von Schwankungen, nahelegt.

Fazit:
Insgesamt liefert RNN_2 stabilere und präzisere Ergebnisse, während RNN_3 durch zusätzliche Features und eine detaillierte Bewertung punktet,
jedoch in der Gesamtleistung zurückbleibt.




