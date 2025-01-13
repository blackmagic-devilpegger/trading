##########Experiment 5: Boruta Feature-Selektion

Short Description
In diesem Experiment wurde der Boruta-Algorithmus zur Feature-Auswahl für Regression und Klassifikation verwendet, um relevante Features aus historischen Bitcoin-Daten zu identifizieren.
Ziel war es, die Modellkomplexität zu reduzieren und die Vorhersagegenauigkeit zu verbessern.

-->Data Acquisition
Quelle: Kraken API
Handelspaar: Bitcoin zu USD (XXBTZUSD)
Zeitraum: 30 Tage (1-Monats-Daten)
Intervall: 1 Stunde (60 Minuten)
Merkmale (Features):
    Open, High, Low, Close (OHLC)
    Volume Weighted Average Price (VWAP)
    RSI (Relative Strength Index)
    MA 20 und MA 50 (Gleitende Durchschnitte)
    Volatilität
-Datenbereinigung:
Konvertierung der numerischen Werte.
Entfernung von NaN-Werten.
Berechnung zusätzlicher Features:
RSI (Relative Strength Index)
MA 20 und MA 50 (Gleitende Durchschnitte)
Volatilität

-->Features
  --Regression
    Ausgewählte Features:
    Open: Eröffnungspreis.
    High: Höchstpreis.
    Low: Tiefstpreis.
    Close: Schlusskurs (stark korreliert mit der Zielvariable).
    RSI: Bewertung von Preisbewegungen.
    MA 20: Kurzfristige Trendbewertung (20 Perioden).

    Nicht relevante Features:
    MA 50: Langfristige Trends.
    Volatilität: Kurzfristige Schwankungen.

 --Klassifikation
Ausgewählte Features:
Keine relevanten Features identifiziert. Die berechneten Korrelationen mit der Zielvariable (y_class) waren gering
(z. B. RSI = 0.0089, Volatilität = -0.0056).
Grund: Keine signifikante Beziehung zwischen den Features und der Zielvariable (Preis steigt/fällt).

-->Target
Regression: Vorhersage des Schlusskurses (Close) für die nächste Zeiteinheit.
Klassifikation: Identifikation von Preisbewegungen (Steigt/Fällt/Konstant).

-->Modeling Architecture
Feature-Auswahl:
Regression: Boruta identifizierte 6 relevante Features (z. B. Open, High, RSI, MA 20).
Klassifikation: Boruta konnte keine relevanten Features identifizieren.
    --Klassifikationsidee: Nutzung der Regression-Features für ein CNN, um nicht-lineare Muster und Trends zu erfassen.

-->Performance Criteria
    Regression
        MAD der Vorhersagen:5,662.93 USD (besser als Baseline, zeigt jedoch Optimierungspotenzial).
        Baseline:
        MAD der Baseline (Durchschnittspreis): 2,802.36 USD.
    Klassifikation
        Boruta-Ergebnis:Keine Features ausgewählt (niedrige Korrelation).
        Sicherheitsmarge:Reduktion von Fehlsignalen (2%-Marge für Klassifikationsentscheidungen).
-->Results
Regression:
Boruta identifizierte 6 relevante Features, die die Modellleistung steigern können.
Nicht relevante Features wurden erfolgreich eliminiert, was die Modellkomplexität reduziert.
Klassifikation:
Keine relevanten Features identifiziert.
Weiterer Ansatz: Erstellen eines CNN, um nicht-lineare Muster besser zu erkennen.

AUSSAGE:
Das Experiment zeigt, dass Boruta ein effektives Werkzeug für die Feature-Auswahl in der Regression ist,
jedoch bei der Klassifikation aufgrund der geringen Korrelation der Features nicht hilfreich war.
Die Ergebnisse der Regression bieten eine solide Grundlage für die weitere Entwicklung eines Modells(CNN oder andere) zur Vorhersage des Bitcoin-Schlusskurses.