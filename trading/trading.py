from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import pandas as pd
from datetime import datetime

# API URL für historische Daten
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical'

# Parameter für die Anfrage: Zeitintervall und Währung
parameters = {
    'symbol': 'BTC',  # Symbol für Bitcoin
    'convert': 'USD',  # Umrechnung in US-Dollar
    'time_start': '2023-01-01',  # Startdatum für historische Daten (Format: YYYY-MM-DD)
    'time_end': '2024-01-01',  # Enddatum für historische Daten (Format: YYYY-MM-DD)
    'interval': 'daily'  # Daten im täglichen Intervall
}

# Headers für die API-Anfrage
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': '0102f7e8-ebe8-4da1-8be1-1526eaabba4f',  # Dein API-Schlüssel
}
# Session für die Anfrage
session = Session()
session.headers.update(headers)

try:
    # Anfrage an die API senden
    response = session.get(url, params=parameters)

    # Überprüfen, ob die Anfrage erfolgreich war
    if response.status_code == 200:
        # Antwort im JSON-Format verarbeiten
        data = json.loads(response.text)

        # Die historischen Daten extrahieren (Bitcoin-Kurs pro Tag)
        quotes = data['data']['quotes']

        # Umwandlung der Daten in ein DataFrame
        df = pd.DataFrame(quotes)

        # Konvertiere den Zeitstempel in lesbares Datum
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Wähle die relevanten Spalten aus (Datum und Preis in USD)
        df = df[['timestamp', 'quote']['USD']['price']]

        # Zeige die ersten 5 Zeilen des DataFrames an
        print(df.head())

        # Optional: Speichere die Daten als CSV-Datei
        df.to_csv('historical_bitcoin_data.csv', index=False)

    else:
        print(f"Fehler bei der API-Anfrage: {response.status_code}")
except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)
