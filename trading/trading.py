import krakenex

api = krakenex.API()
api.key = 'w3if4ZjPEKdgCVsj7J/KVRgkSKhAhYBcJJrrp8gXTfrRdlylAVafK85F'
api.secret = 'zYOQHH+XVOXsHGddoDKEbUL8JkB3mvHdRjZSP4QLqRV5wkwDZ4iELkfXOwneDWfTTBrHYQkoc8hgLtS1u+rYlg=='

response = api.query_public('Ticker', {'pair': 'XXBTZUSD'})
print(response['result'])

import requests
import time
import pandas as pd

# API-Endpunkt für historische Daten
url = "https://api.kraken.com/0/public/OHLC"

# Parameter für die API
params = {
    'pair': 'XXBTZUSD',  # Bitcoin (XBT) zu US-Dollar (USD)
    'interval': 60,  # Zeitintervall (z. B. 60 Minuten)
    'since': int(time.time()) - 60 * 60 * 24 * 30  # Daten der letzten 30 Tage
}

# API-Abfrage
response = requests.get(url, params=params)
data = response.json()

if len(data['error']) == 0:  # Keine Fehler
    ohlc = data['result']['XXBTZUSD']  # Daten für BTC/USD-Paar
    df = pd.DataFrame(
        ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    )
    df['time'] = pd.to_datetime(df['time'], unit='s')  # Zeit in Datetime umwandeln
    print(df.head())
else:
    print("Error:", data['error'])

