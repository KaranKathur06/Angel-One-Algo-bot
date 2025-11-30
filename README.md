# SMARTAPI-PYTHON

SMARTAPI-PYTHON is a Python library for interacting with Angel One's Trading platform, that is a set of REST-like HTTP APIs that expose many capabilities required to build stock market investment and trading platforms. It lets you execute orders in real time.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install smartapi-python.

```bash
pip install -r requirements.txt       # for downloading the other required packages
```

## Environment configuration

Secrets such as the SmartAPI key, client ID, password, and TOTP seed
must never be committed to source control. Copy `env.example` to a local
`.env` file and fill in your credentials:

```bash
cp env.example .env
```

The bot automatically loads `.env` (via `python-dotenv`) and overrides
the `smart_api` section in `config.json`, so you can keep that file in
version control with blank placeholders.

## GUI trading bot

The bot supports two GUI options:

### Option 1: Streamlit Web GUI (Recommended for servers)

Perfect for headless servers and remote access. Runs as a web application accessible via browser.

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501` (or your server's IP:8501).

**For production servers**, you can run it with custom host/port:

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Option 2: Tkinter Desktop GUI (Local use only)

For local desktop use (requires X11 display). Launch it with:

```bash
python gui_trading_bot.py
```

**Note:** Tkinter requires a display server and won't work on headless servers. Use Streamlit for server deployments.

### GUI pages

Both GUIs provide the same functionality:

- **Dashboard** – start/stop the threaded strategy, monitor status, and view
  live logs without blocking the UI.
- **Instruments** – add, edit, remove, and review all configured symbols with
  RSI/target/stop parameters and intervals.
- **Positions** – live snapshot of open positions with entry/current price,
  P&L%, timestamps, and strategy context.
- **Settings** – inspect or edit `config.json`, reload from disk, or force-save
  the in-memory configuration.

All changes made via the GUI persist back to `config.json`, and SmartAPI
credentials remain sourced from `.env`.

### Additional Required Packages

Download the following packages:

```bash
pip install pyotp
pip install logzero
pip install websocket-client
```

For downloading pycryptodome package:

```bash
pip uninstall pycrypto
pip install pycryptodome
```

## Usage

```python
# package import statement
from SmartApi import SmartConnect #or from SmartApi.smartConnect import SmartConnect
import pyotp
from logzero import logger

api_key = 'Your Api Key'
username = 'Your client code'
pwd = 'Your pin'
smartApi = SmartConnect(api_key)
try:
    token = "Your QR value"
    totp = pyotp.TOTP(token).now()
except Exception as e:
    logger.error("Invalid Token: The provided token is not valid.")
    raise e

correlation_id = "abcde"
data = smartApi.generateSession(username, pwd, totp)

if data['status'] == False:
    logger.error(data)
    
else:
    # login api call
    # logger.info(f"You Credentials: {data}")
    authToken = data['data']['jwtToken']
    refreshToken = data['data']['refreshToken']
    # fetch the feedtoken
    feedToken = smartApi.getfeedToken()
    # fetch User Profile
    res = smartApi.getProfile(refreshToken)
    smartApi.generateToken(refreshToken)
    res=res['data']['exchanges']

    #place order
    try:
        orderparams = {
            "variety": "NORMAL",
            "tradingsymbol": "SBIN-EQ",
            "symboltoken": "3045",
            "transactiontype": "BUY",
            "exchange": "NSE",
            "ordertype": "LIMIT",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": "19500",
            "squareoff": "0",
            "stoploss": "0",
            "quantity": "1"
            }
        # Method 1: Place an order and return the order ID
        orderid = smartApi.placeOrder(orderparams)
        logger.info(f"PlaceOrder : {orderid}")
        # Method 2: Place an order and return the full response
        response = smartApi.placeOrderFullResponse(orderparams)
        logger.info(f"PlaceOrder : {response}")
    except Exception as e:
        logger.exception(f"Order placement failed: {e}")

    #gtt rule creation
    try:
        gttCreateParams={
                "tradingsymbol" : "SBIN-EQ",
                "symboltoken" : "3045",
                "exchange" : "NSE", 
                "producttype" : "MARGIN",
                "transactiontype" : "BUY",
                "price" : 100000,
                "qty" : 10,
                "disclosedqty": 10,
                "triggerprice" : 200000,
                "timeperiod" : 365
            }
        rule_id=smartApi.gttCreateRule(gttCreateParams)
        logger.info(f"The GTT rule id is: {rule_id}")
    except Exception as e:
        logger.exception(f"GTT Rule creation failed: {e}")
        
    #gtt rule list
    try:
        status=["FORALL"] #should be a list
        page=1
        count=10
        lists=smartApi.gttLists(status,page,count)
    except Exception as e:
        logger.exception(f"GTT Rule List failed: {e}")

    #Historic api
    try:
        historicParam={
        "exchange": "NSE",
        "symboltoken": "3045",
        "interval": "ONE_MINUTE",
        "fromdate": "2021-02-08 09:00", 
        "todate": "2021-02-08 09:16"
        }
        smartApi.getCandleData(historicParam)
    except Exception as e:
        logger.exception(f"Historic Api failed: {e}")
    #logout
    try:
        logout=smartApi.terminateSession('Your Client Id')
        logger.info("Logout Successful")
    except Exception as e:
        logger.exception(f"Logout failed: {e}")
```

## Getting started with SmartAPI Websockets

### Websocket V2 sample code

```python
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from logzero import logger

AUTH_TOKEN = "authToken"
API_KEY = "api_key"
CLIENT_CODE = "client code"
FEED_TOKEN = "feedToken"
correlation_id = "abc123"
action = 1
mode = 1

token_list = [
    {
        "exchangeType": 1,
        "tokens": ["26009"]
    }
]
token_list1 = [
    {
        "action": 0,
        "exchangeType": 1,
        "tokens": ["26009"]
    }
]

sws = SmartWebSocketV2(AUTH_TOKEN, API_KEY, CLIENT_CODE, FEED_TOKEN)

def on_data(wsapp, message):
    logger.info("Ticks: {}".format(message))
    # close_connection()

def on_open(wsapp):
    logger.info("on open")
    sws.subscribe(correlation_id, mode, token_list)
    # sws.unsubscribe(correlation_id, mode, token_list1)


def on_error(wsapp, error):
    logger.error(error)


def on_close(wsapp):
    logger.info("Close")


def close_connection():
    sws.close_connection()


# Assign the callbacks.
sws.on_open = on_open
sws.on_data = on_data
sws.on_error = on_error
sws.on_close = on_close

sws.connect()
```

### SmartWebSocket OrderUpdate Sample Code

```python
from SmartApi.smartWebSocketOrderUpdate import SmartWebSocketOrderUpdate
client = SmartWebSocketOrderUpdate(AUTH_TOKEN, API_KEY, CLIENT_CODE, FEED_TOKEN)
client.connect()
```
## Changelog
### 1.4.5
- Upgraded TLS Version

### 1.4.7
- Added Error log file

### 1.4.8
- Integrated EDIS, Brokerage Calculator, Option Greek, TopGainersLosers, PutRatio API
