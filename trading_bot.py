import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Import SmartApi with better error handling
SmartConnect = None
try:
    from SmartApi import SmartConnect  # type: ignore
except ImportError:
    # Try alternative import path silently (will retry at runtime)
    try:
        from SmartApi.smartConnect import SmartConnect  # type: ignore
    except ImportError:
        SmartConnect = None

# Import pyotp with better error handling
pyotp = None
try:
    import pyotp  # type: ignore
except ImportError:
    pyotp = None

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.json")
DEFAULT_INSTRUMENTS = [
    {
        "symbol": "NIFTY",
        "timeframe": "5m",
        "rsi_length": 14,
        "rsi_level": 30,
        "profit_target_pct": 5.0,
        "stop_loss_pct": 10.0,
        "quantity": 1,
        "exchange": "NFO",
        "symbol_token": "99926000",
        "product_type": "INTRADAY",
    },
    
]

SMART_API_DEFAULTS = {
    "enabled": False,
    "api_key": "",
    "client_id": "",
    "password": "",
    "totp_secret": "",
}

# SmartAPI timeframe mapping
SMARTAPI_TIMEFRAME_MAP: Dict[str, str] = {
    "1m": "ONE_MINUTE",
    "3m": "THREE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
}

TIMEFRAME_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "1m": ("1m", None),
    "3m": ("1m", "3min"),
    "5m": ("1m", "5min"),
    "15m": ("15m", None),
    "30m": ("30m", None),
    "45m": ("15m", "45min"),
    "1h": ("60m", None),
    "2h": ("60m", "120min"),
}


@dataclass
class InstrumentConfig:
    symbol: str
    timeframe: str = "5m"
    rsi_length: int = 14
    rsi_level: float = 30.0
    profit_target_pct: float = 5.0
    stop_loss_pct: float = 10.0
    quantity: int = 1
    exchange: str = "NSE"
    symbol_token: Optional[str] = None
    product_type: str = "INTRADAY"


@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: int
    target_price: float
    stop_price: float
    timestamp: float
    order_id: Optional[str] = None
    exchange: str = "NSE"
    product_type: str = "INTRADAY"


class ConfigManager:
    def __init__(self, path: Path, logger: Optional[Callable[[str], None]] = None):
        self.path = path
        self.logger = logger or print
        self._config = {
            "instruments": DEFAULT_INSTRUMENTS.copy(),
            "smart_api": SMART_API_DEFAULTS.copy(),
        }
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.save()
            return
        try:
            self._config = json.loads(self.path.read_text())
            if "smart_api" not in self._config:
                self._config["smart_api"] = SMART_API_DEFAULTS.copy()
            if "instruments" not in self._config:
                self._config["instruments"] = DEFAULT_INSTRUMENTS.copy()
        except json.JSONDecodeError:
            self.logger("Config file is corrupted. Regenerating defaults.")
            self._config = {
                "instruments": DEFAULT_INSTRUMENTS.copy(),
                "smart_api": SMART_API_DEFAULTS.copy(),
            }
            self.save()
        
        # Load SmartAPI credentials from environment variables
        self._load_smartapi_from_env()

    def save(self) -> None:
        self.path.write_text(json.dumps(self._config, indent=2))

    def _load_smartapi_from_env(self) -> None:
        """Load SmartAPI credentials from environment variables."""
        smart_api_config = self._config.get("smart_api", {})
        
        # Load from environment variables
        smart_api_config["enabled"] = os.getenv("SMART_API_ENABLED", "false").lower() == "true"
        smart_api_config["api_key"] = os.getenv("SMART_API_KEY", smart_api_config.get("api_key", ""))
        smart_api_config["client_id"] = os.getenv("SMART_API_CLIENT_ID", smart_api_config.get("client_id", ""))
        smart_api_config["password"] = os.getenv("SMART_API_PASSWORD", smart_api_config.get("password", ""))
        smart_api_config["totp_secret"] = os.getenv("SMART_API_TOTP_SECRET", smart_api_config.get("totp_secret", ""))
        
        self._config["smart_api"] = smart_api_config

    def get_instruments(self) -> List[InstrumentConfig]:
        return [InstrumentConfig(**cfg) for cfg in self._config.get("instruments", [])]

    def upsert_instrument(self, config: InstrumentConfig) -> None:
        instruments = self._config.setdefault("instruments", [])
        for idx, existing in enumerate(instruments):
            if existing["symbol"].upper() == config.symbol.upper():
                instruments[idx] = asdict(config)
                self.save()
                return
        instruments.append(asdict(config))
        self.save()

    def remove_instrument(self, symbol: str) -> bool:
        instruments = self._config.setdefault("instruments", [])
        filtered = [inst for inst in instruments if inst["symbol"].upper() != symbol.upper()]
        if len(filtered) == len(instruments):
            return False
        self._config["instruments"] = filtered
        self.save()
        return True

    def get_smart_api_config(self) -> Dict[str, Any]:
        return self._config.get("smart_api", SMART_API_DEFAULTS.copy())

    def update_smart_api_config(self, config: Dict[str, Any]) -> None:
        merged = SMART_API_DEFAULTS.copy()
        merged.update(config)
        self._config["smart_api"] = merged
        self.save()

    def get_raw_config(self) -> Dict[str, Any]:
        return self._config.copy()


class DataProvider:
    """Fetches candle data from SmartAPI instead of yfinance."""
    
    def __init__(self, smart_client: Optional["SmartAPIClient"] = None, logger: Optional[Callable[[str], None]] = None):
        self.smart_client = smart_client
        self.logger = logger or print

    def fetch_candles(self, symbol: str, symbol_token: str, exchange: str, timeframe: str, lookback: int = 200) -> Optional[pd.DataFrame]:
        """Fetch candles from SmartAPI."""
        if not self.smart_client or not self.smart_client.is_ready:
            self.logger(f"SmartAPI not ready. Cannot fetch candles for {symbol}.")
            return None

        if timeframe not in SMARTAPI_TIMEFRAME_MAP:
            self.logger(f"Unsupported timeframe: {timeframe}")
            return None

        try:
            # Get current date
            from datetime import datetime, timedelta
            to_date = datetime.now()
            from_date = to_date - timedelta(days=5)  # Fetch last 5 days
            
            # Convert to SmartAPI format
            smartapi_timeframe = SMARTAPI_TIMEFRAME_MAP[timeframe]
            
            # Call SmartAPI getCandleData
            response = self.smart_client.client.getCandleData({
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": smartapi_timeframe,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M"),
            })
            
            if not response or response.get("status") != True:
                self.logger(f"Failed to fetch candles for {symbol}: {response}")
                return None
            
            data = response.get("data", [])
            if not data:
                self.logger(f"No candle data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            # SmartAPI returns data in format: [timestamp, open, high, low, close, volume]
            # Rename columns to standard format
            if len(df.columns) >= 6:
                df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            else:
                self.logger(f"Unexpected candle data format for {symbol}: {len(df.columns)} columns")
                return None
            
            # Parse timestamp (try multiple formats)
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            except Exception:
                self.logger(f"Error parsing timestamps for {symbol}")
                return None
            
            df = df.dropna(subset=['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            # Take last lookback rows
            return df.tail(lookback) if len(df) > lookback else df
            
        except Exception as exc:
            self.logger(f"Error fetching candles for {symbol}: {exc}")
            return None


def _check_smartapi_import() -> Tuple[bool, Optional[str]]:
    """Check if SmartApi can be imported. Returns (success, error_message)."""
    global SmartConnect
    if SmartConnect is not None:
        return True, None
    
    # Try to import again in case package was installed after module load
    try:
        from SmartApi import SmartConnect as SC  # type: ignore
        SmartConnect = SC
        return True, None
    except ImportError as e:
        try:
            from SmartApi.smartConnect import SmartConnect as SC  # type: ignore
            SmartConnect = SC
            return True, None
        except ImportError:
            return False, f"SmartApi import failed: {e}. Run 'pip install smartapi-python'."


class SmartAPIClient:
    """Enhanced SmartAPI client with fund validation, order verification, and position fetching."""
    
    def __init__(self, credentials: Dict[str, Any], logger: Optional[Callable[[str], None]] = None):
        self.credentials = credentials
        self.logger = logger or print
        self.client: Optional["SmartConnect"] = None
        self.feed_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.user_id: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return bool(self.client and self.credentials.get("enabled"))

    def update_credentials(self, credentials: Dict[str, Any]) -> None:
        self.credentials = credentials
        self.client = None
        self.feed_token = None
        self.refresh_token = None
        self.user_id = None

    def connect(self) -> bool:
        """Connect to SmartAPI and validate session."""
        if not self.credentials.get("enabled"):
            self.logger("SmartAPI trading disabled. Enable it in the config to place live orders.")
            return False
        
        # Check import with retry (in case package was installed after module load)
        import_ok, import_error = _check_smartapi_import()
        if not import_ok:
            self.logger(import_error or "SmartApi package not installed. Run 'pip install smartapi-python'.")
            return False
        
        if pyotp is None:
            self.logger("pyotp not installed. Run 'pip install pyotp'.")
            return False
        
        missing_keys = [
            key
            for key in ("api_key", "client_id", "password", "totp_secret")
            if not self.credentials.get(key)
        ]
        if missing_keys:
            self.logger(f"SmartAPI credentials missing: {', '.join(missing_keys)}")
            return False
        
        try:
            self.client = SmartConnect(api_key=self.credentials["api_key"])
            otp = pyotp.TOTP(self.credentials["totp_secret"]).now()
            session = self.client.generateSession(
                self.credentials["client_id"],
                self.credentials["password"],
                otp,
            )
            
            if session.get("status") != True:
                error_msg = session.get("message", "Unknown error")
                self.logger(f"SmartAPI login failed: {error_msg}")
                self.client = None
                return False
            
            self.refresh_token = session["data"]["refreshToken"]
            self.feed_token = self.client.getfeedToken()
            
            # Get user profile to validate
            profile = self.client.getProfile(self.refresh_token)
            if profile.get("status") == True:
                self.user_id = profile.get("data", {}).get("clientcode")
                self.logger(f"SmartAPI session established. User ID: {self.user_id}")
            else:
                self.logger("Warning: Could not fetch user profile")
            
            return True
            
        except Exception as exc:
            self.logger(f"SmartAPI login failed: {exc}")
            self.client = None
            return False

    def login_with_pin(self, pin: str) -> Tuple[bool, str]:
        """
        Login to SmartAPI using PIN from UI.
        Credentials (API key, client ID, TOTP secret) come from .env via credentials dict.
        Returns (success, message).
        """
        # Check import with retry
        import_ok, import_error = _check_smartapi_import()
        if not import_ok:
            return False, import_error or "SmartApi package not installed. Run 'pip install smartapi-python'."
        
        if pyotp is None:
            return False, "pyotp not installed. Run 'pip install pyotp'."
        
        # Validate required credentials from env
        required_keys = ["api_key", "client_id", "totp_secret"]
        missing_keys = [key for key in required_keys if not self.credentials.get(key)]
        if missing_keys:
            return False, f"Missing credentials in .env: {', '.join(missing_keys)}"
        
        if not pin or not pin.strip():
            return False, "PIN cannot be empty"
        
        try:
            # Create SmartConnect client
            self.client = SmartConnect(api_key=self.credentials["api_key"])
            
            # Generate TOTP from secret
            totp = pyotp.TOTP(self.credentials["totp_secret"]).now()
            
            # Call generateSession with client_id, PIN, and TOTP
            session = self.client.generateSession(
                self.credentials["client_id"],
                pin,
                totp,
            )
            
            if session.get("status") != True:
                error_msg = session.get("message", "Unknown error")
                self.client = None
                return False, f"Login failed: {error_msg}"
            
            # Extract tokens
            self.refresh_token = session.get("data", {}).get("refreshToken")
            self.feed_token = self.client.getfeedToken()
            
            # Get user profile to validate
            profile = self.client.getProfile(self.refresh_token)
            if profile.get("status") == True:
                self.user_id = profile.get("data", {}).get("clientcode")
                self.logger(f"Login successful for {self.user_id}")
                return True, f"Login successful for {self.user_id}"
            else:
                self.logger("Warning: Could not fetch user profile after login")
                return True, f"Login successful for {self.credentials['client_id']}"
            
        except Exception as exc:
            self.logger(f"Login exception: {exc}")
            self.client = None
            return False, f"Login failed: {exc}"

    def validate_connectivity(self) -> Tuple[bool, str]:
        """Validate SmartAPI connectivity and return (success, message)."""
        if not self.is_ready:
            return False, "SmartAPI client not connected"
        
        try:
            # Test with profile fetch
            profile = self.client.getProfile(self.refresh_token)
            if profile.get("status") != True:
                return False, f"Profile validation failed: {profile.get('message', 'Unknown error')}"
            
            # Test token mapping (try to search for a symbol)
            # This is optional, but good to validate
            return True, "SmartAPI connectivity validated"
        except Exception as exc:
            return False, f"Connectivity validation failed: {exc}"

    def check_funds(self, required_margin: float) -> Tuple[bool, float, str]:
        """Check available funds using rmsLimit(). Returns (has_funds, available, message)."""
        if not self.is_ready or not self.client:
            return False, 0.0, "SmartAPI client not connected"
        
        try:
            response = self.client.rmsLimit()
            if response.get("status") != True:
                return False, 0.0, f"Failed to fetch RMS limits: {response.get('message', 'Unknown error')}"
            
            data = response.get("data", {})
            available = float(data.get("available", 0))
            
            if available >= required_margin:
                return True, available, f"Funds available: ₹{available:.2f}"
            else:
                return False, available, f"Insufficient funds. Required: ₹{required_margin:.2f}, Available: ₹{available:.2f}"
                
        except Exception as exc:
            return False, 0.0, f"Error checking funds: {exc}"

    def get_latest_price(self, symbol_token: str, exchange: str) -> Optional[float]:
        """Get latest price using LTP data."""
        if not self.is_ready or not self.client:
            return None
        
        try:
            response = self.client.ltpData(exchange, symbol_token)
            if response.get("status") == True:
                data = response.get("data", {})
                return float(data.get("ltp", 0))
            return None
        except Exception as exc:
            self.logger(f"Error fetching LTP for {symbol_token}: {exc}")
            return None

    def check_funds_and_place_order(
        self,
        *,
        symbol: str,
        symbol_token: Optional[str],
        exchange: str,
        transaction_type: str,
        quantity: int,
        product_type: str,
        price: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Check available funds before placing order. Returns (success, order_id, message).
        This is the main entry point for order placement with fund validation.
        """
        if not self.is_ready or not self.client:
            return False, None, "SmartAPI client not connected"
        
        if not symbol_token:
            return False, None, f"Symbol token missing for {symbol}"
        
        # Get current price if not provided (for market orders)
        if price is None:
            price = self.get_latest_price(symbol_token, exchange)
            if price is None:
                return False, None, f"Could not fetch current price for {symbol}"
        
        # Calculate required margin (rough estimate: price * quantity * 0.1 for options)
        # For more accurate margin, use SmartAPI margin calculator
        required_margin = price * quantity * 0.1  # Conservative estimate
        
        # Check funds before placing order
        has_funds, available, fund_msg = self.check_funds(required_margin)
        if not has_funds:
            self.logger(f"Insufficient funds for {symbol} (needed ₹{required_margin:.2f}, available ₹{available:.2f})")
            return False, None, f"Insufficient funds for {symbol} (needed ₹{required_margin:.2f}, available ₹{available:.2f})"
        
        # Funds are sufficient, proceed with order placement
        return self.place_order(
            symbol=symbol,
            symbol_token=symbol_token,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            product_type=product_type,
            price=price,
        )

    def place_order(
        self,
        *,
        symbol: str,
        symbol_token: Optional[str],
        exchange: str,
        transaction_type: str,
        quantity: int,
        product_type: str,
        price: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Place order via SmartAPI. Returns (success, order_id, message).
        NOTE: Call check_funds_and_place_order() instead to include fund validation.
        """
        if not self.is_ready or not self.client:
            return False, None, "SmartAPI client not connected"
        
        if not symbol_token:
            return False, None, f"Symbol token missing for {symbol}"
        
        # Get current price if not provided (for market orders)
        if price is None:
            price = self.get_latest_price(symbol_token, exchange)
            if price is None:
                return False, None, f"Could not fetch current price for {symbol}"
        
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": symbol_token,
            "transactiontype": transaction_type.upper(),
            "exchange": exchange.upper(),
            "ordertype": "MARKET",
            "producttype": product_type.upper(),
            "duration": "DAY",
            "price": str(price) if price else "0",
            "quantity": str(quantity),
            "squareoff": "0",
            "stoploss": "0",
            "trailingStopLoss": "0",
        }
        
        try:
            response = self.client.placeOrder(order_params)
            
            if response.get("status") != True:
                error_msg = response.get("message", "Unknown error")
                self.logger(f"Order rejected for {symbol}: {error_msg}")
                return False, None, f"Order rejected for {symbol}: {error_msg}"
            
            order_id = response.get("data", {}).get("orderid")
            if not order_id:
                return False, None, "Order placed but no order ID returned"
            
            self.logger(f"Order placed for {symbol} @ ₹{price:.2f}, qty {quantity}")
            return True, str(order_id), "Order placed successfully"
            
        except Exception as exc:
            self.logger(f"Order placement exception for {symbol}: {exc}")
            return False, None, f"Order placement exception: {exc}"

    def verify_order_execution(self, order_id: str, max_wait_seconds: int = 30) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Verify order execution by checking orderBook. Returns (executed, order_data, message)."""
        if not self.is_ready or not self.client:
            return False, None, "SmartAPI client not connected"
        
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            try:
                response = self.client.orderBook()
                if response.get("status") == True:
                    orders = response.get("data", [])
                    for order in orders:
                        if str(order.get("orderid")) == str(order_id):
                            status = order.get("status", "").upper()
                            if status in ["COMPLETE", "FILLED"]:
                                return True, order, "Order executed"
                            elif status in ["REJECTED", "CANCELLED"]:
                                return False, order, f"Order {status.lower()}"
                            # Still pending
                            break
                
                time.sleep(2)  # Wait 2 seconds before next check
            except Exception as exc:
                return False, None, f"Error checking order status: {exc}"
        
        return False, None, "Order execution timeout"

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get real positions from SmartAPI positionBook()."""
        if not self.is_ready or not self.client:
            return []
        
        try:
            response = self.client.positionBook()
            if response.get("status") == True:
                return response.get("data", [])
            return []
        except Exception as exc:
            self.logger(f"Error fetching positions: {exc}")
            return []

    def get_trades(self) -> List[Dict[str, Any]]:
        """Get real trades from SmartAPI tradeBook()."""
        if not self.is_ready or not self.client:
            return []
        
        try:
            response = self.client.tradeBook()
            if response.get("status") == True:
                return response.get("data", [])
            return []
        except Exception as exc:
            self.logger(f"Error fetching trades: {exc}")
            return []


def calculate_rsi(series: pd.Series, length: int) -> pd.Series:
    """Calculate RSI using exponential moving average."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean().replace(0, np.nan)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


class TradingBot:
    """Trading bot that uses REAL SmartAPI orders only."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        data_provider: DataProvider,
        smart_client: Optional["SmartAPIClient"] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.config_manager = config_manager
        self.data_provider = data_provider
        self.smart_client = smart_client
        self.logger = logger or print
        
        # Store only REAL positions from SmartAPI
        self.positions: Dict[str, Position] = {}
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        # Track RSI history for crossover detection
        self.rsi_history: Dict[str, List[float]] = {}

    def start(self) -> None:
        """Start the trading bot with validation."""
        if self.thread and self.thread.is_alive():
            self.logger("Trading bot is already running.")
            return
        
        # Validate SmartAPI connectivity
        if self.smart_client:
            if not self.smart_client.is_ready:
                if not self.smart_client.connect():
                    self.logger("ERROR: SmartAPI connection failed. Bot cannot start.")
                    return
            
            # Validate connectivity
            is_valid, msg = self.smart_client.validate_connectivity()
            if not is_valid:
                self.logger(f"ERROR: SmartAPI validation failed: {msg}. Bot cannot start.")
                return
            
            self.logger(f"SmartAPI validated: {msg}")
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.logger("Trading bot started.")

    def stop(self) -> None:
        """Stop the trading bot."""
        if not self.thread or not self.thread.is_alive():
            self.logger("Trading bot is not running.")
            return
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.logger("Trading bot stopped.")

    def is_running(self) -> bool:
        """Check if bot is running."""
        return self.thread is not None and self.thread.is_alive()

    def _run_loop(self) -> None:
        """Main trading loop."""
        while not self.stop_event.is_set():
            try:
                # Sync positions from SmartAPI
                self._sync_positions_from_smartapi()
                
                # Process each instrument
                instruments = self.config_manager.get_instruments()
                for instrument in instruments:
                    if self.stop_event.is_set():
                        break
                    self.process_instrument(instrument)
                
                time.sleep(5)  # Wait 5 seconds between cycles
            except Exception as exc:
                self.logger(f"Error in trading loop: {exc}")
                time.sleep(10)

    def _sync_positions_from_smartapi(self) -> None:
        """Sync positions from SmartAPI positionBook()."""
        if not self.smart_client or not self.smart_client.is_ready:
            return
        
        try:
            api_positions = self.smart_client.get_positions()
            with self.lock:
                # Clear existing positions
                self.positions.clear()
                
                # Add only positions with non-zero quantity
                for pos in api_positions:
                    qty = int(pos.get("netqty", 0))
                    if qty != 0:  # Only non-zero positions
                        symbol = pos.get("tradingsymbol", "").upper()
                        entry_price = float(pos.get("averageprice", 0))
                        
                        # Find instrument config for target/stop
                        inst = self._find_instrument_config(symbol)
                        if inst:
                            target_price = entry_price * (1 + inst.profit_target_pct / 100)
                            stop_price = entry_price * (1 - inst.stop_loss_pct / 100)
                        else:
                            # Default targets if instrument not found
                            target_price = entry_price * 1.05
                            stop_price = entry_price * 0.90
                        
                        position = Position(
                            symbol=symbol,
                            entry_price=entry_price,
                            quantity=abs(qty),
                            target_price=target_price,
                            stop_price=stop_price,
                            timestamp=time.time(),
                            exchange=pos.get("exchange", "NSE"),
                            product_type=pos.get("producttype", "INTRADAY"),
                        )
                        self.positions[symbol] = position
        except Exception as exc:
            self.logger(f"Error syncing positions: {exc}")

    def process_instrument(self, instrument: InstrumentConfig) -> None:
        """Process a single instrument for RSI signals."""
        if not instrument.symbol_token:
            return
        
        # Fetch candles from SmartAPI
        candles = self.data_provider.fetch_candles(
            instrument.symbol,
            instrument.symbol_token,
            instrument.exchange,
            instrument.timeframe,
        )
        
        if candles is None or candles.empty or len(candles) < instrument.rsi_length + 1:
            return
        
        # Calculate RSI
        rsi = calculate_rsi(candles["Close"], instrument.rsi_length)
        candles = candles.assign(RSI=rsi)
        
        # Get last 2 RSI values for crossover detection
        last_rows = candles.tail(2)
        if len(last_rows) < 2:
            return
        
        prev_rsi = last_rows["RSI"].iloc[-2]
        current_rsi = last_rows["RSI"].iloc[-1]
        current_price = last_rows["Close"].iloc[-1]
        
        # Store RSI history
        symbol_key = instrument.symbol.upper()
        if symbol_key not in self.rsi_history:
            self.rsi_history[symbol_key] = []
        self.rsi_history[symbol_key].append(current_rsi)
        if len(self.rsi_history[symbol_key]) > 100:
            self.rsi_history[symbol_key] = self.rsi_history[symbol_key][-100:]
        
        with self.lock:
            position = self.positions.get(symbol_key)
            
            if position:
                # Check exit conditions
                if self.should_take_profit(position, current_price):
                    self.execute_sell(position, current_price, reason="Target hit")
                elif self.should_stop_loss(position, current_price):
                    self.execute_sell(position, current_price, reason="Stop loss hit")
            else:
                # Check for RSI crossover BUY signal
                if self.is_rsi_crossover(prev_rsi, current_rsi, instrument.rsi_level):
                    self.execute_buy(instrument, current_price)

    @staticmethod
    def is_rsi_crossover(prev_rsi: float, current_rsi: float, threshold: float) -> bool:
        """
        Correct RSI crossover logic:
        Buy when previous_RSI < target_RSI AND current_RSI >= target_RSI
        """
        return prev_rsi < threshold and current_rsi >= threshold

    def execute_buy(self, instrument: InstrumentConfig, price: float) -> None:
        """Execute BUY order via SmartAPI. Only create position if order is executed."""
        if not self.smart_client or not self.smart_client.is_ready:
            self.logger(f"Cannot place BUY order for {instrument.symbol}: SmartAPI not ready")
            return
        
        # Place order with fund check
        success, order_id, message = self.smart_client.check_funds_and_place_order(
            symbol=instrument.symbol,
            symbol_token=instrument.symbol_token,
            exchange=instrument.exchange,
            transaction_type="BUY",
            quantity=instrument.quantity,
            product_type=instrument.product_type,
        )
        
        if not success:
            self.logger(f"BUY order failed for {instrument.symbol}: {message}")
            return
        
        self.logger(f"BUY order placed for {instrument.symbol}: {order_id}")
        
        # Verify order execution
        executed, order_data, exec_message = self.smart_client.verify_order_execution(order_id)
        
        if not executed:
            self.logger(f"BUY order not executed for {instrument.symbol}: {exec_message}")
            return
        
        # Order executed - get filled price
        filled_price = float(order_data.get("averageprice", price)) if order_data else price
        
        # Calculate targets
        target_price = filled_price * (1 + instrument.profit_target_pct / 100)
        stop_price = filled_price * (1 - instrument.stop_loss_pct / 100)
        
        # Create position only after verified execution
        position = Position(
            symbol=instrument.symbol.upper(),
            entry_price=filled_price,
            quantity=instrument.quantity,
            target_price=target_price,
            stop_price=stop_price,
            timestamp=time.time(),
            order_id=order_id,
            exchange=instrument.exchange,
            product_type=instrument.product_type,
        )
        
        with self.lock:
            self.positions[position.symbol] = position
        
        self.logger(
            f"[BUY EXECUTED] {instrument.symbol} | qty={instrument.quantity} | "
            f"price={filled_price:.2f} | target={target_price:.2f} | stop={stop_price:.2f}"
        )

    def execute_sell(self, position: Position, price: float, reason: str) -> None:
        """Execute SELL order via SmartAPI."""
        if not self.smart_client or not self.smart_client.is_ready:
            self.logger(f"Cannot place SELL order for {position.symbol}: SmartAPI not ready")
            return
        
        # Find symbol token
        inst = self._find_instrument_config(position.symbol)
        symbol_token = inst.symbol_token if inst else None
        
        if not symbol_token:
            self.logger(f"Cannot place SELL: symbol token not found for {position.symbol}")
            return
        
        # Place order
        success, order_id, message = self.smart_client.place_order(
            symbol=position.symbol,
            symbol_token=symbol_token,
            exchange=position.exchange,
            transaction_type="SELL",
            quantity=position.quantity,
            product_type=position.product_type,
        )
        
        if not success:
            self.logger(f"SELL order failed for {position.symbol}: {message}")
            return
        
        self.logger(f"SELL order placed for {position.symbol}: {order_id}")
        
        # Verify execution
        executed, order_data, exec_message = self.smart_client.verify_order_execution(order_id)
        
        if executed:
            filled_price = float(order_data.get("averageprice", price)) if order_data else price
            pnl = (filled_price - position.entry_price) * position.quantity
            pnl_pct = ((filled_price - position.entry_price) / position.entry_price) * 100
            
            self.logger(
                f"[SELL EXECUTED] {position.symbol} | qty={position.quantity} | "
                f"price={filled_price:.2f} | P&L=₹{pnl:.2f} ({pnl_pct:.2f}%) | reason={reason}"
            )
        else:
            self.logger(f"SELL order not executed for {position.symbol}: {exec_message}")
        
        # Remove position (will be synced from SmartAPI on next cycle)
        with self.lock:
            self.positions.pop(position.symbol, None)

    def _find_instrument_config(self, symbol: str) -> Optional[InstrumentConfig]:
        """Find instrument config by symbol."""
        for inst in self.config_manager.get_instruments():
            if inst.symbol.upper() == symbol.upper():
                return inst
        return None

    @staticmethod
    def should_take_profit(position: Position, price: float) -> bool:
        """Check if take profit condition is met."""
        return price >= position.target_price

    @staticmethod
    def should_stop_loss(position: Position, price: float) -> bool:
        """Check if stop loss condition is met."""
        return price <= position.stop_price

    def get_positions_snapshot(self) -> List[Dict[str, Any]]:
        """Get positions snapshot with current prices and P&L."""
        snapshot = []
        with self.lock:
            for pos in self.positions.values():
                # Get current price from SmartAPI
                inst = self._find_instrument_config(pos.symbol)
                current_price = pos.entry_price  # Default to entry if can't fetch
                
                if inst and inst.symbol_token and self.smart_client and self.smart_client.is_ready:
                    ltp = self.smart_client.get_latest_price(inst.symbol_token, inst.exchange)
                    if ltp:
                        current_price = ltp
                
                pnl = (current_price - pos.entry_price) * pos.quantity
                pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                
                snapshot.append({
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": current_price,
                    "target_price": pos.target_price,
                    "stop_price": pos.stop_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "timestamp": pos.timestamp,
                })
        return snapshot
