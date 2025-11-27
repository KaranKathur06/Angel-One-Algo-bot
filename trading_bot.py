import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from SmartApi import SmartConnect  # type: ignore   
except ImportError:  # pragma: no cover - optional dependency
    SmartConnect = None  # type: ignore

try:
    import pyotp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyotp = None


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
    {
        "symbol": "BANKNIFTY",
        "timeframe": "5m",
        "rsi_length": 14,
        "rsi_level": 30,
        "profit_target_pct": 5.0,
        "stop_loss_pct": 10.0,
        "quantity": 1,
        "exchange": "NFO",
        "symbol_token": "99926009",
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

SYMBOL_ALIASES = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
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


class ConfigManager:
    def __init__(self, path: Path):
        self.path = path
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
            print("Config file is corrupted. Regenerating defaults.")
            self._config = {
                "instruments": DEFAULT_INSTRUMENTS.copy(),
                "smart_api": SMART_API_DEFAULTS.copy(),
            }
            self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self._config, indent=2))

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


class DataProvider:
    def __init__(self):
        pass

    @staticmethod
    def resolve_symbol(symbol: str) -> str:
        return SYMBOL_ALIASES.get(symbol.upper(), symbol)

    def fetch_candles(self, symbol: str, timeframe: str, lookback: int = 200) -> Optional[pd.DataFrame]:
        if timeframe not in TIMEFRAME_MAP:
            print(f"Unsupported timeframe: {timeframe}")
            return None

        base_interval, resample_rule = TIMEFRAME_MAP[timeframe]
        ticker = self.resolve_symbol(symbol)

        try:
            data = yf.download(
                tickers=ticker,
                period="5d",
                interval=base_interval,
                progress=False,
                auto_adjust=False,
                actions=False,
            )
        except Exception as exc:
            print(f"Failed to download data for {symbol}: {exc}")
            return None

        if data.empty:
            print(f"No data returned for {symbol}")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(set(data.columns)):
            print(f"Downloaded data missing OHLCV columns for {symbol}. Columns: {list(data.columns)}")
            return None

        data = data.tail(max(lookback * 2, 100))

        if resample_rule:
            data = (
                data.resample(resample_rule)
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna()
            )

        return data.tail(lookback)


class SmartAPIClient:
    def __init__(self, credentials: Dict[str, Any]):
        self.credentials = credentials
        self.client: Optional["SmartConnect"] = None
        self.feed_token: Optional[str] = None
        self.refresh_token: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return bool(self.client and self.credentials.get("enabled"))

    def update_credentials(self, credentials: Dict[str, Any]) -> None:
        self.credentials = credentials
        self.client = None
        self.feed_token = None
        self.refresh_token = None

    def connect(self) -> bool:
        if not self.credentials.get("enabled"):
            print("SmartAPI trading disabled. Enable it in the config to place live orders.")
            return False
        if SmartConnect is None:
            print("SmartApi package not installed. Run 'pip install smartapi-python'.")
            return False
        if pyotp is None:
            print("pyotp not installed. Run 'pip install pyotp'.")
            return False
        missing_keys = [
            key
            for key in ("api_key", "client_id", "password", "totp_secret")
            if not self.credentials.get(key)
        ]
        if missing_keys:
            print(f"SmartAPI credentials missing: {', '.join(missing_keys)}")
            return False
        try:
            self.client = SmartConnect(api_key=self.credentials["api_key"])
            otp = pyotp.TOTP(self.credentials["totp_secret"]).now()
            session = self.client.generateSession(
                self.credentials["client_id"],
                self.credentials["password"],
                otp,
            )
            self.refresh_token = session["data"]["refreshToken"]
            self.feed_token = self.client.getfeedToken()
            print("SmartAPI session established.")
            return True
        except Exception as exc:  # pragma: no cover - depends on external API
            print(f"SmartAPI login failed: {exc}")
            self.client = None
            return False

    def place_order(
        self,
        *,
        symbol: str,
        symbol_token: Optional[str],
        exchange: str,
        transaction_type: str,
        quantity: int,
        product_type: str,
    ) -> bool:
        if not self.is_ready or not self.client:
            print("SmartAPI client not connected. Order simulated only.")
            return False
        if not symbol_token:
            print(f"Symbol token missing for {symbol}. Update instrument settings to place live orders.")
            return False

        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": symbol_token,
            "transactiontype": transaction_type.upper(),
            "exchange": exchange.upper(),
            "ordertype": "MARKET",
            "producttype": product_type.upper(),
            "duration": "DAY",
            "price": 0,
            "quantity": quantity,
            "squareoff": "0",
            "stoploss": "0",
            "trailingStopLoss": "0",
        }

        try:
            response = self.client.placeOrder(order_params)
            print(f"SmartAPI order response: {response}")
            return True
        except Exception as exc:  # pragma: no cover - depends on external API
            print(f"SmartAPI order failed: {exc}")
            return False


def calculate_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean().replace(0, np.nan)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


class TradingBot:
    def __init__(self, config_manager: ConfigManager, data_provider: DataProvider, smart_client: Optional["SmartAPIClient"] = None):
        self.config_manager = config_manager
        self.data_provider = data_provider
        self.smart_client = smart_client
        self.positions: Dict[str, Position] = {}
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            print("Trading bot is already running.")
            return
        if self.smart_client and not self.smart_client.is_ready:
            self.smart_client.connect()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("Trading bot started.")

    def stop(self) -> None:
        if not self.thread or not self.thread.is_alive():
            print("Trading bot is not running.")
            return
        self.stop_event.set()
        self.thread.join()
        print("Trading bot stopped.")

    def _run_loop(self) -> None:
        while not self.stop_event.is_set():
            instruments = self.config_manager.get_instruments()
            for instrument in instruments:
                if self.stop_event.is_set():
                    break
                self.process_instrument(instrument)
            time.sleep(5)

    def process_instrument(self, instrument: InstrumentConfig) -> None:
        candles = self.data_provider.fetch_candles(instrument.symbol, instrument.timeframe)
        if candles is None or candles.empty:
            return

        rsi = calculate_rsi(candles["Close"], instrument.rsi_length)
        candles = candles.assign(RSI=rsi)
        last_rows = candles.tail(2)
        if len(last_rows) < 2:
            return

        prev_rsi, current_rsi = last_rows["RSI"].iloc[-2], last_rows["RSI"].iloc[-1]
        current_price = last_rows["Close"].iloc[-1]

        with self.lock:
            position = self.positions.get(instrument.symbol.upper())

            if position:
                if self.should_take_profit(position, current_price):
                    self.execute_sell(position, current_price, reason="Target hit")
                elif self.should_stop_loss(position, current_price):
                    self.execute_sell(position, current_price, reason="Stop loss hit")
            else:
                if self.is_rsi_crossover(prev_rsi, current_rsi, instrument.rsi_level):
                    self.execute_buy(instrument, current_price)

    @staticmethod
    def is_rsi_crossover(prev_rsi: float, current_rsi: float, threshold: float) -> bool:
        return prev_rsi < threshold <= current_rsi

    def execute_buy(self, instrument: InstrumentConfig, price: float) -> None:
        # Placeholder for live trading integration
        target_price = price * (1 + instrument.profit_target_pct / 100)
        stop_price = price * (1 - instrument.stop_loss_pct / 100)
        position = Position(
            symbol=instrument.symbol.upper(),
            entry_price=price,
            quantity=instrument.quantity,
            target_price=target_price,
            stop_price=stop_price,
            timestamp=time.time(),
        )
        self.positions[position.symbol] = position
        if self.smart_client and self.smart_client.is_ready:
            self.smart_client.place_order(
                symbol=instrument.symbol,
                symbol_token=instrument.symbol_token,
                exchange=instrument.exchange,
                transaction_type="BUY",
                quantity=instrument.quantity,
                product_type=instrument.product_type,
            )
        print(
            f"[BUY] {instrument.symbol} | qty={instrument.quantity} | "
            f"price={price:.2f} | target={target_price:.2f} | stop={stop_price:.2f}"
        )

    def execute_sell(self, position: Position, price: float, reason: str) -> None:
        # Placeholder for live trading integration
        pnl = (price - position.entry_price) * position.quantity
        if self.smart_client and self.smart_client.is_ready:
            self.smart_client.place_order(
                symbol=position.symbol,
                symbol_token=self._find_symbol_token(position.symbol),
                exchange=self._find_exchange(position.symbol),
                transaction_type="SELL",
                quantity=position.quantity,
                product_type=self._find_product_type(position.symbol),
            )
        print(
            f"[SELL] {position.symbol} | qty={position.quantity} | price={price:.2f} | "
            f"P&L={pnl:.2f} | reason={reason}"
        )
        self.positions.pop(position.symbol, None)

    def _find_instrument_config(self, symbol: str) -> Optional[InstrumentConfig]:
        for inst in self.config_manager.get_instruments():
            if inst.symbol.upper() == symbol.upper():
                return inst
        return None

    def _find_exchange(self, symbol: str) -> str:
        inst = self._find_instrument_config(symbol)
        return inst.exchange if inst else "NSE"

    def _find_symbol_token(self, symbol: str) -> Optional[str]:
        inst = self._find_instrument_config(symbol)
        return inst.symbol_token if inst else None

    def _find_product_type(self, symbol: str) -> str:
        inst = self._find_instrument_config(symbol)
        return inst.product_type if inst else "INTRADAY"

    @staticmethod
    def should_take_profit(position: Position, price: float) -> bool:
        return price >= position.target_price

    @staticmethod
    def should_stop_loss(position: Position, price: float) -> bool:
        return price <= position.stop_price

    def show_positions(self) -> None:
        with self.lock:
            if not self.positions:
                print("No open positions.")
                return
            for pos in self.positions.values():
                print(
                    f"{pos.symbol} | qty={pos.quantity} | entry={pos.entry_price:.2f} | "
                    f"target={pos.target_price:.2f} | stop={pos.stop_price:.2f}"
                )


def prompt_float(prompt: str, default: float) -> float:
    user_input = input(f"{prompt} [{default}]: ").strip()
    if not user_input:
        return default
    try:
        return float(user_input)
    except ValueError:
        print("Invalid input. Using default.")
        return default


def prompt_int(prompt: str, default: int) -> int:
    user_input = input(f"{prompt} [{default}]: ").strip()
    if not user_input:
        return default
    try:
        return int(user_input)
    except ValueError:
        print("Invalid input. Using default.")
        return default


def choose_symbol(config_manager: ConfigManager) -> Optional[str]:
    instruments = config_manager.get_instruments()
    if not instruments:
        print("No instruments configured.")
        return None
    for idx, inst in enumerate(instruments, start=1):
        print(f"{idx}. {inst.symbol}")
    choice = input("Select symbol #: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(instruments)):
        print("Invalid selection.")
        return None
    return instruments[int(choice) - 1].symbol


def add_symbol(config_manager: ConfigManager) -> None:
    symbol = input("Enter symbol (e.g., RELIANCE.NS): ").strip().upper()
    if not symbol:
        print("Symbol cannot be empty.")
        return
    timeframe = input(f"Timeframe {list(TIMEFRAME_MAP.keys())} [5m]: ").strip() or "5m"
    if timeframe not in TIMEFRAME_MAP:
        print("Unsupported timeframe. Using 5m.")
        timeframe = "5m"

    config = InstrumentConfig(
        symbol=symbol,
        timeframe=timeframe,
        rsi_length=prompt_int("RSI length", 14),
        rsi_level=prompt_float("RSI level", 30.0),
        profit_target_pct=prompt_float("Profit target %", 5.0),
        stop_loss_pct=prompt_float("Stop loss %", 10.0),
        quantity=prompt_int("Quantity (lots)", 1),
        exchange=input("Exchange [NSE]: ").strip().upper() or "NSE",
        symbol_token=input("Symbol token (SmartAPI) [leave blank if unknown]: ").strip() or None,
        product_type=input("Product type [INTRADAY]: ").strip().upper() or "INTRADAY",
    )
    config_manager.upsert_instrument(config)
    print(f"{symbol} added/updated.")


def remove_symbol(config_manager: ConfigManager) -> None:
    symbol = choose_symbol(config_manager)
    if not symbol:
        return
    if config_manager.remove_instrument(symbol):
        print(f"{symbol} removed.")
    else:
        print(f"{symbol} not found.")


def update_parameters(config_manager: ConfigManager) -> None:
    symbol = choose_symbol(config_manager)
    if not symbol:
        return

    instruments = {inst.symbol.upper(): inst for inst in config_manager.get_instruments()}
    instrument = instruments.get(symbol.upper())
    if not instrument:
        print("Instrument not found.")
        return

    timeframe = input(f"Timeframe {list(TIMEFRAME_MAP.keys())} [{instrument.timeframe}]: ").strip() or instrument.timeframe
    if timeframe not in TIMEFRAME_MAP:
        print("Unsupported timeframe. Keeping current value.")
        timeframe = instrument.timeframe

    updated = InstrumentConfig(
        symbol=instrument.symbol,
        timeframe=timeframe,
        rsi_length=prompt_int("RSI length", instrument.rsi_length),
        rsi_level=prompt_float("RSI level", instrument.rsi_level),
        profit_target_pct=prompt_float("Profit target %", instrument.profit_target_pct),
        stop_loss_pct=prompt_float("Stop loss %", instrument.stop_loss_pct),
        quantity=prompt_int("Quantity (lots)", instrument.quantity),
        exchange=input(f"Exchange [{instrument.exchange}]: ").strip().upper() or instrument.exchange,
        symbol_token=input(
            f"Symbol token (SmartAPI) [{instrument.symbol_token or 'None'}]: "
        ).strip() or instrument.symbol_token,
        product_type=input(f"Product type [{instrument.product_type}]: ").strip().upper() or instrument.product_type,
    )
    config_manager.upsert_instrument(updated)
    print(f"{symbol} updated.")


def show_instruments(config_manager: ConfigManager) -> None:
    instruments = config_manager.get_instruments()
    if not instruments:
        print("No instruments configured.")
        return
    for inst in instruments:
        print(
            f"{inst.symbol} | tf={inst.timeframe} | RSI len={inst.rsi_length} | "
            f"RSI lvl={inst.rsi_level} | target={inst.profit_target_pct}% | "
            f"stop={inst.stop_loss_pct}% | qty={inst.quantity} | exch={inst.exchange} | "
            f"token={inst.symbol_token or 'N/A'} | product={inst.product_type}"
        )


def configure_smart_api(config_manager: ConfigManager, smart_client: SmartAPIClient) -> None:
    current = config_manager.get_smart_api_config()
    enabled_default = "y" if current.get("enabled") else "n"
    enabled_input = input(f"Enable SmartAPI trading? (y/n) [{enabled_default}]: ").strip().lower()
    enabled = current.get("enabled")
    if enabled_input in {"y", "n"}:
        enabled = enabled_input == "y"

    def prompt_secret(field: str, existing: str, mask: bool = False) -> str:
        label = f"{field} [{'****' if (mask and existing) else existing or 'empty'}]: "
        value = input(label).strip()
        return value or existing

    updated = {
        "enabled": enabled,
        "api_key": prompt_secret("API Key", current.get("api_key", "")),
        "client_id": prompt_secret("Client ID", current.get("client_id", "")),
        "password": prompt_secret("Password", current.get("password", ""), mask=True),
        "totp_secret": prompt_secret("TOTP Secret", current.get("totp_secret", ""), mask=True),
    }
    config_manager.update_smart_api_config(updated)
    smart_client.update_credentials(updated)
    print("SmartAPI configuration saved.")
    if updated["enabled"]:
        smart_client.connect()


def show_menu() -> None:
    print(
        "\n--- Trading Bot Menu ---\n"
        "1. Start Trading\n"
        "2. Stop Trading\n"
        "3. Add Symbol\n"
        "4. Remove Symbol\n"
        "5. Update Parameters\n"
        "6. Show Instruments\n"
        "7. Show Positions\n"
        "8. Exit\n"
        "9. Configure SmartAPI\n"
    )


def main() -> None:
    config_manager = ConfigManager(CONFIG_PATH)
    data_provider = DataProvider()
    smart_client = SmartAPIClient(config_manager.get_smart_api_config())
    if smart_client.credentials.get("enabled"):
        smart_client.connect()
    bot = TradingBot(config_manager, data_provider, smart_client)

    while True:
        show_menu()
        choice = input("Select option: ").strip()

        if choice == "1":
            bot.start()
        elif choice == "2":
            bot.stop()
        elif choice == "3":
            add_symbol(config_manager)
        elif choice == "4":
            remove_symbol(config_manager)
        elif choice == "5":
            update_parameters(config_manager)
        elif choice == "6":
            show_instruments(config_manager)
        elif choice == "7":
            bot.show_positions()
        elif choice == "8":
            bot.stop()
            print("Goodbye!")
            break
        elif choice == "9":
            configure_smart_api(config_manager, smart_client)
        else:
            print("Invalid option.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

