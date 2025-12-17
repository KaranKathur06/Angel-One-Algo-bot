"""
Streamlit-based web GUI for the RSI Trading Bot.
Uses REAL SmartAPI orders only - no simulated trades.
"""
import json
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from trading_bot import (
    CONFIG_PATH,
    ConfigManager,
    DataProvider,
    InstrumentConfig,
    SmartAPIClient,
    TIMEFRAME_MAP,
    TradingBot,
)

TIMEFRAME_OPTIONS = list(TIMEFRAME_MAP.keys())

# Page config
st.set_page_config(
    page_title="RSI Trading Bot - Real SmartAPI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "bot" not in st.session_state:
    st.session_state.bot = None
if "config_manager" not in st.session_state:
    st.session_state.config_manager = None
if "data_provider" not in st.session_state:
    st.session_state.data_provider = None
if "smart_client" not in st.session_state:
    st.session_state.smart_client = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None


def enqueue_log(message: str) -> None:
    """Add a log message to the session state."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.logs.append(log_entry)
    # Keep only last 500 lines
    if len(st.session_state.logs) > 500:
        st.session_state.logs = st.session_state.logs[-500:]


def login_smartapi_from_ui(pin: str) -> None:
    """Handle SmartAPI login from UI with PIN."""
    if not st.session_state.smart_client:
        enqueue_log("ERROR: SmartAPI client not initialized")
        st.error("SmartAPI client not initialized")
        return
    
    enqueue_log("Attempting SmartAPI login...")
    success, message = st.session_state.smart_client.login_with_pin(pin)
    
    if success:
        st.session_state.is_logged_in = True
        st.session_state.logged_in_user = st.session_state.smart_client.user_id or st.session_state.smart_client.credentials.get("client_id")
        enqueue_log(message)
        st.success(message)
    else:
        st.session_state.is_logged_in = False
        st.session_state.logged_in_user = None
        enqueue_log(f"Login failed: {message}")
        st.error(message)


def initialize_bot() -> None:
    """Initialize the trading bot components."""
    if st.session_state.config_manager is None:
        st.session_state.config_manager = ConfigManager(CONFIG_PATH, logger=enqueue_log)
        st.session_state.data_provider = DataProvider(logger=enqueue_log)
        st.session_state.smart_client = SmartAPIClient(
            st.session_state.config_manager.get_smart_api_config(),
            logger=enqueue_log
        )
        
        # Connect SmartAPI if enabled
        if st.session_state.smart_client.credentials.get("enabled"):
            enqueue_log("Connecting to SmartAPI...")
            if st.session_state.smart_client.connect():
                enqueue_log("SmartAPI connected successfully")
                # Validate connectivity
                is_valid, msg = st.session_state.smart_client.validate_connectivity()
                if is_valid:
                    enqueue_log(f"SmartAPI validated: {msg}")
                else:
                    enqueue_log(f"WARNING: SmartAPI validation failed: {msg}")
            else:
                enqueue_log("ERROR: SmartAPI connection failed")
        
        # Update data provider with smart client
        st.session_state.data_provider.smart_client = st.session_state.smart_client
        
        st.session_state.bot = TradingBot(
            st.session_state.config_manager,
            st.session_state.data_provider,
            st.session_state.smart_client,
            logger=enqueue_log,
        )
        enqueue_log("Bot initialized.")


# Initialize on first run
initialize_bot()

# Sidebar
with st.sidebar:
    st.title("📈 RSI Trading Bot")
    st.caption("Real SmartAPI Trading")
    st.divider()

    # Login Status
    if st.session_state.is_logged_in:
        st.success(f"✅ Logged in: {st.session_state.logged_in_user}")
    else:
        st.warning("⚠️ Not logged in")
    
    st.divider()

    # Login Section
    st.subheader("🔐 SmartAPI Login")
    pin_input = st.text_input("Enter PIN", type="password", key="pin_input")
    if st.button("🔓 Login to Angel One", type="primary", use_container_width=True):
        if not pin_input:
            st.error("PIN cannot be empty")
        else:
            login_smartapi_from_ui(pin_input)
            st.rerun()
    
    st.divider()

    # System Status
    is_running = st.session_state.bot and st.session_state.bot.is_running()
    status = "🟢 Running" if is_running else "🔴 Stopped"
    st.metric("System Status", status.split()[-1])
    
    # SmartAPI Status
    smartapi_status = "🟢 Connected" if (st.session_state.smart_client and st.session_state.smart_client.is_ready) else "🔴 Disconnected"
    st.metric("SmartAPI", smartapi_status.split()[-1])

    st.divider()

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", type="primary", use_container_width=True):
            if not st.session_state.is_logged_in:
                st.error("Please login first")
                enqueue_log("ERROR: Cannot start bot - not logged in")
            elif is_running:
                st.warning("Bot already running.")
            else:
                st.session_state.bot.start()
                enqueue_log("Bot start command issued")
                st.rerun()

    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            if not is_running:
                st.warning("Bot already stopped.")
            else:
                st.session_state.bot.stop()
                enqueue_log("Bot stop command issued")
                st.rerun()
    
    st.divider()
    
    # SmartAPI Connection
    if st.button("🔄 Reconnect SmartAPI", use_container_width=True):
        if st.session_state.smart_client:
            if st.session_state.smart_client.connect():
                enqueue_log("SmartAPI reconnected")
                st.success("Reconnected!")
                st.rerun()
            else:
                enqueue_log("SmartAPI reconnection failed")
                st.error("Reconnection failed!")

    st.divider()
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Instruments", "💰 Positions", "⚙️ Settings"])

# ==================== DASHBOARD TAB ====================
with tab1:
    st.header("Dashboard")
    
    # Status cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        is_running = st.session_state.bot and st.session_state.bot.is_running()
        status_text = "🟢 Running" if is_running else "🔴 Stopped"
        st.metric("Bot Status", status_text.split()[-1])
    
    with col2:
        smartapi_ready = st.session_state.smart_client and st.session_state.smart_client.is_ready
        smartapi_text = "🟢 Connected" if smartapi_ready else "🔴 Disconnected"
        st.metric("SmartAPI", smartapi_text.split()[-1])
    
    with col3:
        positions_count = len(st.session_state.bot.positions) if st.session_state.bot else 0
        st.metric("Open Positions", positions_count)
    
    with col4:
        instruments_count = len(st.session_state.config_manager.get_instruments()) if st.session_state.config_manager else 0
        st.metric("Instruments", instruments_count)

    st.divider()

    # Control buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        is_running = st.session_state.bot and st.session_state.bot.is_running()
        status_text = "🟢 Running" if is_running else "🔴 Stopped"
        st.markdown(f"### {status_text}")

    with col2:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

    with col3:
        if st.button("🗑️ Clear Logs", use_container_width=True):
            st.session_state.logs = []
            st.rerun()

    st.divider()

    # Live Logs
    st.subheader("Live Logs / Status")
    log_container = st.container()
    with log_container:
        log_text = "\n".join(st.session_state.logs[-100:])  # Show last 100 lines
        st.text_area(
            "Logs",
            value=log_text if log_text else "No logs yet...",
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )

    # Auto-refresh
    is_running = st.session_state.bot and st.session_state.bot.is_running()
    if is_running:
        time.sleep(2)
        st.rerun()

# ==================== INSTRUMENTS TAB ====================
with tab2:
    st.header("Instruments Management")

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("➕ Add Instrument", use_container_width=True):
            st.session_state.show_add_form = True
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("📥 Export Config", use_container_width=True):
            config_json = json.dumps(st.session_state.config_manager.get_raw_config(), indent=2)
            st.download_button(
                "Download JSON",
                config_json,
                file_name="config.json",
                mime="application/json",
            )

    st.divider()

    # Instruments table
    instruments = st.session_state.config_manager.get_instruments()
    if instruments:
        data = []
        for inst in instruments:
            data.append(
                {
                    "Symbol": inst.symbol,
                    "Qty": inst.quantity,
                    "Interval": inst.timeframe,
                    "RSI Level": inst.rsi_level,
                    "Target (%)": inst.profit_target_pct,
                    "Stop (%)": inst.stop_loss_pct,
                    "Exchange": inst.exchange,
                    "Product": inst.product_type,
                    "Token": inst.symbol_token or "N/A",
                }
            )
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Edit/Delete interface
        st.subheader("Edit or Delete Instrument")
        selected_symbol = st.selectbox("Select instrument to edit/delete", [inst.symbol for inst in instruments])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ Edit Selected", use_container_width=True):
                st.session_state.edit_symbol = selected_symbol
                st.rerun()

        with col2:
            if st.button("🗑️ Delete Selected", use_container_width=True, type="secondary"):
                if st.session_state.config_manager.remove_instrument(selected_symbol):
                    enqueue_log(f"{selected_symbol} removed.")
                    st.success(f"{selected_symbol} removed.")
                    st.rerun()
                else:
                    st.error("Failed to remove instrument.")
    else:
        st.info("No instruments configured. Add one to get started.")

    # Add/Edit form
    if st.session_state.get("show_add_form") or st.session_state.get("edit_symbol"):
        st.divider()
        form_title = "Add Instrument" if not st.session_state.get("edit_symbol") else f"Edit {st.session_state.get('edit_symbol')}"
        st.subheader(form_title)

        # Get existing instrument if editing
        existing = None
        if st.session_state.get("edit_symbol"):
            existing = next(
                (inst for inst in instruments if inst.symbol == st.session_state.get("edit_symbol")), None
            )

        with st.form("instrument_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                symbol = st.text_input(
                    "Symbol *",
                    value=existing.symbol if existing else "",
                    disabled=existing is not None,
                    help="e.g., NIFTY02DEC2526000PE, NIFTY, BANKNIFTY",
                )
                timeframe = st.selectbox(
                    "Interval *",
                    TIMEFRAME_OPTIONS,
                    index=TIMEFRAME_OPTIONS.index(existing.timeframe) if existing and existing.timeframe in TIMEFRAME_OPTIONS else 2
                )
                rsi_length = st.number_input(
                    "RSI Length",
                    min_value=1,
                    max_value=100,
                    value=existing.rsi_length if existing else 14
                )
                rsi_level = st.number_input(
                    "RSI Level",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(existing.rsi_level) if existing else 30.0,
                    step=0.1
                )

            with col2:
                profit_target = st.number_input(
                    "Profit Target (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(existing.profit_target_pct) if existing else 5.0,
                    step=0.1
                )
                stop_loss = st.number_input(
                    "Stop Loss (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(existing.stop_loss_pct) if existing else 10.0,
                    step=0.1
                )
                quantity = st.number_input(
                    "Quantity",
                    min_value=1,
                    value=existing.quantity if existing else 1
                )
                exchange = st.text_input(
                    "Exchange",
                    value=existing.exchange if existing else "NFO",
                    help="e.g., NSE, NFO (for options)"
                )

            symbol_token = st.text_input(
                "Symbol Token (SmartAPI) *",
                value=existing.symbol_token if existing else "",
                help="Required for SmartAPI trading. Get from Angel One market watch."
            )
            product_type = st.text_input(
                "Product Type",
                value=existing.product_type if existing else "INTRADAY",
                help="e.g., INTRADAY, DELIVERY"
            )

            submitted = st.form_submit_button("💾 Save", type="primary", use_container_width=True)

            if submitted:
                if not symbol.strip():
                    st.error("Symbol cannot be empty.")
                elif not symbol_token.strip():
                    st.error("Symbol Token is required for SmartAPI trading.")
                else:
                    try:
                        config = InstrumentConfig(
                            symbol=symbol.strip().upper(),
                            timeframe=timeframe,
                            rsi_length=int(rsi_length),
                            rsi_level=float(rsi_level),
                            profit_target_pct=float(profit_target),
                            stop_loss_pct=float(stop_loss),
                            quantity=int(quantity),
                            exchange=exchange.strip().upper() or "NSE",
                            symbol_token=symbol_token.strip() or None,
                            product_type=product_type.strip().upper() or "INTRADAY",
                        )
                        st.session_state.config_manager.upsert_instrument(config)
                        enqueue_log(f"{config.symbol} saved.")
                        st.success(f"{config.symbol} saved successfully!")
                        st.session_state.show_add_form = False
                        st.session_state.edit_symbol = None
                        time.sleep(1)
                        st.rerun()
                    except ValueError as e:
                        st.error(f"Invalid input: {e}")

        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_add_form = False
            st.session_state.edit_symbol = None
            st.rerun()

# ==================== POSITIONS TAB ====================
with tab3:
    st.header("Open Positions")
    st.caption("Real positions from SmartAPI - No simulated trades")

    if st.button("🔄 Refresh Positions", use_container_width=True):
        # Force sync positions from SmartAPI
        if st.session_state.bot:
            st.session_state.bot._sync_positions_from_smartapi()
        st.rerun()

    # Get positions snapshot with real-time P&L
    if st.session_state.bot:
        positions_snapshot = st.session_state.bot.get_positions_snapshot()
        instruments = {inst.symbol.upper(): inst for inst in st.session_state.config_manager.get_instruments()}

        if positions_snapshot:
            data = []
            for pos in positions_snapshot:
                inst = instruments.get(pos["symbol"])
                strategy = f"RSI {inst.rsi_level} @ {inst.timeframe}" if inst else "RSI Strategy"
                entry_dt = datetime.fromtimestamp(pos["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                
                # Color code P&L
                pnl_color = "🟢" if pos["pnl"] >= 0 else "🔴"

                data.append(
                    {
                        "Symbol": pos["symbol"],
                        "Qty": pos["quantity"],
                        "Entry Price": f"₹{pos['entry_price']:.2f}",
                        "Current Price": f"₹{pos['current_price']:.2f}",
                        "P&L": f"{pnl_color} ₹{pos['pnl']:.2f}",
                        "P&L %": f"{pos['pnl_pct']:.2f}%",
                        "Target": f"₹{pos['target_price']:.2f}",
                        "Stop": f"₹{pos['stop_price']:.2f}",
                        "Entry Time": entry_dt,
                        "Strategy": strategy,
                    }
                )

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Summary
            total_pnl = sum(p["pnl"] for p in positions_snapshot)
            total_invested = sum(p["entry_price"] * p["quantity"] for p in positions_snapshot)
            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
            st.info(
                f"Showing {len(positions_snapshot)} open position(s). "
                f"Total P&L: ₹{total_pnl:.2f} ({total_pnl_pct:.2f}%). "
                f"Prices updated from SmartAPI in real-time."
            )
        else:
            st.info("No open positions. Positions are synced from SmartAPI positionBook().")
    else:
        st.warning("Bot not initialized.")

    # Auto-refresh if bot is running
    is_running = st.session_state.bot and st.session_state.bot.is_running()
    if is_running:
        time.sleep(3)
        st.rerun()

# ==================== SETTINGS TAB ====================
with tab4:
    st.header("Configuration Settings")
    
    # SmartAPI Status
    st.subheader("SmartAPI Connection")
    if st.session_state.smart_client:
        creds = st.session_state.smart_client.credentials
        is_enabled = creds.get("enabled", False)
        is_connected = st.session_state.smart_client.is_ready
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", "🟢 Enabled" if is_enabled else "🔴 Disabled")
        with col2:
            st.metric("Connection", "🟢 Connected" if is_connected else "🔴 Disconnected")
        
        if is_connected:
            # Validate connectivity
            is_valid, msg = st.session_state.smart_client.validate_connectivity()
            if is_valid:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
        
        # Check funds
        if is_connected:
            if st.button("💰 Check Available Funds", use_container_width=True):
                has_funds, available, fund_msg = st.session_state.smart_client.check_funds(0)
                if has_funds or available > 0:
                    st.success(f"Available Funds: ₹{available:.2f}")
                else:
                    st.warning(fund_msg)
    
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reload from Disk", use_container_width=True):
            # Reload by creating new ConfigManager
            st.session_state.config_manager = ConfigManager(CONFIG_PATH, logger=enqueue_log)
            st.session_state.bot.config_manager = st.session_state.config_manager
            # Sync SmartAPI credentials
            creds = st.session_state.config_manager.get_smart_api_config()
            st.session_state.smart_client.update_credentials(creds)
            if creds.get("enabled"):
                st.session_state.smart_client.connect()
            enqueue_log("Configuration reloaded from disk.")
            st.success("Configuration reloaded!")
            st.rerun()

    with col2:
        if st.button("💾 Force Save to File", use_container_width=True):
            st.session_state.config_manager.save()
            enqueue_log("Configuration force-saved to disk.")
            st.success("Configuration saved!")

    with col3:
        if st.button("📥 Download Config", use_container_width=True):
            # Get config from manager
            config_data = st.session_state.config_manager.get_raw_config()
            config_json = json.dumps(config_data, indent=2)
            st.download_button(
                "Download",
                config_json,
                file_name="config.json",
                mime="application/json",
            )

    st.divider()

    # Config editor
    st.subheader("Edit Configuration (JSON)")
    # Get current config from manager
    current_config = st.session_state.config_manager.get_raw_config()
    config_text = st.text_area(
        "Config JSON",
        value=json.dumps(current_config, indent=2),
        height=400,
        label_visibility="collapsed",
    )

    if st.button("💾 Save Editor -> Config", type="primary", use_container_width=True):
        try:
            parsed = json.loads(config_text)
            # Validate structure
            if not isinstance(parsed, dict):
                raise ValueError("Config must be a dictionary")
            # Ensure required keys exist
            if "instruments" not in parsed:
                parsed["instruments"] = []
            if "smart_api" not in parsed:
                parsed["smart_api"] = {}
            # Write to file
            CONFIG_PATH.write_text(json.dumps(parsed, indent=2))
            # Reload config manager
            st.session_state.config_manager = ConfigManager(CONFIG_PATH, logger=enqueue_log)
            st.session_state.bot.config_manager = st.session_state.config_manager
            # Sync SmartAPI credentials
            creds = st.session_state.config_manager.get_smart_api_config()
            st.session_state.smart_client.update_credentials(creds)
            if creds.get("enabled"):
                st.session_state.smart_client.connect()
            enqueue_log("Configuration updated from editor.")
            st.success("Configuration updated!")
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

# Auto-refresh footer
is_running = st.session_state.bot and st.session_state.bot.is_running()
if is_running:
    if time.time() - st.session_state.last_refresh > 5:
        st.session_state.last_refresh = time.time()
        st.rerun()
