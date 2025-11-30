"""
Streamlit-based web GUI for the RSI Trading Bot.
Runs headless on servers and accessible via web browser.
"""
import json
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
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
    page_title="RSI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "bot" not in st.session_state:
    st.session_state.bot = None
if "config_manager" not in st.session_state:
    st.session_state.config_manager = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()


def enqueue_log(message: str) -> None:
    """Add a log message to the session state."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.logs.append(log_entry)
    # Keep only last 500 lines
    if len(st.session_state.logs) > 500:
        st.session_state.logs = st.session_state.logs[-500:]


def initialize_bot() -> None:
    """Initialize the trading bot components."""
    if st.session_state.config_manager is None:
        st.session_state.config_manager = ConfigManager(CONFIG_PATH)
        data_provider = DataProvider()
        smart_client = SmartAPIClient(
            st.session_state.config_manager.get_smart_api_config()
        )
        if smart_client.credentials.get("enabled"):
            smart_client.connect()
        st.session_state.bot = TradingBot(
            st.session_state.config_manager,
            data_provider,
            smart_client,
        )
        enqueue_log("Bot initialized.")


# Initialize on first run
initialize_bot()

# Sidebar
with st.sidebar:
    st.title("📈 RSI Trading Bot")
    st.divider()

    # System Status
    is_running = st.session_state.bot and st.session_state.bot.thread and st.session_state.bot.thread.is_alive()
    status = "🟢 Running" if is_running else "🔴 Stopped"
    st.metric("System Status", status.split()[-1])

    st.divider()

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", type="primary", use_container_width=True):
            is_running = st.session_state.bot.thread and st.session_state.bot.thread.is_alive()
            if is_running:
                st.warning("Bot already running.")
            else:
                st.session_state.bot.start()
                st.rerun()

    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            is_running = st.session_state.bot.thread and st.session_state.bot.thread.is_alive()
            if not is_running:
                st.warning("Bot already stopped.")
            else:
                st.session_state.bot.stop()
                st.rerun()

    st.divider()
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Instruments", "💰 Positions", "⚙️ Settings"])

# ==================== DASHBOARD TAB ====================
with tab1:
    st.header("Dashboard")

    # Status and controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        is_running = st.session_state.bot.thread and st.session_state.bot.thread.is_alive()
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
    is_running = st.session_state.bot and st.session_state.bot.thread and st.session_state.bot.thread.is_alive()
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
                    help="e.g., NIFTY, BANKNIFTY",
                )
                timeframe = st.selectbox("Interval *", TIMEFRAME_OPTIONS, index=TIMEFRAME_OPTIONS.index(existing.timeframe) if existing and existing.timeframe in TIMEFRAME_OPTIONS else 2)
                rsi_length = st.number_input("RSI Length", min_value=1, max_value=100, value=existing.rsi_length if existing else 14)
                rsi_level = st.number_input("RSI Level", min_value=0.0, max_value=100.0, value=float(existing.rsi_level) if existing else 30.0, step=0.1)

            with col2:
                profit_target = st.number_input("Profit Target (%)", min_value=0.0, max_value=100.0, value=float(existing.profit_target_pct) if existing else 5.0, step=0.1)
                stop_loss = st.number_input("Stop Loss (%)", min_value=0.0, max_value=100.0, value=float(existing.stop_loss_pct) if existing else 10.0, step=0.1)
                quantity = st.number_input("Quantity", min_value=1, value=existing.quantity if existing else 1)
                exchange = st.text_input("Exchange", value=existing.exchange if existing else "NFO", help="e.g., NSE, NFO")

            symbol_token = st.text_input("Symbol Token (SmartAPI)", value=existing.symbol_token if existing else "", help="Leave blank if unknown")
            product_type = st.text_input("Product Type", value=existing.product_type if existing else "INTRADAY", help="e.g., INTRADAY, DELIVERY")

            submitted = st.form_submit_button("💾 Save", type="primary", use_container_width=True)

            if submitted:
                if not symbol.strip():
                    st.error("Symbol cannot be empty.")
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

    if st.button("🔄 Refresh Positions", use_container_width=True):
        st.rerun()

    # Get positions from bot
    with st.session_state.bot.lock:
        positions = list(st.session_state.bot.positions.values())
    
    instruments = {inst.symbol.upper(): inst for inst in st.session_state.config_manager.get_instruments()}

    if positions:
        data = []
        for pos in positions:
            inst = instruments.get(pos.symbol)
            strategy = f"RSI {inst.rsi_level} @ {inst.timeframe}" if inst else "RSI Strategy"
            entry_dt = datetime.fromtimestamp(pos.timestamp).strftime("%Y-%m-%d %H:%M:%S")

            data.append(
                {
                    "Symbol": pos.symbol,
                    "Qty": pos.quantity,
                    "Entry Price": f"{pos.entry_price:.2f}",
                    "Target Price": f"{pos.target_price:.2f}",
                    "Stop Price": f"{pos.stop_price:.2f}",
                    "Entry Time": entry_dt,
                    "Strategy": strategy,
                }
            )

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.info(f"Showing {len(positions)} open position(s). Current prices and P&L are calculated in real-time by the bot.")
    else:
        st.info("No open positions.")

    # Auto-refresh if bot is running
    is_running = st.session_state.bot and st.session_state.bot.thread and st.session_state.bot.thread.is_alive()
    if is_running:
        time.sleep(2)
        st.rerun()

# ==================== SETTINGS TAB ====================
with tab4:
    st.header("Configuration Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reload from Disk", use_container_width=True):
            # Reload by creating new ConfigManager
            st.session_state.config_manager = ConfigManager(CONFIG_PATH)
            st.session_state.bot.config_manager = st.session_state.config_manager
            # Sync SmartAPI credentials
            creds = st.session_state.config_manager.get_smart_api_config()
            st.session_state.bot.smart_client.update_credentials(creds)
            if creds.get("enabled"):
                st.session_state.bot.smart_client.connect()
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
            config_data = st.session_state.config_manager._config
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
    current_config = st.session_state.config_manager._config
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
            st.session_state.config_manager = ConfigManager(CONFIG_PATH)
            st.session_state.bot.config_manager = st.session_state.config_manager
            # Sync SmartAPI credentials
            creds = st.session_state.config_manager.get_smart_api_config()
            st.session_state.bot.smart_client.update_credentials(creds)
            if creds.get("enabled"):
                st.session_state.bot.smart_client.connect()
            enqueue_log("Configuration updated from editor.")
            st.success("Configuration updated!")
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

# Auto-refresh footer
is_running = st.session_state.bot and st.session_state.bot.thread and st.session_state.bot.thread.is_alive()
if is_running:
    if time.time() - st.session_state.last_refresh > 5:
        st.session_state.last_refresh = time.time()
        st.rerun()

