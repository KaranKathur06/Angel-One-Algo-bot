import json
import tkinter as tk
from datetime import datetime
from queue import Empty, Queue
from tkinter import messagebox, ttk
from typing import Callable, Optional

from trading_bot import (
    CONFIG_PATH,
    ConfigManager,
    DataProvider,
    InstrumentConfig,
    SmartAPIClient,
    TIMEFRAME_MAP,
    TradingBot,
)

LOG_MAX_LINES = 500
TIMEFRAME_OPTIONS = list(TIMEFRAME_MAP.keys())


class TradingBotGUI(tk.Tk):
    """Tkinter-based GUI wrapper for the RSI trading bot."""

    def __init__(self) -> None:
        super().__init__()
        self.title("RSI Trading Bot")
        self.geometry("1200x720")
        self.minsize(1100, 650)

        self.log_queue: "Queue[str]" = Queue()
        self.config_manager = ConfigManager(CONFIG_PATH, logger=self.enqueue_log)
        self.data_provider = DataProvider(logger=self.enqueue_log)
        self.smart_client = SmartAPIClient(self.config_manager.get_smart_api_config(), logger=self.enqueue_log)
        if self.smart_client.credentials.get("enabled"):
            self.smart_client.connect()
        self.bot = TradingBot(
            self.config_manager,
            self.data_provider,
            self.smart_client,
            logger=self.enqueue_log,
        )

        self._build_style()
        self._build_ui()

        self.after(500, self._process_log_queue)
        self.after(1000, self._refresh_status)
        self.after(2000, self._refresh_positions_table)
        self.after(5000, self._refresh_instrument_table_periodic)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.enqueue_log("GUI initialized.")

    # ------------------------------------------------------------------ UI Setup
    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background="#1f1f1f")
        style.configure("TFrame", background="#1f1f1f")
        style.configure("TLabel", background="#1f1f1f", foreground="#f5f5f5")
        style.configure("Accent.TButton", padding=6)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.dashboard_tab = ttk.Frame(notebook, padding=15)
        self.instrument_tab = ttk.Frame(notebook, padding=15)
        self.positions_tab = ttk.Frame(notebook, padding=15)
        self.settings_tab = ttk.Frame(notebook, padding=15)

        notebook.add(self.dashboard_tab, text="Dashboard")
        notebook.add(self.instrument_tab, text="Instruments")
        notebook.add(self.positions_tab, text="Positions")
        notebook.add(self.settings_tab, text="Settings")

        self._build_dashboard()
        self._build_instruments()
        self._build_positions()
        self._build_settings()

    def _build_dashboard(self) -> None:
        header = ttk.Frame(self.dashboard_tab)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="System Status:", font=("Segoe UI", 12, "bold")).pack(side="left")
        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(header, textvariable=self.status_var, font=("Segoe UI", 12)).pack(side="left", padx=(8, 0))

        btn_frame = ttk.Frame(self.dashboard_tab)
        btn_frame.pack(fill="x", pady=(0, 15))
        ttk.Button(btn_frame, text="Start Trading", style="Accent.TButton", command=self.start_bot).pack(
            side="left", padx=(0, 10)
        )
        ttk.Button(btn_frame, text="Stop Trading", command=self.stop_bot).pack(side="left")

        log_label = ttk.Label(self.dashboard_tab, text="Live Logs / Status")
        log_label.pack(anchor="w")
        log_container = ttk.Frame(self.dashboard_tab)
        log_container.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            log_container,
            wrap="word",
            height=20,
            state="disabled",
            background="#111",
            foreground="#e5e5e5",
            insertbackground="#fff",
        )
        log_scroll = ttk.Scrollbar(log_container, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _build_instruments(self) -> None:
        action_frame = ttk.Frame(self.instrument_tab)
        action_frame.pack(fill="x", pady=(0, 10))
        ttk.Button(action_frame, text="Add Instrument", command=self._add_instrument).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="Edit Selected", command=self._edit_instrument).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="Remove Selected", command=self._remove_instrument).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="Refresh", command=self._refresh_instrument_table).pack(side="left")

        columns = ("symbol", "quantity", "timeframe", "rsi_level", "target", "stop")
        self.instrument_tree = ttk.Treeview(
            self.instrument_tab,
            columns=columns,
            show="headings",
            height=15,
        )
        headings = {
            "symbol": "Symbol",
            "quantity": "Qty",
            "timeframe": "Interval",
            "rsi_level": "RSI Level",
            "target": "Target (%)",
            "stop": "Stop (%)",
        }
        for col, label in headings.items():
            self.instrument_tree.heading(col, text=label)
            self.instrument_tree.column(col, stretch=True, anchor="center")
        self.instrument_tree.pack(fill="both", expand=True)
        self._refresh_instrument_table()

    def _build_positions(self) -> None:
        columns = ("symbol", "quantity", "entry", "current", "pnl", "entry_time", "strategy")
        self.positions_tree = ttk.Treeview(
            self.positions_tab,
            columns=columns,
            show="headings",
            height=15,
        )
        headings = {
            "symbol": "Symbol",
            "quantity": "Qty",
            "entry": "Entry Price",
            "current": "Current Price",
            "pnl": "P&L (%)",
            "entry_time": "Entry Time",
            "strategy": "Strategy",
        }
        for col, label in headings.items():
            self.positions_tree.heading(col, text=label)
            self.positions_tree.column(col, stretch=True, anchor="center")
        self.positions_tree.pack(fill="both", expand=True)

    def _build_settings(self) -> None:
        button_frame = ttk.Frame(self.settings_tab)
        button_frame.pack(fill="x", pady=(0, 10))
        ttk.Button(button_frame, text="Reload from Disk", command=self._reload_config).pack(side="left", padx=(0, 8))
        ttk.Button(button_frame, text="Save Editor -> Config", command=self._save_config_from_editor).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(button_frame, text="Force Save to File", command=self._force_save_config).pack(side="left")

        self.config_text = tk.Text(
            self.settings_tab,
            wrap="none",
            background="#111",
            foreground="#e5e5e5",
            insertbackground="#fff",
            font=("Consolas", 10),
        )
        x_scroll = ttk.Scrollbar(self.settings_tab, orient="horizontal", command=self.config_text.xview)
        y_scroll = ttk.Scrollbar(self.settings_tab, orient="vertical", command=self.config_text.yview)
        self.config_text.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.config_text.pack(fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        self._load_config_into_editor()

    # ------------------------------------------------------------------ Callbacks
    def enqueue_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def start_bot(self) -> None:
        if self.bot.is_running():
            self.enqueue_log("Bot already running.")
            return
        self.bot.start()
        self.enqueue_log("Start command issued.")

    def stop_bot(self) -> None:
        if not self.bot.is_running():
            self.enqueue_log("Bot already stopped.")
            return
        self.bot.stop()
        self.enqueue_log("Stop command issued.")

    def _add_instrument(self) -> None:
        InstrumentForm(self, title="Add Instrument", on_save=self._save_instrument)

    def _edit_instrument(self) -> None:
        symbol = self._get_selected_instrument_symbol()
        if not symbol:
            messagebox.showinfo("Edit Instrument", "Select an instrument to edit.")
            return
        instrument = next((inst for inst in self.config_manager.get_instruments() if inst.symbol == symbol), None)
        if not instrument:
            messagebox.showerror("Edit Instrument", "Instrument no longer exists.")
            self._refresh_instrument_table()
            return
        InstrumentForm(self, title=f"Edit {symbol}", instrument=instrument, on_save=self._save_instrument)

    def _remove_instrument(self) -> None:
        symbol = self._get_selected_instrument_symbol()
        if not symbol:
            messagebox.showinfo("Remove Instrument", "Select an instrument to remove.")
            return
        if not messagebox.askyesno("Remove Instrument", f"Remove {symbol}?"):
            return
        if self.config_manager.remove_instrument(symbol):
            self.enqueue_log(f"{symbol} removed.")
            self._refresh_instrument_table()
        else:
            messagebox.showerror("Remove Instrument", "Instrument not found.")

    def _save_instrument(self, config: InstrumentConfig) -> None:
        self.config_manager.upsert_instrument(config)
        self.enqueue_log(f"{config.symbol} saved.")
        self._refresh_instrument_table()

    def _get_selected_instrument_symbol(self) -> Optional[str]:
        selection = self.instrument_tree.selection()
        if not selection:
            return None
        item = self.instrument_tree.item(selection[0])
        return item["values"][0] if item["values"] else None

    def _reload_config(self) -> None:
        self.config_manager.reload()
        self._load_config_into_editor()
        self._refresh_instrument_table()
        self._sync_smart_api_credentials()
        self.enqueue_log("Configuration reloaded from disk.")

    def _save_config_from_editor(self) -> None:
        raw_text = self.config_text.get("1.0", "end").strip()
        if not raw_text:
            messagebox.showerror("Config", "Config text cannot be empty.")
            return
        try:
            parsed = json.loads(raw_text)
            self.config_manager.replace_config(parsed)
            self._refresh_instrument_table()
            self._sync_smart_api_credentials()
            self.enqueue_log("Configuration updated from editor.")
        except json.JSONDecodeError as exc:
            messagebox.showerror("Config", f"Invalid JSON: {exc}")
        except ValueError as exc:
            messagebox.showerror("Config", str(exc))

    def _force_save_config(self) -> None:
        self.config_manager.save()
        self.enqueue_log("Configuration force-saved to disk.")

    # ------------------------------------------------------------------ Refresh Helpers
    def _refresh_instrument_table(self) -> None:
        for row in self.instrument_tree.get_children():
            self.instrument_tree.delete(row)
        for inst in self.config_manager.get_instruments():
            self.instrument_tree.insert(
                "",
                "end",
                values=(
                    inst.symbol,
                    inst.quantity,
                    inst.timeframe,
                    inst.rsi_level,
                    inst.profit_target_pct,
                    inst.stop_loss_pct,
                ),
            )

    def _refresh_positions_table(self) -> None:
        for row in self.positions_tree.get_children():
            self.positions_tree.delete(row)
        positions = self.bot.get_positions_snapshot()
        instruments = {inst.symbol.upper(): inst for inst in self.config_manager.get_instruments()}
        for row in positions:
            inst = instruments.get(row["symbol"])
            strategy = f"RSI {inst.rsi_level} @ {inst.timeframe}" if inst else "RSI Strategy"
            entry_dt = datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            self.positions_tree.insert(
                "",
                "end",
                values=(
                    row["symbol"],
                    row["quantity"],
                    f"{row['entry_price']:.2f}",
                    f"{row['current_price']:.2f}",
                    f"{row['pnl_pct']:.2f}",
                    entry_dt,
                    strategy,
                ),
            )
        self.after(2000, self._refresh_positions_table)

    def _refresh_instrument_table_periodic(self) -> None:
        self._refresh_instrument_table()
        self.after(5000, self._refresh_instrument_table_periodic)

    def _refresh_status(self) -> None:
        status = "Running" if self.bot.is_running() else "Stopped"
        self.status_var.set(status)
        self.after(1000, self._refresh_status)

    def _process_log_queue(self) -> None:
        updated = False
        while True:
            try:
                message = self.log_queue.get_nowait()
            except Empty:
                break
            self._append_log(message)
            updated = True
        if updated:
            self.log_text.see("end")
        self.after(500, self._process_log_queue)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        current_lines = int(self.log_text.index("end-1c").split(".")[0])
        if current_lines > LOG_MAX_LINES:
            self.log_text.delete("1.0", "2.0")
        self.log_text.configure(state="disabled")

    def _load_config_into_editor(self) -> None:
        self.config_text.configure(state="normal")
        self.config_text.delete("1.0", "end")
        pretty = json.dumps(self.config_manager.get_raw_config(), indent=2)
        self.config_text.insert("1.0", pretty)
        self.config_text.configure(state="normal")

    def _sync_smart_api_credentials(self) -> None:
        creds = self.config_manager.get_smart_api_config()
        self.smart_client.update_credentials(creds)
        if creds.get("enabled"):
            self.smart_client.connect()

    def _on_close(self) -> None:
        if self.bot.is_running():
            if not messagebox.askyesno("Exit", "Bot is running. Stop and exit?"):
                return
            self.bot.stop()
        self.destroy()


class InstrumentForm(tk.Toplevel):
    """Modal form for creating or editing an instrument."""

    def __init__(
        self,
        master: tk.Tk,
        title: str,
        instrument: Optional[InstrumentConfig] = None,
        on_save: Optional[Callable[[InstrumentConfig], None]] = None,
    ):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.instrument = instrument
        self.on_save = on_save

        self.symbol_var = tk.StringVar(value=instrument.symbol if instrument else "")
        self.timeframe_var = tk.StringVar(value=instrument.timeframe if instrument else TIMEFRAME_OPTIONS[2])
        self.rsi_length_var = tk.StringVar(value=str(instrument.rsi_length if instrument else 14))
        self.rsi_level_var = tk.StringVar(value=str(instrument.rsi_level if instrument else 30.0))
        self.target_var = tk.StringVar(value=str(instrument.profit_target_pct if instrument else 5.0))
        self.stop_var = tk.StringVar(value=str(instrument.stop_loss_pct if instrument else 10.0))
        self.quantity_var = tk.StringVar(value=str(instrument.quantity if instrument else 1))
        self.exchange_var = tk.StringVar(value=instrument.exchange if instrument else "NFO")
        self.symbol_token_var = tk.StringVar(value=instrument.symbol_token or "")
        self.product_type_var = tk.StringVar(value=instrument.product_type if instrument else "INTRADAY")

        self._build_form()
        self.transient(master)
        self.grab_set()
        self.wait_visibility()
        self.focus()

    def _build_form(self) -> None:
        frame = ttk.Frame(self, padding=15)
        frame.pack(fill="both", expand=True)

        def add_row(row: int, label: str, widget: tk.Widget) -> None:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
            widget.grid(row=row, column=1, sticky="ew", pady=4)

        symbol_entry = ttk.Entry(frame, textvariable=self.symbol_var)
        if self.instrument:
            symbol_entry.configure(state="disabled")

        add_row(0, "Symbol", symbol_entry)

        timeframe_combo = ttk.Combobox(
            frame, textvariable=self.timeframe_var, values=TIMEFRAME_OPTIONS, state="readonly"
        )
        add_row(1, "Interval", timeframe_combo)

        add_row(2, "RSI Length", ttk.Entry(frame, textvariable=self.rsi_length_var))
        add_row(3, "RSI Level", ttk.Entry(frame, textvariable=self.rsi_level_var))
        add_row(4, "Target %", ttk.Entry(frame, textvariable=self.target_var))
        add_row(5, "Stop %", ttk.Entry(frame, textvariable=self.stop_var))
        add_row(6, "Quantity", ttk.Entry(frame, textvariable=self.quantity_var))
        add_row(7, "Exchange", ttk.Entry(frame, textvariable=self.exchange_var))
        add_row(8, "Symbol Token", ttk.Entry(frame, textvariable=self.symbol_token_var))
        add_row(9, "Product Type", ttk.Entry(frame, textvariable=self.product_type_var))

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(button_frame, text="Save", style="Accent.TButton", command=self._handle_save).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side="left")

        frame.columnconfigure(1, weight=1)

    def _handle_save(self) -> None:
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showerror("Instrument", "Symbol cannot be empty.")
            return
        try:
            config = InstrumentConfig(
                symbol=symbol,
                timeframe=self.timeframe_var.get(),
                rsi_length=int(self.rsi_length_var.get()),
                rsi_level=float(self.rsi_level_var.get()),
                profit_target_pct=float(self.target_var.get()),
                stop_loss_pct=float(self.stop_var.get()),
                quantity=int(self.quantity_var.get()),
                exchange=self.exchange_var.get().strip().upper() or "NSE",
                symbol_token=self.symbol_token_var.get().strip() or None,
                product_type=self.product_type_var.get().strip().upper() or "INTRADAY",
            )
        except ValueError as exc:
            messagebox.showerror("Instrument", f"Invalid numeric value: {exc}")
            return
        if self.on_save:
            self.on_save(config)
        self.destroy()


if __name__ == "__main__":
    app = TradingBotGUI()
    app.mainloop()

