# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['desktop_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('streamlit_app.py', '.'), ('trading_bot.py', '.')],
    hiddenimports=['SmartApi', 'streamlit', 'pandas', 'numpy', 'pyotp', 'requests', 'pywebview'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AngelOneTradingBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
