# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
        (r'C:\Users\iuyam\AppData\Local\Programs\Python\Python313\Lib\site-packages\mediapipe\tasks\c', 'mediapipe/tasks/c'),
    ],
    datas=[
        ('hand_landmarker.task', '.'),
        ('hand_mouse.py', '.'),
        ('voice_module.py', '.'),
        ('hand_mouse_with_voice.py', '.'),
        (r'C:\Users\iuyam\AppData\Local\Programs\Python\Python313\Lib\site-packages\mediapipe', 'mediapipe'),
        (r'C:\Users\iuyam\AppData\Local\Programs\Python\Python313\Lib\site-packages\speech_recognition', 'speech_recognition'),    ],
    hiddenimports=[
        'mediapipe.tasks.c',
        'mediapipe.tasks.python.core.mediapipe_c_bindings',
        'speech_recognition',
        'pyperclip',     
    ],
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
    name='main',
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
    icon=['HandMouseIcon.ico'],
)
