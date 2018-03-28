# -*- mode: python -*-

block_cipher = None


a = Analysis(['blivion.py'],
             pathex=['C:\\Users\\schilsm\\Dropbox\\Crick\\Programming\\Deployment\\Blivion'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='blivion',
          debug=False,
		  windowed=True,
          strip=False,
          upx=True,
          console=False, 
		  icon='blimp.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='blivion')
