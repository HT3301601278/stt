# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 获取项目根目录
project_root = os.path.abspath('.')

# 收集数据文件
datas = []
# 添加模板文件
datas += [('templates', 'templates')]
# 添加静态文件
datas += [('static', 'static')]
# 添加配置文件
datas += [('set.ini', '.')]
# 添加版本文件
datas += [('version.json', '.')]

# 收集faster-whisper相关数据
datas += collect_data_files('faster_whisper')
datas += collect_data_files('torch')

# 隐藏导入的模块
hiddenimports = []
# faster-whisper及其依赖
hiddenimports += collect_submodules('faster_whisper')
# torch相关（faster-whisper需要）
hiddenimports += collect_submodules('torch')
# 项目自定义模块
hiddenimports += ['stslib', 'stslib.cfg', 'stslib.tool']
# Flask和gevent相关
hiddenimports += ['gevent.monkey', 'gevent._gevent_c_abstract_linkable']
# werkzeug相关
hiddenimports += ['werkzeug.security', 'werkzeug.utils']

# 排除不需要的模块以减小体积
# 项目实际使用: torch, flask, gevent, faster_whisper, requests, werkzeug
excludes = [
    # 数据科学和可视化库（项目不需要）
    'matplotlib', 'matplotlib.pyplot', 'pylab',
    'scipy', 'scipy.stats',
    'pandas', 'pandas.io',
    'numpy.distutils',  # numpy核心保留，仅排除distutils
    # Jupyter相关（项目不需要）
    'jupyter', 'jupyter_client', 'jupyter_core',
    'notebook', 'nbconvert', 'nbformat',
    'IPython', 'ipykernel', 'ipywidgets',
    # GUI框架（项目不需要，仅用Flask Web界面）
    'tkinter', '_tkinter', 'Tkinter',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx', 'wxPython',
    # 测试框架（打包后不需要）
    'pytest', 'unittest2', 'nose', 'nose2', 'coverage', 'mock',
    # 文档工具（打包后不需要）
    'sphinx', 'docutils',
    # 打包工具（打包后不需要）
    'setuptools', 'pip', 'wheel',
]

block_cipher = None

a = Analysis(
    ['start.py'],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 过滤掉一些不需要的文件
def filter_binaries(binaries):
    filtered = []
    exclude_patterns = [
        'mkl', 'openblas',  # 大型数学库
        'tcl', 'tk',  # Tkinter相关
        'Qt5', 'Qt6',  # Qt框架
        '_testcapi', '_test',  # 测试相关
        'libopencv',  # OpenCV（如果不需要）
    ]
    for name, path, type_info in binaries:
        # 排除一些大的不必要的库
        if any(exclude in name.lower() for exclude in exclude_patterns):
            continue
        filtered.append((name, path, type_info))
    return filtered

def filter_datas(datas):
    filtered = []
    exclude_patterns = [
        'test', 'tests', 'Testing',  # 测试文件
        'doc', 'docs', 'documentation',  # 文档
        'example', 'examples', 'sample',  # 示例文件
        'LICENSE', 'README',  # 第三方包的文档（保留项目自己的）
        '.pyc', '.pyo',  # 编译文件
    ]
    for dest, source, type_info in datas:
        # 检查是否应该排除
        should_exclude = False
        for pattern in exclude_patterns:
            # 排除第三方库的文档和测试，但保留项目自己的
            if pattern in dest.lower() and not dest.startswith('.'):
                if 'site-packages' in source or 'Lib' in source:
                    should_exclude = True
                    break
        if not should_exclude:
            filtered.append((dest, source, type_info))
    return filtered

a.binaries = filter_binaries(a.binaries)
a.datas = filter_datas(a.datas)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='stt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加图标文件路径
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='stt'
)
