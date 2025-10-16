## 项目简介
STT 是一个专注于本地离线语音识别的开源工具，基于 faster-whisper 推理引擎和 Flask Web 服务实现。项目提供图形化网页前端、REST API 以及 OpenAI Speech API 兼容接口，可无缝替换包括 OpenAI Whisper、百度语音识别在内的多种云端服务，满足本地化部署、隐私保护与成本控制等场景需求。

## 目录
- [项目简介](#项目简介)
- [功能亮点](#功能亮点)
- [快速上手](#快速上手)
- [源码部署（Windows/Linux/macOS）](#源码部署windowslinuxmacos)
- [模型管理](#模型管理)
- [Web 界面使用指南](#web-界面使用指南)
- [REST API](#rest-api)
- [OpenAI Speech API 兼容模式](#openai-speech-api-兼容模式)
- [配置说明](#配置说明)
- [CUDA 与显卡加速](#cuda-与显卡加速)
- [常见问题](#常见问题)
- [发展方向与贡献](#发展方向与贡献)
- [许可证](#许可证)

## 功能亮点
- 离线部署：所有推理均在本地执行，数据不出内网，保护隐私。
- 多语言识别：支持中文、英语、法语、日语等十余种语言，可自动检测。
- 多种模型：内置 tiny 版本，支持 base/small/medium/large-v3 及 distil 系列模型，按需扩展。
- 丰富输出格式：支持纯文本、JSON 结构化字幕、SRT 带时间轴字幕。
- 图形化界面：拖拽上传、任务进度条、结果预览等交互化体验。
- API 能力：提供 `/api` REST 接口与 `/v1/audio/transcriptions` OpenAI 兼容接口，便于集成。
- GPU 加速：自动检测 NVIDIA GPU 并在配置完成后启用 CUDA，显著提升推理速度。

## 源码部署（Windows/Linux/macOS）
### 环境要求
- Python 3.9 – 3.11
- git（克隆项目所需）
- ffmpeg（音视频解码；Windows 需确保 `ffmpeg.exe`、`ffprobe.exe` 在项目根目录或 PATH 中）
- 可选：NVIDIA GPU + CUDA 11.x/12.x（用于加速）

### 部署步骤
1. **获取源码**
   ```bash
   git clone https://github.com/HT3301601278/stt.git
   cd stt
   ```
2. **创建虚拟环境并激活**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```
3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   若遇到依赖冲突，可执行 `pip install -r requirements.txt --no-deps`。需要 CUDA 加速时，先卸载 CPU 版本的 torch，再从官方 CUDA 源安装：
   ```bash
   pip uninstall -y torch
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
4. **准备 ffmpeg**
   - Windows：将 `ffmpeg.exe` 与 `ffprobe.exe` 放在项目根目录（或配置 PATH）。
   - Linux/macOS：使用系统包管理器安装，如 `apt install ffmpeg` 或 `brew install ffmpeg`。
5. **下载模型**
   - 内置 tiny 模型适合快速体验。
   - 访问 [模型下载页面](https://github.com/jianchang512/stt/releases/tag/0.0)，将解压后的模型文件夹放入 `models/` 目录。
6. **启动服务**
   ```bash
   python start.py
   ```
   程序会自动启动后台识别线程并打开浏览器，默认监听 `http://127.0.0.1:9977`。

## 模型管理
- faster-whisper 提供 tiny/base/small/medium/large-v3 等多个尺寸，模型越大准确率越高，对显存与内存需求也更高。
- 将下载的模型目录（例如 `models--Systran--faster-whisper-base`）直接放入 `models/` 下即可被自动识别。
- 通过 Web 前端或 API 选择 `model` 参数即可切换使用的模型。
- 若使用 distil 系列模型，保持原始名称（示例：`distil-large-v2`），程序会自动适配。

## Web 界面使用指南
- 拖拽或点击上传音频/视频文件，支持 mp4、mp3、wav、flac、aac、m4a 等格式。
- 可从 `upload/` 目录选择历史文件，或粘贴网络链接自动下载至本地。
- 支持选择输出格式（Text/JSON/SRT），并在识别过程中显示实时进度。
- 任务完成后，可在界面内复制文本或将字幕文件下载保存。

## REST API
- 地址：`POST http://127.0.0.1:9977/api`
- 表单字段：
  - `file`：二进制音视频文件
  - `language`：语言代码（`zh`、`en`、`fr`、`de`、`ja`、`ko`、`ru`、`es`、`th`、`it`、`pt`、`vi`、`ar`、`tr`、`auto`）
  - `model`：模型名称（如 `base`、`small`、`medium`、`large-v3`）
  - `response_format`：`text`、`json` 或 `srt`
- 响应：`code` 为 0 表示成功，`data` 字段返回转写内容。
```python
import requests

url = "http://127.0.0.1:9977/api"
files = {"file": open("sample.wav", "rb")}
data = {"language": "zh", "model": "base", "response_format": "json"}
response = requests.post(url, data=data, files=files, timeout=600)
print(response.json())
```

## OpenAI Speech API 兼容模式
- 地址：`POST http://127.0.0.1:9977/v1/audio/transcriptions`
- 完整兼容 OpenAI SDK（`openai`/`openai-python`）。
```python
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://127.0.0.1:9977/v1")
with open("sample.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="tiny",
        file=audio_file,
        response_format="text"  # 支持 text、srt、json
    )
print(transcription.text)
```

## 配置说明
项目根目录的 `set.ini` 可用于自定义行为，常用字段如下：
- `web_address`：监听地址与端口（默认 `127.0.0.1:9977`）。
- `devtype`：推理设备，`cpu` 或 `cuda`。
- `beam_size`、`best_of`：控制解码质量与速度平衡。
- `temperature`：解码温度；设为 0 时使用贪心搜索。
- `vad`：是否启用语音活动检测，减少静音片段误识别。
- `condition_on_previous_text`：是否依据上文上下文进行解码。
- `initial_prompt_zh`：中文场景的初始提示词，可引导输出风格。

修改配置后需重新启动服务生效。

## CUDA 与显卡加速
1. 安装或升级 NVIDIA 显卡驱动。
2. 安装匹配版本的 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 与 [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)。
3. 打开命令行执行 `nvcc --version` 与 `nvidia-smi` 确认环境配置正确。
4. 在 `set.ini` 中将 `devtype=cpu` 修改为 `devtype=cuda`，重新启动程序。
5. 运行 `python testcuda.py` 验证 CUDA 版本关联是否成功。

注意：`large/large-v3` 模型对显存要求较高，建议显存 ≥ 12GB；显存不足时可改用 `medium` 或更小模型。

## 常见问题
- **识别中突然退出？** 通常是 CUDA 环境缺失或显存不足，确保正确安装 cuDNN，或改用更小模型。
- **提示 `cublasXX.dll` 不存在？** 从 [cuBLAS 下载包](https://github.com/jianchang512/stt/releases/download/0.0/cuBLAS_win.7z) 解压，将 DLL 拷贝到 `C:\Windows\System32`。
- **中文输出繁体？** 这是模型特性，可在后处理中转换或在 `initial_prompt_zh` 中加入“输出简体中文”提示。
- **控制台出现 onnxruntime 相关 WARNING？** 属于已知日志，不影响使用，可忽略。

## 发展方向与贡献
- 欢迎提交 Issue 或 Pull Request 反馈问题、贡献特性。
- 计划引入的能力包含：批量任务队列、长音频切片优化、自动字幕段落合并、多语言界面优化等。
- 如果该项目对你有帮助，欢迎 Star 支持。

## 许可证
本项目基于 [GPL-3.0](./LICENSE) 协议开源，使用时请遵守相关条款。