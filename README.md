# Offline AI Transcriber

![Create Spectral Whisper Cyber Theme](https://github.com/user-attachments/assets/8e4bc601-8e6d-47b9-b591-199ccdc0b04f)

A step-by-step educational guide and implementation showing how to build a Graphical User Interface (GUI) for Whisper AI locally for audio transcription using whisper.cpp. This project demonstrates how to set up efficient, offline speech recognition without relying on cloud services. 

Learn more about [Bitforge Dynamics](https://bitforgedynamics.com) and our [Dark Engine Project](https://darkengine.ai).

## Educational Overview

This repository serves as a learning resource to understand:
- How to run AI models locally on your machine
- Setting up and using whisper.cpp for efficient CPU-based inference
- Basic audio processing and transcription workflows
- Practical implementation of offline AI tools

## Technical Background

### What is Whisper?
Whisper is an automatic speech recognition (ASR) system trained by OpenAI. While OpenAI provides cloud APIs, this project focuses on running Whisper locally using whisper.cpp, which is:
- Optimized for CPU usage
- No internet connection required
- No API costs
- Privacy-preserving (all processing happens on your machine)

### Why whisper.cpp?
The C++ implementation offers several advantages for learning:
- Faster inference than Python implementations
- Lower memory usage
- No Python dependencies required
- Great for understanding low-level AI deployment
## Installation Guide

### Prerequisites
- Python 3.10+ (You may need to install a different version depending on your system)
- Virtual environment (recommended)
- macOS (M1/M2) specific instructions included below

<img width="889" alt="Whisper GUI ss1" src="https://github.com/user-attachments/assets/6ea20b22-8981-4250-971d-b3c6f7ba049b" />

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/bitforgeagi/offline-ai-transcriber.git
cd offline-ai-transcriber
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

If you are using Windows, you can use the following command to create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. **IMPORTANT FOR M1/M2 MAC USERS**: 
Install PyTorch separately first (this is crucial - regular pip install won't work):
```bash
pip3 install --pre torch torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

4. Install other requirements:
```bash
pip install -r requirements.txt
```

5. Install audio dependencies (the brackets need quotes in shell):
```bash
pip install 'datasets[audio]'
```

### Installation Issues You May Encounter

#### M1/M2 Mac PyTorch Installation
If you see errors like:
```
ERROR: Could not find a version that satisfies the requirement torch
ERROR: No matching distribution found for torch
```
This is a common issue on M1/M2 Macs. The solution is to:
1. Skip the PyTorch installation in requirements.txt
2. Use the special installation command in step 3 above
3. Then proceed with the rest of the requirements

#### Shell Interpretation Issues
If you see:
```
zsh: no matches found: datasets[audio]
```
The shell is trying to interpret the square brackets. Always use quotes:
```bash
pip install 'datasets[audio]'
```

### Installation with Pre-bundled Models

Option 1: Download models separately:
1. Download models from Hugging Face or Github
2. Extract to `./models/` directory
3. Continue with regular installation

Option 2: Automatic download (slower):
- Skip the above steps
- Models will download automatically on first run

## Usage

1. Place your audio file in the project directory
2. Update the audio file path in transcribe.py (default: "test_audio.mp3")
3. Run the transcription:
```bash
python transcribe.py
```

### Output Format
The script provides:
- Complete transcription text
- Timestamped chunks showing when each part was spoken

## Features

- Offline transcription using Whisper AI (large-v3-turbo model)
- Timestamp support for precise audio segment identification
- Automatic GPU detection and utilization if available
- Optimized for Apple Silicon (M1/M2) Macs
- Support for long-form audio files

## Technical Details

### Components
- OpenAI's Whisper (large-v3-turbo model) - A fast and accurate speech recognition model
- 🤗 Transformers library - Provides the pipeline for easy model usage
- PyTorch - Powers the underlying computations
- Automatic hardware acceleration when available

### Model Reset Instructions

To force a fresh download:
```bash
rm -rf models/*  # Remove local models
```

### Model Directory Structure
```
models/
├── .locks/                    # Concurrent access management
├── models--openai--whisper-large-v3-turbo/
│   ├── snapshots/            # Model versions
│   │   └── [hash]/          # Version-specific files
│   │       ├── config.json   # Model configuration
│   │       ├── model.safetensors  # Model weights
│   │       └── ...          # Other model files
│   └── refs/                 # Version references
└── .gitkeep                  # Git directory marker
```

The model directory uses a structured format to manage:
- Concurrent access through `.locks`
- Version control through `snapshots`
- Configuration and weight files
- Safe tensor storage for efficient loading

### Dependencies
Core requirements:
- torch>=2.0.0
- torchaudio
- transformers>=4.36.0
- datasets
- accelerate>=0.27.0
- soundfile

Select a model using the --model flag:
```bash
python transcribe.py --model small  # Default
python transcribe.py --model tiny   # Fastest
python transcribe.py --model large  # Most accurate
```

### GUI Interface

The application includes a graphical interface with:
- Upload support for audio files
- Model selection dropdown
- Progress indication
- Scrollable output with timestamps
- Status updates

To run the GUI version:
```bash
python gui.py
```
<img width="891" alt="Whisper GUI ss2" src="https://github.com/user-attachments/assets/1c4131aa-06f3-48bb-bf3b-02a4cc6207bb" />

## License

MIT License - feel free to use this code for your projects!

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the Transformers library
- The PyTorch team for M1/M2 support
