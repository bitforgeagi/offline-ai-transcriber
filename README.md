# Offline AI Transcriber

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
- C++ compiler (gcc/clang)
- CMake
- Basic command line knowledge
- ~2GB storage for models

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/bitforgeagi/offline-ai-transcriber.git
cd offline-ai-transcriber
```

2. Build whisper.cpp:
```bash
cd whisper.cpp
make
```

3. Download a Whisper model:
```bash
# For beginners, start with the small model
bash ./models/download-ggml-model.sh small

# For better accuracy, try the medium model
bash ./models/download-ggml-model.sh medium
```

### Understanding the Models

Whisper offers several model sizes:
- `tiny`: ~75MB - Fastest, least accurate
- `base`: ~142MB - Good balance for learning
- `small`: ~466MB - Recommended starting point
- `medium`: ~1.5GB - Better accuracy
- `large`: ~3GB - Best accuracy, slower

We recommend starting with `small` to learn the basics, then experimenting with other sizes.

## Usage & Learning

### Basic Transcription
```bash
# Transcribe an audio file
./main -m models/ggml-small.bin -f audio.mp3

# With more detailed output
./main -m models/ggml-small.bin -f audio.mp3 --print-timestamps
```

### Learning Exercises

1. **Basic Transcription**
   - Try transcribing different audio files
   - Compare accuracy between model sizes
   - Experiment with different audio formats

2. **Performance Testing**
   - Time transcription speeds
   - Monitor CPU usage
   - Compare memory requirements

3. **Advanced Features**
   - Try different languages
   - Experiment with timestamp generation
   - Test different audio qualities

## Project Structure
```
project/
├── whisper.cpp/           # Core C++ implementation
├── models/               # Model storage
├── examples/            # Example usage scenarios
├── docs/               # Detailed documentation
└── exercises/         # Learning exercises
```

## Common Learning Challenges

### Memory Management
- Understanding model loading
- Managing RAM usage
- Optimizing for your hardware

### Performance Tuning
- CPU thread optimization
- Model size selection
- Audio preprocessing

## Resources for Learning

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [whisper.cpp Documentation](https://github.com/ggerganov/whisper.cpp)

## License

MIT License - Use this for learning and your own projects!

## Acknowledgments

- OpenAI for the Whisper model
- Georgi Gerganov for whisper.cpp
- The open-source AI community