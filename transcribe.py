import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import warnings
import argparse

# Add these constants at the top
MODELS = {
    'tiny': "openai/whisper-tiny",        # 39M parameters
    'base': "openai/whisper-base",        # 74M parameters
    'small': "openai/whisper-small",      # 244M parameters
    'medium': "openai/whisper-medium",    # 769M parameters
    'large': "openai/whisper-large-v3-turbo"  # 809M parameters
}

MODEL_DIR = "models"  # Local directory to store models

def setup_whisper(model_size='small'):  # Default to small for better speed
    # Filter out specific warnings
    warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated")
    
    # Set device and dtype
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load model with optimizations, saving to local directory
    model_id = MODELS.get(model_size, MODELS['small'])
    print(f"\nDownloading/Loading {model_size} model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=False,
        cache_dir=MODEL_DIR
    )
    print(f"Model loaded successfully!")
    model.to(device)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=False,
        cache_dir=MODEL_DIR
    )

    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe

def transcribe_audio(audio_path, pipeline):
    """Transcribe an audio file using Whisper"""
    try:
        # Transcribe with timestamps and chunking
        result = pipeline(
            audio_path, 
            return_timestamps=True,
            chunk_length_s=30,  # Process 30-second chunks
            stride_length_s=5,  # 5-second overlap between chunks
            batch_size=8,      # Process multiple chunks in parallel
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
                "forced_decoder_ids": None
            }
        )
        return result
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def format_timestamp(timestamp):
    """Convert timestamp to readable format"""
    if isinstance(timestamp, tuple):
        start, end = timestamp
        return f"[{start:.2f}s -> {end:.2f}s]"
    return f"[{timestamp:.2f}s]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS.keys(), default='small',
                       help='Model size to use')
    parser.add_argument('--file', default='test_audio_file.mp3',
                       help='Audio file to transcribe')
    args = parser.parse_args()
    
    print(f"\nUsing {args.model} model...")
    pipe = setup_whisper(args.model)
    
    result = transcribe_audio(args.file, pipe)
    
    if result:
        print("\nTranscription:")
        print(result["text"].strip())
        
        print("\nTimestamped chunks:")
        for chunk in result["chunks"]:
            timestamp = format_timestamp(chunk["timestamp"])
            print(f"{timestamp} {chunk['text'].strip()}")

if __name__ == "__main__":
    main() 