import os
import whisper
import logging
from pydub import AudioSegment
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_large_file(file_path, size_threshold_mb=100):
    """Check if the file size exceeds a certain threshold (in MB)"""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return file_size_mb > size_threshold_mb

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video file using ffmpeg directly"""
    logging.info(f"Extracting audio from video: {video_path}")
    exit_code = os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}"')
    if exit_code != 0:
        raise Exception("FFmpeg failed to extract audio from the video file.")

def convert_audio_format(file_path, target_format="mp3"):
    """Convert audio file to a target format using pydub"""
    logging.info(f"Converting audio file to {target_format}: {file_path}")
    audio = AudioSegment.from_file(file_path)
    converted_path = file_path.rsplit('.', 1)[0] + f".{target_format}"
    audio.export(converted_path, format=target_format)
    return converted_path

def split_audio(file_path, chunk_length_ms=60000):
    """Split audio file into chunks of specified length in milliseconds"""
    logging.info(f"Splitting audio file into chunks: {file_path}")
    audio = AudioSegment.from_file(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def transcribe_audio(file_path):
    """Transcribe audio file using Whisper with English as the default language"""
    logging.info(f"Starting transcription for: {file_path}")
    model = whisper.load_model("base")
    # Set English as the default language
    result = model.transcribe(file_path, language="en", verbose=True)
    return result["text"]

def transcribe_large_audio_parallel(file_path):
    """Transcribe large audio file by splitting it into chunks and processing in parallel"""
    chunks = split_audio(file_path)
    full_transcription = ""
    chunk_paths = []
    
    # Export all chunks first
    for i, chunk in enumerate(chunks):
        chunk_path = f"temp_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, chunk_path in enumerate(chunk_paths):
            logging.info(f"Transcribing chunk {i + 1}/{len(chunks)}: {chunk_path}")
            futures.append(executor.submit(transcribe_audio, chunk_path))
        
        for future in futures:
            full_transcription += future.result() + " "
    
    # Clean up temp files
    for chunk_path in chunk_paths:
        os.remove(chunk_path)
    
    return full_transcription.strip()

def format_transcription_with_timestamps(result):
    """Format transcription with timestamps"""
    formatted_text = ""
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        formatted_text += f"[{start_time:.2f} - {end_time:.2f}] {text}\n"
    return formatted_text

def validate_file_path(file_path):
    """Validate the file path and ensure it exists"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in ['.mp3', '.mp4', '.wav', '.flac']:
        raise ValueError("Only MP3, MP4, WAV, and FLAC files are supported!")

def main():
    try:
        # Get file path from user
        file_path = input("Please enter the path to your audio/video file: ").strip()
        
        # Validate file path
        validate_file_path(file_path)
        
        # Handle large files
        if is_large_file(file_path):
            logging.warning("File is large. Splitting into chunks for processing...")
            transcription = transcribe_large_audio_parallel(file_path)
        else:
            # Handle MP4 files by extracting audio first
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.mp4':
                temp_audio_path = file_path.replace('.mp4', '_temp.mp3')
                extract_audio_from_video(file_path, temp_audio_path)
                file_path = temp_audio_path
            
            # Convert non-MP3 files to MP3
            if file_ext not in ['.mp3', '.mp4']:
                file_path = convert_audio_format(file_path)
            
            # Perform transcription
            transcription = transcribe_audio(file_path)
        
        # Create output file path
        output_path = file_path.rsplit('.', 1)[0] + '_transcription.txt'
        
        # Save transcription
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        logging.info(f"Transcription completed successfully! Output saved to: {output_path}")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()