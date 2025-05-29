import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from echosum.transcript.open_ai import transcribe_audio
from echosum.summary.open_ai import summarize_text
from echosum.audiorecord.convert_wav_to_mp3 import convert_wav_to_mp3_voice

class File_handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            if event.src_path.lower().endswith(".wav") :
                convert_wav_to_mp3_voice(event.src_path)   
            
            if event.src_path.lower().endswith(".mp3"):
                text = transcribe_audio(event.src_path)
                print("full transcription : ", text)
                summary = summarize_text(text)
                print("summary", summary)

def watch_directory(path):
    event_handler = File_handler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)  # Set recursive to True if you want to watch subdirectories
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Define the directory you want to watch
    watch_directory_path = "audio_recordings"  # <<< IMPORTANT: Change this to your desired directory

    # Create the directory if it doesn't exist
    if not os.path.exists(watch_directory_path):
        os.makedirs(watch_directory_path)
        print(f"Created watch directory: {watch_directory_path}")

    print(f"Watching directory for new .mp3 files: {watch_directory_path}")
    watch_directory(watch_directory_path)