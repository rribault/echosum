import os
import torch
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import PydubException
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook # For progress bar

# Global variable for the pipeline to initialize it only once if processing multiple files
DIARIZATION_PIPELINE = None

def initialize_diarization_pipeline(hf_token: str = None):
    """
    Initializes the pyannote.audio speaker diarization pipeline.
    Caches the pipeline in a global variable to avoid re-initialization.

    Args:
        hf_token (str, optional): Hugging Face access token.
            If None, will try to use a token from `huggingface-cli login`.
    """
    global DIARIZATION_PIPELINE
    if DIARIZATION_PIPELINE is None:
        print("Initializing speaker diarization pipeline (this may take a moment)...")
        try:
            # Using a recent, robust model. Ensure you've accepted terms on Hugging Face.
            model_name = "pyannote/speaker-diarization-3.1"
            DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                model_name,
                use_auth_token=hf_token if hf_token else True
            )
            
            if torch.cuda.is_available():
                print("Moving pipeline to GPU...")
                DIARIZATION_PIPELINE = DIARIZATION_PIPELINE.to(torch.device("cuda"))
            else:
                print("No GPU found. Pipeline will run on CPU (slower).")
            print(f"Pipeline '{model_name}' initialized successfully.")
        except Exception as e:
            print(f"Error initializing diarization pipeline: {e}")
            print("Please ensure you have an internet connection, accepted model terms on Hugging Face,")
            print("and a valid Hugging Face token (either via `huggingface-cli login` or passed as hf_token).")
            DIARIZATION_PIPELINE = None # Ensure it stays None if failed
            raise # Re-raise the exception to stop execution if pipeline fails.


def diarize_audio_file(
    mp3_file_path: str,
    output_txt_path: str,
    target_sample_rate: int = 16000 # Common sample rate for speech models
):
    """
    Performs speaker diarization on a single MP3 audio file and saves the results.

    Args:
        mp3_file_path (str): Path to the input MP3 audio file.
        output_txt_path (str): Path to save the diarization results (text file).
        target_sample_rate (int): Sample rate to convert audio to before processing.
    """
    if DIARIZATION_PIPELINE is None:
        print("Error: Diarization pipeline is not initialized. Call initialize_diarization_pipeline() first.")
        return

    if not os.path.exists(mp3_file_path):
        print(f"Error: Input MP3 file not found at '{mp3_file_path}'")
        return

    print(f"\nProcessing '{mp3_file_path}'...")

    try:
        # 1. Load MP3 and prepare audio data for pyannote.audio
        sound = AudioSegment.from_mp3(mp3_file_path)
        sound = sound.set_channels(1) # Convert to mono
        sound = sound.set_frame_rate(target_sample_rate) # Resample
        
        # Get samples as numpy array and normalize to float32 range [-1.0, 1.0]
        samples_int16 = np.array(sound.get_array_of_samples())
        samples_float32 = samples_int16.astype(np.float32) / 32768.0 # Max value for int16 is 32767

        # Convert to PyTorch tensor, shape (num_channels, num_samples)
        waveform_tensor = torch.from_numpy(samples_float32).unsqueeze(0) # Add channel dimension: (1, num_samples)
        
        audio_data_for_pipeline = {"waveform": waveform_tensor, "sample_rate": target_sample_rate}
        
        print(f"  Audio loaded: Duration={sound.duration_seconds:.2f}s, SampleRate={target_sample_rate}Hz")

        # 2. Perform Diarization
        print("  Performing speaker diarization...")
        with ProgressHook() as hook: # Visual progress bar in console
             diarization_result = DIARIZATION_PIPELINE(audio_data_for_pipeline, hook=hook)
        
        # 3. Save results to text file
        # The result is an 'Annotation' object. We can iterate over speaker turns.
        output_dir = os.path.dirname(output_txt_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"  Created output directory: {output_dir}")

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Diarization for: {os.path.basename(mp3_file_path)}\n")
            f.write(f"Total duration analyzed: {sound.duration_seconds:.2f} seconds\n")
            f.write("-----------------------------------------------------------\n")
            if not diarization_result.labels():
                f.write("No speakers detected or diarization result is empty.\n")
            else:
                for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
                    f.write(f"[{turn.start:.2f}s - {turn.end:.2f}s] Speaker: {speaker_label}\n")
        
        print(f"  Diarization results saved to '{output_txt_path}'")
        num_speakers = len(diarization_result.labels())
        print(f"  Detected {num_speakers} speaker(s): {', '.join(sorted(diarization_result.labels()))}")

    except PydubException as e:
        print(f"Error processing MP3 file '{mp3_file_path}' with pydub: {e}")
        print("  Ensure FFmpeg is installed and accessible, and the file is a valid MP3.")
    except RuntimeError as e: # Catches potential CUDA errors or other runtime issues from pyannote
        print(f"Runtime error during diarization for '{mp3_file_path}': {e}")
        if "CUDA out of memory" in str(e):
            print("  Try reducing audio file length or running on a GPU with more memory / on CPU.")
    except Exception as e:
        print(f"An unexpected error occurred processing '{mp3_file_path}': {e}")


def diarize_mp3_folder(input_folder: str, output_folder: str, hf_token: str = None):
    """
    Processes all MP3 files in a given folder, performs speaker diarization,
    and saves results to an output folder.

    Args:
        input_folder (str): Path to the folder containing input MP3 files.
        output_folder (str): Path to the folder where .txt results will be saved.
        hf_token (str, optional): Hugging Face access token.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: '{output_folder}'")

    try:
        initialize_diarization_pipeline(hf_token=hf_token)
    except Exception: # Catch initialization errors
        print("Exiting due to pipeline initialization failure.")
        return

    if DIARIZATION_PIPELINE is None: # Double check after initialization attempt
        print("Exiting as diarization pipeline could not be initialized.")
        return

    mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

    if not mp3_files:
        print(f"No MP3 files found in '{input_folder}'.")
        return

    print(f"\nFound {len(mp3_files)} MP3 file(s) in '{input_folder}'. Starting processing...")

    for mp3_filename in mp3_files:
        mp3_file_path = os.path.join(input_folder, mp3_filename)
        base_filename, _ = os.path.splitext(mp3_filename)
        output_txt_path = os.path.join(output_folder, base_filename + "_diarization.txt")
        
        diarize_audio_file(mp3_file_path, output_txt_path)

    print("\nAll MP3 files processed.")


if __name__ == "__main__":
    # --- Configuration ---
    # !! IMPORTANT: If you don't have `huggingface-cli login` configured, provide your token here.
    # Example: HUGGING_FACE_TOKEN = "hf_YOUR_ACTUAL_TOKEN_HERE" 
    HUGGING_FACE_TOKEN = None # Set your token string here or ensure you've logged in via CLI

    # Specify your input and output folders
    # Create these folders and place some MP3 files in the input_audio_dir
    # For example, use files from your previous recording script.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    input_audio_dir = os.path.join(current_script_dir, "sample_mp3s_for_diarization") 
    output_results_dir = os.path.join(current_script_dir, "diarization_results")

    # Create dummy MP3s for testing if the input folder doesn't exist
    # (using the WAV-to-MP3 converter from a previous step)
    if not os.path.exists(input_audio_dir):
        print(f"Input directory '{input_audio_dir}' not found. Creating dummy MP3s for testing.")
        os.makedirs(input_audio_dir, exist_ok=True)
        
        # Attempt to create dummy files (requires functions from previous steps)
        try:
            # Create dummy WAV (from previous example, simplified)
            from scipy.io.wavfile import write as write_wav
            dummy_wav_path1 = os.path.join(input_audio_dir, "dummy_speakerA.wav")
            dummy_wav_path2 = os.path.join(input_audio_dir, "dummy_speakerB.wav")
            
            sr = 16000; dur = 3; freq1 = 300; freq2 = 500
            t = np.linspace(0, dur, int(sr*dur), endpoint=False)
            amp = np.iinfo(np.int16).max // 4
            
            # Speaker A
            note1_a = amp * np.sin(2*np.pi*freq1*t)
            note2_a = amp * np.sin(2*np.pi*(freq1*1.2)*t) # Slightly different freq
            signal_a = (note1_a * (t < dur/2)) + (note2_a * (t >= dur/2)) # Speaker A speaks twice
            write_wav(dummy_wav_path1, sr, signal_a.astype(np.int16))

            # Speaker B (different characteristics or silence for simplicity now)
            # For a better diarization test, you'd mix these or have distinct segments.
            # Here, we'll just make two separate files. Diarization on single speaker files is trivial.
            # A better test would be to concatenate these into one file.
            note1_b = amp * np.sin(2*np.pi*freq2*t)
            write_wav(dummy_wav_path2, sr, note1_b.astype(np.int16))
            
            # Convert dummy WAVs to MP3s (requires your convert_wav_to_mp3_voice function)
            # For this example, I'll assume you have a function `convert_wav_to_mp3_voice`
            # If not, you'll need to provide your own MP3s.
            # This part is illustrative and depends on your project structure.
            print("Dummy WAVs created. You would convert these to MP3s using a previous function.")
            print(f"Please manually convert '{dummy_wav_path1}' and '{dummy_wav_path2}' to MP3")
            print(f"and place them in '{input_audio_dir}' or use your own MP3s.")
            print("For a proper test, create an MP3 with multiple speakers.")

            # Illustrative: If you have the converter function in the same context:
            # from your_previous_module import convert_wav_to_mp3_voice
            # if 'convert_wav_to_mp3_voice' in globals():
            #    convert_wav_to_mp3_voice(dummy_wav_path1, os.path.join(input_audio_dir, "dummy_speakerA.mp3"))
            #    convert_wav_to_mp3_voice(dummy_wav_path2, os.path.join(input_audio_dir, "dummy_speakerB.mp3"))
            # else:
            #    print("`convert_wav_to_mp3_voice` function not found. Skipping dummy MP3 creation.")
            
        except ImportError:
            print("Could not import scipy.io.wavfile or numpy. Skipping dummy WAV/MP3 creation.")
        except Exception as e:
            print(f"Error creating dummy files: {e}")
    
    # Check if input directory exists and has MP3s before proceeding
    if os.path.exists(input_audio_dir) and any(f.lower().endswith(".mp3") for f in os.listdir(input_audio_dir)):
         diarize_mp3_folder(input_audio_dir, output_results_dir, hf_token=HUGGING_FACE_TOKEN)
    else:
        print(f"\nInput directory '{input_audio_dir}' is missing or contains no MP3 files.")
        print("Please create it, add some MP3 files, and re-run the script.")
        print("The dummy file creation part above is illustrative; ensure valid MP3s are present for testing.")