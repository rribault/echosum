import sounddevice as sd
import soundfile as sf
import os
from datetime import datetime

# --- Configuration ---
DEFAULT_OUTPUT_FOLDER = "audio_recordings"
DEFAULT_CHUNK_DURATION_SECONDS = 30  # Duration of each audio chunk in seconds
DEFAULT_SAMPLE_RATE_HZ = 44100       # Sample rate in Hz
DEFAULT_CHANNELS = 1                 # Number of audio channels (1 for mono, 2 for stereo)
DEFAULT_FILENAME_PREFIX = "rec_"     # Prefix for a_recorded_chunk_nameles
DEFAULT_SUBTYPE = 'PCM_16'           # Subtype for WAV file (PCM_16 means 16-bit resolution)
                                     # For this subtype, dtype in sd.rec should be 'int16'

def check_audio_devices():
    """Checks for available input devices and prints info about the default one."""
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]

        if not input_devices:
            print("Error: No microphone/input audio device found.")
            print("Please ensure a microphone is connected and configured.")
            if not devices:
                print("  Additionally, no audio devices (input or output) were detected at all.")
            else:
                print("Available devices found:")
                for i, device in enumerate(devices):
                    print(f"  Device {i}: {device['name']} (Inputs: {device['max_input_channels']}, Outputs: {device['max_output_channels']})")
            return False

        # Try to get and print default input device info
        # sd.default.device[0] is the input device index
        if sd.default.device[0] == -1 and len(input_devices) > 0:
            # If no explicit default, sounddevice might pick the first available one
            print(f"Warning: No explicit default input device set, but {len(input_devices)} input device(s) available.")
            print(f"Sounddevice will attempt to use one. Often this is: {input_devices[0]['name']}")
        elif sd.default.device[0] != -1:
            default_input_device_info = sd.query_devices(sd.default.device[0])
            print(f"Using input device: {default_input_device_info['name']} (Sample Rate: {default_input_device_info['default_samplerate']:.0f} Hz)")
        else: # Should not happen if input_devices is not empty
             print("Could not determine default input device, but input devices are present.")
             
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        print("Attempting to proceed, but recording may fail if no suitable device is found.")
        return True # Allow to proceed, sd.rec will fail later if no device
    return True


def record_audio_chunks(
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    chunk_duration_seconds: int = DEFAULT_CHUNK_DURATION_SECONDS,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    channels: int = DEFAULT_CHANNELS,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    subtype: str = DEFAULT_SUBTYPE
) -> None:
    """
    Records audio from the microphone in chunks and saves them to a specified folder.

    Args:
        output_folder (str): The folder where audio chunks will be saved.
                             Created if it doesn't exist.
        chunk_duration_seconds (int): Duration of each audio chunk in seconds.
        sample_rate_hz (int): The sample rate for the recording in Hz.
        channels (int): The number of audio channels (e.g., 1 for mono, 2 for stereo).
        filename_prefix (str): Prefix for the names of the recorded audio files.
        subtype (str): Subtype for the WAV file (e.g., 'PCM_16', 'FLOAT').
                       Determines bit depth and data type.
    """

    if not check_audio_devices():
        return

    # Determine dtype based on subtype for sounddevice recording
    # Common WAV subtypes: PCM_S8, PCM_16, PCM_24, PCM_32, FLOAT, DOUBLE
    if subtype in ['PCM_S8', 'PCM_16', 'PCM_24', 'PCM_32']:
        # For PCM types, sounddevice works well with corresponding integer dtypes
        # For example, 'PCM_16' -> 'int16'
        # 'PCM_S8' isn't directly supported by common dtypes, 'int8' might map if 'PCM_U8' is the target
        # but soundfile handles subtypes well, so we can often use float32 from sounddevice
        # and let soundfile convert. However, to be explicit:
        if subtype == 'PCM_16':
            record_dtype = 'int16'
        elif subtype == 'PCM_24': # sounddevice might not directly support 'int24'
                                  # record in 'int32' and let soundfile handle subtype='PCM_24'
            record_dtype = 'int32' # soundfile can write int32 data as PCM_24
        elif subtype == 'PCM_32':
            record_dtype = 'int32'
        else: # Default to float32 and let soundfile manage conversion
            print(f"Warning: For subtype {subtype}, recording with 'float32' and letting soundfile handle conversion.")
            record_dtype = 'float32'
    elif subtype in ['FLOAT', 'DOUBLE']:
        record_dtype = 'float32' if subtype == 'FLOAT' else 'float64'
    else:
        print(f"Warning: Unknown subtype '{subtype}'. Defaulting recording dtype to 'float32'.")
        record_dtype = 'float32'


    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Audio recordings will be saved to: {os.path.abspath(output_folder)}")
    except OSError as e:
        print(f"Error creating output directory {output_folder}: {e}")
        return

    print(f"Starting recording with chunks of {chunk_duration_seconds}s.")
    print(f"Parameters: SR={sample_rate_hz}Hz, Channels={channels}, Format={subtype} (rec_dtype={record_dtype})")
    print("Press Ctrl+C to stop recording.")

    chunk_count = 0
    try:
        while True:
            chunk_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] # YYYYMMDD_HHMMSS_mmm
            filename = f"{filename_prefix}{timestamp}.wav"
            file_path = os.path.join(output_folder, filename)

            print(f"\nRecording chunk {chunk_count}: {filename} for {chunk_duration_seconds}s...")

            # Calculate the number of frames
            num_frames = int(chunk_duration_seconds * sample_rate_hz)

            # Record audio
            # Using blocking=True makes sd.rec wait until recording is finished
            try:
                recording = sd.rec(
                    frames=num_frames,
                    samplerate=sample_rate_hz,
                    channels=channels,
                    dtype=record_dtype, # e.g., 'int16', 'float32'
                    blocking=True
                )
                # sd.wait() # Not needed if blocking=True
            except sd.PortAudioError as e:
                print(f"PortAudio error during recording: {e}")
                print("This might be due to issues with the audio device or configuration.")
                print("Please check your microphone and audio settings.")
                break # Exit the loop on PortAudio error

            except KeyboardInterrupt:
                print("\nRecording stopped by user.")
                sf.write(file_path, recording, sample_rate_hz, subtype=subtype)
                print(f"Recording chunk stoped by user -  {chunk_count}. Saving to {file_path}...")

            except Exception as e:
                print(f"An unexpected error occurred during recording: {e}")
                break


            print(f"Finished recording chunk {chunk_count}. Saving to {file_path}...")

            try:
                # Save the recorded audio to a WAV file using soundfile
                # soundfile will handle the subtype correctly based on data or explicit subtype
                sf.write(file_path, recording, sample_rate_hz, subtype=subtype)
                print(f"Chunk {chunk_count} saved successfully.")
            except Exception as e:
                print(f"Error saving file {file_path}: {e}")
                # Decide if you want to stop or continue on save error
                # For now, we continue to the next chunk

            # Optional: small delay if you want a pause between chunks,
            # but blocking recording already paces the loop.
            # time.sleep(0.1)


    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        saved_chunks = chunk_count -1 if chunk_count > 0 else 0 # if interrupted during first record
        # Check if the last attempted chunk was actually saved before decrementing
        # For simplicity, this counts attempts.
        print(f"\nRecording process terminated. {saved_chunks} full chunk(s) attempted to be saved.")
        print(f"Files are in: {os.path.abspath(output_folder)}")


if __name__ == "__main__":
    # --- Example Usage ---
    print("Microphone recording script started.")
    
    # You can customize parameters here:
    custom_output_folder = "my_audio_sessions"
    custom_chunk_duration = 60*5  # seconds
    custom_sample_rate = 48000  # Hz
    custom_channels = 1         # Mono
    custom_filename_prefix = "sessionX_chunk_"
    custom_subtype = 'PCM_16'   # 16-bit WAV

    # Create a specific subfolder for this session based on current time
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = os.path.join(custom_output_folder, f"recording_session_{session_timestamp}")

    # record_audio_chunks(output_folder=session_folder, chunk_duration_seconds=custom_chunk_duration)
    
    # Or run with defaults:
    record_audio_chunks()

    # Example with stereo and 32-bit float WAV:
    # print("\nStarting a new recording session: Stereo, 32-bit Float, 15s chunks")
    # record_audio_chunks(
    #     output_folder="stereo_float_recordings",
    #     chunk_duration_seconds=15,
    #     sample_rate_hz=44100,
    #     channels=2,
    #     filename_prefix="stereo_rec_",
    #     subtype='FLOAT' # Results in a 32-bit float WAV file
    # )