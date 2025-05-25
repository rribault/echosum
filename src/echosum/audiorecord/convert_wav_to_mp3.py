from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
from pydub.exceptions import CouldntEncodeError, PydubException
import os

def convert_wav_to_mp3_voice(
    input_wav_path: str,
    output_mp3_path: str = None,
    bitrate: str = "96k",
    to_mono: bool = True,
    apply_normalize: bool = True,
    normalize_headroom_db: float = 0.1,
    apply_high_pass: bool = True,
    high_pass_cutoff_hz: int = 80
) -> str | None:
    """
    Converts a WAV file to MP3, with optimizations for voice recordings.

    Args:
        input_wav_path (str): Path to the input WAV file.
        output_mp3_path (str, optional): Path to save the output MP3 file.
            If None, defaults to the same name as input_wav_path but with .mp3
            extension, in the same directory.
        bitrate (str): Bitrate for MP3 encoding (e.g., "64k", "96k", "128k").
                       "96k" is a good balance for mono voice.
        to_mono (bool): If True and the source is stereo, convert audio to mono.
        apply_normalize (bool): If True, normalize the audio peak volume.
        normalize_headroom_db (float): Headroom in dB for peak normalization
                                       (e.g., 0.1 means peak at -0.1 dBFS).
                                       Only used if apply_normalize is True.
        apply_high_pass (bool): If True, apply a high-pass filter to remove
                                low-frequency rumble.
        high_pass_cutoff_hz (int): Cutoff frequency for the high-pass filter in Hz.
                                   Only used if apply_high_pass is True.

    Returns:
        str: The path to the generated MP3 file, or None if conversion failed.
    """

    if not os.path.exists(input_wav_path):
        print(f"Error: Input WAV file not found at '{input_wav_path}'")
        return None

    if not input_wav_path.lower().endswith(".wav"):
        print(f"Error: Input file '{input_wav_path}' does not appear to be a WAV file.")
        return None

    # Determine output path
    if output_mp3_path is None:
        base, _ = os.path.splitext(input_wav_path)
        final_output_mp3_path = base + ".mp3"
    else:
        final_output_mp3_path = output_mp3_path
        if not final_output_mp3_path.lower().endswith(".mp3"):
            base, _ = os.path.splitext(final_output_mp3_path)
            final_output_mp3_path = base + ".mp3"
            print(f"Warning: Output path did not end with .mp3, changed to '{final_output_mp3_path}'")


    # Ensure output directory exists
    output_dir = os.path.dirname(final_output_mp3_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return None
    elif not output_dir and final_output_mp3_path == os.path.basename(final_output_mp3_path):
        # This means output is in the current working directory.
        pass


    print(f"Processing '{input_wav_path}'...")

    try:
        # 1. Load WAV audio
        audio = AudioSegment.from_wav(input_wav_path)
        print(f"  Loaded WAV: Duration={audio.duration_seconds:.2f}s, Channels={audio.channels}, FrameRate={audio.frame_rate}Hz")

        # 2. Convert to Mono (if specified and necessary)
        if to_mono and audio.channels > 1:
            audio = audio.set_channels(1)
            print(f"  Converted to mono.")
        elif to_mono and audio.channels == 1:
            print(f"  Audio is already mono.")


        # 3. Apply High-pass Filter (if specified)
        if apply_high_pass and high_pass_cutoff_hz > 0:
            audio = audio.high_pass_filter(cutoff=high_pass_cutoff_hz)
            print(f"  Applied high-pass filter at {high_pass_cutoff_hz}Hz.")

        # 4. Normalize Audio (if specified)
        if apply_normalize:
            audio = pydub_normalize(audio, headroom=normalize_headroom_db)
            print(f"  Normalized audio with {normalize_headroom_db}dB headroom.")

        # 5. Export to MP3
        print(f"  Exporting to MP3 with bitrate {bitrate}...")
        # Ensure FFmpeg parameters are correctly passed for mono if set_channels was used
        # For pydub, setting channels on the segment is usually enough.
        # However, sometimes explicitly setting '-ac 1' in parameters is more robust with ffmpeg.
        export_parameters = []
        if audio.channels == 1: # Check after potential set_channels
             export_parameters = ["-ac", "1"]

        audio.export(
            final_output_mp3_path,
            format="mp3",
            bitrate=bitrate,
            parameters=export_parameters # e.g. ["-ac", "1"] for mono
        )
        print(f"Successfully converted and saved to '{final_output_mp3_path}'")
        return final_output_mp3_path

    except FileNotFoundError: # Should be caught by the initial check, but good practice with from_wav
        print(f"Error: Input WAV file not found (during pydub processing) at '{input_wav_path}'")
        return None
    except PydubException as e: # Catches issues like not a valid WAV
        print(f"Error processing WAV file '{input_wav_path}': {e}")
        return None
    except CouldntEncodeError:
        print("\n--- MP3 Encoding Error ---")
        print("Error: Failed to encode to MP3. This usually means FFmpeg is not installed or not found in your system's PATH.")
        print("Please ensure FFmpeg is correctly installed and accessible.")
        print("Installation help:")
        print("  - Windows: Download from ffmpeg.org, add its 'bin' folder to PATH.")
        print("  - Linux: `sudo apt install ffmpeg` or `sudo yum install ffmpeg`.")
        print("  - macOS: `brew install ffmpeg`.")
        print("--------------------------\n")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")
        return None

if __name__ == "__main__":
    # --- Example Usage ---

    # Create a dummy WAV file for testing (if you don't have one readily available)
    # This requires numpy and scipy for the dummy WAV creation part.
    # If you ran the previous microphone recording script, you'll have WAV files.
    dummy_wav_dir = "dummy_wavs_for_conversion"
    os.makedirs(dummy_wav_dir, exist_ok=True)
    dummy_wav_path = os.path.join(dummy_wav_dir, "test_voice.wav")

    try:
        import numpy as np
        from scipy.io.wavfile import write as write_wav

        # Create a simple 5-second stereo tone if the test file doesn't exist
        if not os.path.exists(dummy_wav_path):
            print(f"Creating a dummy WAV file for testing: {dummy_wav_path}")
            sample_rate = 44100
            duration = 5  # seconds
            frequency1 = 440  # A4 note
            frequency2 = 660  # E5 note
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            # Create a stereo signal:
            # Left channel: sine wave
            # Right channel: sine wave + a lower amplitude sine wave
            amplitude = np.iinfo(np.int16).max // 4 # Use a quarter of max amplitude
            audio_l = amplitude * np.sin(2 * np.pi * frequency1 * t)
            audio_r = amplitude * np.sin(2 * np.pi * frequency2 * t) * 0.8
            stereo_signal = np.vstack((audio_l, audio_r)).T.astype(np.int16)
            write_wav(dummy_wav_path, sample_rate, stereo_signal)
            print(f"Dummy stereo WAV file '{dummy_wav_path}' created.")
        else:
            print(f"Using existing dummy WAV file: '{dummy_wav_path}'")

    except ImportError:
        print("Note: Skipping dummy WAV creation as numpy or scipy is not installed.")
        print("Please provide your own WAV file for testing if 'test_voice.wav' doesn't exist.")
    except Exception as e:
        print(f"Error creating dummy WAV: {e}")


    # --- Test Cases ---
    if os.path.exists(dummy_wav_path):
        print("\n--- Test Case 1: Default settings (mono, 96k, normalize, high-pass) ---")
        output_mp3_1 = os.path.join(dummy_wav_dir,"test_voice_default.mp3")
        convert_wav_to_mp3_voice(dummy_wav_path, output_mp3_1)

        print("\n--- Test Case 2: Different bitrate, no normalization, no high-pass ---")
        output_mp3_2 = os.path.join(dummy_wav_dir,"test_voice_custom.mp3")
        convert_wav_to_mp3_voice(
            dummy_wav_path,
            output_mp3_2,
            bitrate="128k",
            apply_normalize=False,
            apply_high_pass=False
        )

        print("\n--- Test Case 3: Output path not specified (saves next to input) ---")
        # Make a copy to avoid overwriting the source if it was already an mp3 somehow
        # (though the function checks for .wav input)
        input_for_test3 = os.path.join(dummy_wav_dir, "another_test_voice.wav")
        if not os.path.exists(input_for_test3) and os.path.exists(dummy_wav_path):
             import shutil
             shutil.copy(dummy_wav_path, input_for_test3)
        if os.path.exists(input_for_test3):
            convert_wav_to_mp3_voice(input_for_test3) # Output will be another_test_voice.mp3
        else:
            print(f"Skipping Test Case 3 as '{input_for_test3}' could not be prepared.")

        print("\n--- Test Case 4: Non-existent input file ---")
        convert_wav_to_mp3_voice("non_existent_file.wav", "output.mp3")

        print("\n--- Test Case 5: Input is not a WAV ---")
        not_a_wav_file = os.path.join(dummy_wav_dir,"test.txt")
        with open(not_a_wav_file, "w") as f:
            f.write("This is not a wav file.")
        convert_wav_to_mp3_voice(not_a_wav_file, "output_txt.mp3")


    else:
        print(f"\nCannot run example usage: Dummy WAV file '{dummy_wav_path}' not found and could not be created.")
        print("Please ensure you have a WAV file to test with, or install numpy and scipy for dummy file creation.")

    # Example for processing a folder (using the previous recording function's output)
    # from your previous request's output folder
    # input_audio_folder = "audio_recordings" # Or "my_audio_sessions/recording_session_YYYYMMDD_HHMMSS"
    # output_mp3_folder = "mp3_converted_recordings"

    # if os.path.exists(input_audio_folder):
    #     os.makedirs(output_mp3_folder, exist_ok=True)
    #     print(f"\n--- Processing all WAVs in '{input_audio_folder}' ---")
    #     for filename in os.listdir(input_audio_folder):
    #         if filename.lower().endswith(".wav"):
    #             wav_file_path = os.path.join(input_audio_folder, filename)
    #             mp3_file_name = os.path.splitext(filename)[0] + ".mp3"
    #             mp3_file_path = os.path.join(output_mp3_folder, mp3_file_name)
    #             convert_wav_to_mp3_voice(wav_file_path, mp3_file_path)
    # else:
    #     print(f"\nSkipping folder processing example: Folder '{input_audio_folder}' not found.")