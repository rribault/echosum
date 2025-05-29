from openai import OpenAI
client = OpenAI()


def transcribe_audio(path:str) ->str :

    #.mp3 file
    audio_file = open(path, "rb")

    transcription_text = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe", 
        file=audio_file, 
        response_format="text"
    )

    return transcription_text