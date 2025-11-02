# import statements:
import io
import os
import time
import traceback
from dataclasses import dataclass, field
import groq

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import spaces
import xxhash
from datasets import Audio

# import groq api key & create client
api_key = os.environ.get("GROQ_API_KEY")
client = groq.Client(api_key=api_key)

@dataclass # automatically adds boilerplate methods to class (e.g. init, etc.)
class AppState:
  conversation: list = field(default_factory=list) # stores chat history
  stopped: bool = False # indicates whether recording/preprocessing has stopped
  model_outs: any = None # flexible placeholder for model outputs

# transcription function
def transcribe_audio(client, file_name):
  if file_name is None:
    return None
  try: 
    with open(file_name, "rb") as audio_file:
      response = client.audio.transcriptions.with_raw_response.create(
        model="whisper-large-v3-turbo",
        file=("audio.wav", audio_file),
        response_format="verbose_json"
      )
      completion = process_whisper_response(response.parse())
      return completion
  except Exception as e:
    print(f"Error in transcription{e}")
    return f"Error in transcription: {str(e)}"





if __name__ == "__main__":
    demo.launch()