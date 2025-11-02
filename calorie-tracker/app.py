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

def process_audio(audio: tuple, state: AppState):
    return audio, state

def start_recording_user(state: AppState):
    return None

# transcription function - takes audio file & sends to Groq's Whisper API
def transcribe_audio(client, file_name):
  if file_name is None:
    return None
  try: 
    with open(file_name, "rb") as audio_file: # opens audio file in binary readmode (closes automatically)
      response = client.audio.transcriptions.with_raw_response.create( # sends the audio file to whisper
        model="whisper-large-v3-turbo",
        file=("audio.wav", audio_file), #1st part is filename label, 2nd is binary file obj
        response_format="verbose_json" # detailed output
      )
      completion = process_whisper_response(response.parse()) # turns the raw respnse into Python obj (internally Pydantic model)
      return completion # final transcription text
  except Exception as e:
    print(f"Error in transcription{e}")
    return f"Error in transcription: {str(e)}"

# decides whether user said something meaningful
def process_whisper_response(completion):
  if completion.segments and len(completion.segments) > 0: # speech detection check
      no_speech_prob = completion.segments[0].get('no_speech_prob', 0) # probability of silence
      print("No speech prob:", no_speech_prob)
      if no_speech_prob > 0.7: # if chance is over 70%, it is probably silence
        return None
      return completion.text.strip() # return cleaned text
  return None


# conversational intelligence -> called whenver the user stops speaking
def generate_chat_completion(client, history):
  messages = []
  messages.append(
    {
      "role": "system", # acts as model's engagment rules & explains expected response
      "content": "In conversation with the user, ask questions to estimate and provide (1) total calories, (2) protein, carbs, and fat in grams, (3) fiber and sugar content. Only ask *one question at a time*. Be conversational and natural."
    }
  )
  for message in history: # loops through each message in convo & adds to list sent to API
    messages.append(message)
  try: 
    completion = client.chat.completions.create( # provide it the message list
      model="llama-3.1-8b-instant",
      messages=messages,
    )
    return completion.choices[0].message.content # returns the model's message text
  except Exception as e:
    return f"Error in generating chat completion: {str(e)}"
  
def response(state: AppState, audio: tuple): # audio = recording mic input from gradio
  if not audio: # check for user provided audio
    return AppState()
  
  file_name = f"/tmp/{xxhash.xxh32(bytes(audio[1])).hexdigest()}.wav" # unique filename
  sf.write(file_name, audio[1], audio[0], format="wav") # save audio data to wave file

  api_key = os.environ.get("GROQ_API_KEY")
  client = groq.Client(api_key=api_key) # create Groq client

  # transcribe audio file
  transcription = transcribe_audio(client, file_name) # call helper function by sending wav file
  if transcription:
    state.conversation.append({"role": "user", "content": transcription}) # adds spoken text to chat history -> assistant has full context

    assistant_message = generate_chat_completion(client, state.conversation) # returns the mode's next reply
    state.conversation.append({"role": "assistant", "content": assistant_message}) # append assistant's
    print(state.conversation) # debugging

    os.remove(file_name)
  return state,  state.conversation

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)
    

# VAD detection function -> storing a block of js in python string
# the link allows ML models to run in the browser
js = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1)
  const script2 = document.createElement("script");
  script2.onload = async () =>  {
    console.log("vad loaded") ;
    var record = document.querySelector('.record-button');
    record.textContent = "Just Start Talking!"
    record.style = "width: fit-content; padding-right: 0.5vw;"
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        var record = document.querySelector('.record-button');
        var player = document.querySelector('#streaming-out')
        if (record != null && (player == null || player.paused)) {
          console.log(record);
          record.click();
        }
      },
      onSpeechEnd: (audio) => {
        var stop = document.querySelector('.stop-button');
        if (stop != null) {
          console.log(stop);
          stop.click();
        }
      }
    })
    myvad.start()
  }
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
  script1.onload = () =>  {
    console.log("onnx loaded") 
    document.head.appendChild(script2)
  };
}
"""

js_reset = """
() => {
  var record = document.querySelector('.record-button');
  record.textContent = "Just Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""

# building UI w/ Gradio
with gr.Blocks(theme=theme, js=js) as demo: # Gradio's layout API, inserts JS for VAD detection
  with gr.Row():
    input_audio= gr.Audio(
      label="Input Audio",
      sources=["microphone"],
      type="numpy",
      streaming=False,
      waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
    )
  with gr.Row():
    chatbot = gr.Chatbot(label="Conversation", type="messages")
  state = gr.State(value=AppState())
  stream = input_audio.start_recording(
    process_audio,
    [input_audio, state],
    [input_audio, state]
  )
  respond = input_audio.stop_recording(
    response, [state, input_audio], [state, chatbot]
  )
  restart = respond.then(start_recording_user, [state], [input_audio]).then(
      lambda state: state, state, state, js=js_reset
  )

  cancel = gr.Button("New Conversation", variant="stop")
  cancel.click(
      lambda: (AppState(), gr.Audio(recording=False)),
      None,
      [state, input_audio],
      cancels=[respond, restart],
  )


if __name__ == "__main__":
    demo.launch()