from transformers import pipeline
from pytube import YouTube
import gradio as gr
import requests

pipe = pipeline(model="Silemo/whisper-it")  # change to "your-username/the-name-you-picked"

def download_audio(audio_url, filename):

    # URL of the image to be downloaded is defined as audio_url
    r = requests.get(audio_url) # create HTTP response object 
  
    # send a HTTP request to the server and save 
    # the HTTP response in a response object called r 
    with open(["audio/" + filename],'wb') as f: 
  
        # Saving received content as a mp3 file in 
        # binary format 
  
        # write the contents of the response (r.content) 
        # to a new file in binary mode. 
        f.write(r.content) 

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

def transcribe_video(url):
    yt = YouTube(url)
    stream = yt.streams.get_audio_only()

    # Saves the audio in the /audio folder
    audio = stream.download(output_path = "audio/")

    text = transcribe(audio)

    return text

audio1_url = "https://github.com/Silemo/sml-lab2-2023-manfredi-meneghin/raw/main/task1/audio/offer.mp3"
audio1_filename = "offer.mp3"
download_audio(audio1_url, audio1_filename)

audio2_url = "https://github.com/Silemo/sml-lab2-2023-manfredi-meneghin/raw/main/task1/audio/fantozzi.mp3"
audio2_filename = "fantozzi.mp3"
download_audio(audio2_url, audio2_filename)

# Multiple interfaces using tabs -> https://github.com/gradio-app/gradio/issues/450

io1 = gr.Interface(
    fn = transcribe,
    inputs = gr.Audio(source=["microphone", "upload"], type="filepath"),
    outputs = "text",

    examples=[
        ["audio/" + audio1_filename],
        ["audio/" + audio2_filename],
    ],

    title = "Whisper Small - Italian - Microphone or Audio file",
    description = "Realtime demo for Italian speech recognition using a fine-tuned Whisper small model. It uses the computer microphone as audio input",
)

io2 = gr.Interface(
    fn = transcribe_video,
    inputs = gr.Textbox(label = "YouTube URL", placeholder = "https://youtu.be/9DImRZERJNs?si=1Lme7o_KH2oCxU7y"),
    outputs = "text",

    examples=[
        # Per me è la cipolla
        ["https://youtu.be/QbwZlURClSA?si=DKMtIiKE-nO2mfcV"],
        
        # Breaking Italy - Lollobrigida ferma il treno
        ["https://youtu.be/9MPBN0tnA_E?si=G9Sgn1AsXSkxfCxV"],
    ],
    
    title = "Whisper Small - Italian - YouTube link",
    description = "Realtime demo for Italian speech recognition using a fine-tuned Whisper small model. It uses a YouTube link as audio input",
)

gr.TabbedInterface(
    [io1, io2], {"Microphone or audio file", "YouTube"}
).launch()