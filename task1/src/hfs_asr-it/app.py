"""
Imports
"""
from transformers import pipeline
from pytube import YouTube
import gradio as gr
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

"""
Pipeline and models
"""
transcribe_pipe = pipeline(model="Silemo/whisper-it")  # change to "your-username/the-name-you-picked"

tags_model = AutoModelForSeq2SeqLM.from_pretrained("efederici/text2tags")
tags_tokenizer = AutoTokenizer.from_pretrained("efederici/text2tags")

"""
Methods
"""
def transcribe(audio):
    text = transcribe_pipe(audio)["text"]
    return text

def transcribe_video(url):
    yt = YouTube(url)
    stream = yt.streams.get_audio_only()

    # Saves the audio in the /audio folder
    audio = stream.download() 

    text = transcribe_and_tag(audio)

    return text

def transcribe_and_tag(audio):
    text = transcribe(audio)
    tags = tag(text=text)
    return text, tags

def download_audio(audio_url, filename):

    # URL of the image to be downloaded is defined as audio_url
    r = requests.get(audio_url) # create HTTP response object 
  
    # send a HTTP request to the server and save 
    # the HTTP response in a response object called r 
    with open(filename,'wb') as f: #"audio/" + 
  
        # Saving received content as a mp3 file in 
        # binary format 
  
        # write the contents of the response (r.content) 
        # to a new file in binary mode. 
        f.write(r.content) 

def tag(text: str):
    """ 
    Generates tags from given text 
    """
    text = text.strip().replace('\n', '')
    text = 'summarize: ' + text
    tokenized_text = tags_tokenizer.encode(text, return_tensors="pt")

    tags_ids = tags_model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        max_length=20,
                                        early_stopping=True)

    output = tags_tokenizer.decode(tags_ids[0], skip_special_tokens=True)
    return output.split(', ')

"""
Downloading audio files
"""
audio1_url = "https://github.com/Silemo/sml-lab2-2023-manfredi-meneghin/raw/main/task1/audio/offer.mp3"
audio1_filename = "offer.mp3"
download_audio(audio1_url, audio1_filename)

audio2_url = "https://github.com/Silemo/sml-lab2-2023-manfredi-meneghin/raw/main/task1/audio/fantozzi.mp3"
audio2_filename = "fantozzi.mp3"
download_audio(audio2_url, audio2_filename)

"""
Interfaces
"""
audio_transcription = gr.Textbox(label="Transcription")
audio_tags = gr.Textbox(label="Tags")

yt_transcription = gr.Textbox(label="Transcription")
yt_tags = gr.Textbox(label="Tags")

# Multiple interfaces using tabs -> https://github.com/gradio-app/gradio/issues/450
io1 = gr.Interface(
    fn = transcribe_and_tag,
    inputs = gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs = [audio_transcription, audio_tags],
    examples = [
        [audio1_filename],
        [audio2_filename],
    ],
    title = "Whisper Small - Italian - Microphone or Audio file",
    description = "Realtime demo for Italian speech recognition using a fine-tuned Whisper small model. It uses the computer microphone as audio input. It outputs a transcription and the tags of the text.",
)

io2 = gr.Interface(
    fn = transcribe_video,
    inputs = gr.Textbox(label = "YouTube URL", placeholder = "https://youtu.be/9DImRZERJNs?si=1Lme7o_KH2oCxU7y"),
    outputs=[yt_transcription, yt_tags],

    examples=[
        # Meloni - Confindustria
        ["https://www.youtube.com/watch?v=qMslwA7RCcc"],
        
        # Montemagno - Ripartire da zero
        ["https://www.youtube.com/watch?v=WlT3dCAGjRo"],
    ],
    
    title = "Whisper Small - Italian - YouTube link",
    description = "Realtime demo for Italian speech recognition using a fine-tuned Whisper small model. It uses a YouTube link as audio input. It outputs a transcription and the tags of the text.",
)

gr.TabbedInterface(
    [io1, io2], {"Microphone or audio file", "YouTube"}
).launch()