import requests
import gradio as gr

import whisper
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from nltk import tokenize
from tokenizers import Tokenizer

tde_theme = gr.themes.Default().set(
    body_background_fill='#040617',
    body_background_fill_dark="linear-gradient(to top right, #4f0829, #040617)",
    button_large_radius='*radius_xxl',
    button_secondary_background_fill='#f5f5f5',
    button_secondary_background_fill_dark='#f5f5f5',
    button_secondary_background_fill_hover='#dcddde',
    button_secondary_background_fill_hover_dark='#dcddde',
    button_secondary_text_color='#040617',
    button_secondary_text_color_dark='#040617',
)

def downloadPodcastFromUrl(url):
    url = url.replace(' ', '%20')
    file = requests.get(url)
    with open('podcast.mp3', 'wb') as f:
        f.write(file.content)


def transcribe(x, inputLang, isUrl):
    if isUrl:
        downloadPodcastFromUrl(x)
        x = './podcast.mp3'

    if inputLang == 'English':
        model = whisper.load_model('small.en')
        result = model.transcribe(x)
        n = 1024
    if inputLang == 'Dutch':
        model = whisper.load_model('small')
        result = model.transcribe(x)
        n = 900
    else:
        model = whisper.load_model('small')
        result = model.transcribe(x, task='Translate')
        n = 900

    #split transcribed text into sentences
    sentences = tokenize.sent_tokenize(text=result['text'], language='English')
    #make full-sentence strings of n tokens long for summarization
    coll = createChunks(sentences, n)
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = Tokenizer.from_pretrained("facebook/bart-large-cnn")
    summary = []

    try:
        #summarize each chunk individually and add them together
        for val in coll:
            numTokens = len(tokenizer.encode(val))
            maxLength = min(500, numTokens)
            minLength = min(100, numTokens)
            summary.append(summarizer(val, max_length=maxLength, min_length=minLength, do_sample=False))
    except:
        coll = createChunks(sentences, 700)
        for val in coll:
            numTokens = len(tokenizer.encode(val))
            maxLength = min(500, numTokens)
            minLength = min(100, numTokens)
            summary.append(summarizer(val, max_length=maxLength, min_length=minLength, do_sample=False))

    result = str(summary)
    result = result.replace("[[{'summary_text': '", '').replace("[[{'summary_text': ", '').replace("'}], ", ' ').replace("}], ", ' ').replace("[{'summary_text': '", '').replace("[{'summary_text': ", '').replace("'}]]", '').replace("}]]", '')

    return result 

    #optional translation step
    if inputLang == outputLang:
        return result
    else:
        return translate(result)
    

#takes a collection of sentences and divides them into strings of a maximum length of tokenCount in tokens    
def createChunks(sentences, tokenCount):
    coll = []
    chunk = ""
    wordCount = 0
    tokenizer = Tokenizer.from_pretrained("facebook/bart-large-cnn")

    #create chunks of sentences to be summarized, each chunk not exceeding n tokens
        #needs an exception in case a single sentence exceeds n (however unlikely)
    for sent in sentences:
        wordCount += len(tokenizer.encode(sent))
        if wordCount <= tokenCount:
            chunk += sent
        else:
            coll.append(chunk)
            chunk = sent
            wordCount = 0
    coll.append(chunk)

    return coll


#works, but translations are very low quality and can only translate to english
def translate(string):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
    return translator(string)


#translation using the t5 model - only supports English, French, German, and Spanish
def t5translate(string, inputLanguage, outputLanguage):
    t5Tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    commandString = "translate " + inputLanguage + " to " + outputLanguage + ": " + string

    input_ids = t5Tokenizer(commandString, max_length=1024, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)

    return str(t5Tokenizer.decode(outputs[0], skip_special_tokens=True))


#creates the Gradio interface
with gr.Blocks(theme=tde_theme) as demo:
    with gr.Tab('Local File'):
        file_input = gr.Audio(type='filepath')
        file_radio = gr.Radio(['English', 'Dutch', 'Spanish', 'Italian', 'Portugese', 'German', 'Other'], value='English', label='Input Language')
        #the buttonclick event doesn't accept a value as argument for its input for some reason, so it happens through a hidden radio button
        file_bool = gr.Radio(value=False, visible=False)
        #radio2 = gr.Radio(['English', 'Dutch'], value='English', label='Output Language')
        file_button = gr.Button('Summarize')
        file_output = gr.Textbox(label='Summary')
    with gr.Tab('URL'):
        url_input = gr.Textbox(label='Enter URL:')
        url_radio = gr.Radio(['English', 'Dutch', 'Spanish', 'Italian', 'Portugese', 'German', 'Other'], value='English', label='Input Language')
        url_bool = gr.Radio(value=True, visible=False)
        url_button = gr.Button('Summarize')
        url_output = gr.Textbox(label='Summary')

    file_button.click(transcribe, inputs=[file_input, file_radio, file_bool], outputs=file_output)
    url_button.click(transcribe, inputs=[url_input, url_radio, url_bool], outputs=url_output)


if __name__ == "__main__":
    demo.launch()  