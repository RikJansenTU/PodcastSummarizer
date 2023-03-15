import gradio as gr
import whisper
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from nltk import tokenize
from tokenizers import Tokenizer








def transcribe(x, inputLang, outputLang):
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
    #make strings of n tokens long for summarization
    coll = createChunks(sentences, n)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    summary = []

 #   try:
        #summarize each chunk individually and add them together
    for val in coll:
        numTokens = len(tokenizer.encode(val))
        maxLength = min(500, numTokens)
        minLength = min(100, numTokens)
        tokenized_text = tokenizer("summarize: " + val, return_tensors="pt").input_ids
        outputs = model.generate(tokenized_text, min_length=minLength, max_length=maxLength)
        summary.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
#    except:
#        coll = createChunks(sentences, 700)
#        for val in coll:
#            numTokens = len(tokenizer.encode(val))
#            maxLength = min(500, numTokens)
#            minLength = min(100, numTokens)
#            summary.append(summarizer(val, max_length=maxLength, min_length=minLength, do_sample=False))

    result = str(summary)
    result = result.replace("[[{'summary_text': '", '').replace("[[{'summary_text': ", '').replace("'}], ", ' ').replace("}], ", ' ').replace("[{'summary_text': '", '').replace("[{'summary_text': ", '').replace("'}]]", '').replace("}]]", '')

    if inputLang == outputLang:
        return result
    

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


with gr.Blocks() as demo:
    inp = gr.Audio(type='filepath')
    radio = gr.Radio(['English', 'Dutch', 'Spanish', 'Italian', 'Portugese', 'German', 'Other'], value='English', label='Input Language')
    radio2 = gr.Radio(['English', 'Dutch'], value='English', label='Output Language')
    button = gr.Button('Summarize')
    outp = gr.Textbox(label='Summary')

    button.click(transcribe, inputs=[inp, radio, radio2], outputs=outp)

if __name__ == "__main__":
    demo.launch()  