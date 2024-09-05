import os
import subprocess
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import spacy
import numpy as np
from textstat import flesch_reading_ease
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_mp4_to_wav(mp4_file, wav_file):
    command = f"ffmpeg -i {mp4_file} -q:a 0 -map a {wav_file}"
    subprocess.call(command, shell=True)
    print(f"Converted {mp4_file} to {wav_file}")

def split_wav_file(wav_file, chunk_length=600):
    command = f"ffmpeg -i {wav_file} -f segment -segment_time {chunk_length} -c copy chunk_%03d.wav"
    subprocess.call(command, shell=True)
    chunks = [f for f in os.listdir('.') if f.startswith('chunk_') and f.endswith('.wav')]
    print(f"Split WAV file into chunks: {chunks}")
    return chunks

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']

def extract_keywords(text, num_keywords=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    indices = tfidf_matrix[0].toarray().argsort()[0][-num_keywords:]
    features = vectorizer.get_feature_names_out()
    keywords = [features[i] for i in indices]
    return keywords

def extract_named_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def readability_score(text):
    return flesch_reading_ease(text)

def content_overlap(original_text, summary_text):
    original_words = set(re.findall(r'\w+', original_text.lower()))
    summary_words = set(re.findall(r'\w+', summary_text.lower()))
    overlap = len(original_words & summary_words) / len(original_words) * 100
    return overlap

def advanced_summarize_text(text):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=1024,  
        truncation=True
    )
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=500,
        min_length=200,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_with_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    X = vectorizer.fit_transform([text])
    scores = X.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    top_indices = scores.argsort()[-10:][::-1]
    summary = ' '.join([feature_names[i] for i in top_indices])
    return summary

def summarize_with_lda(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topic_words = lda.components_[0]
    top_words = [feature_names[i] for i in topic_words.argsort()[-10:]]
    summary = ' '.join(top_words)
    return summary

def summarize_with_frequency(text):
    words = text.split()
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(10)
    summary = ' '.join([word for word, _ in most_common_words])
    return summary

def summarize_mp4(mp4_file):
    wav_file = "output_audio.wav"
    convert_mp4_to_wav(mp4_file, wav_file)
    wav_chunks = split_wav_file(wav_file)
    
    full_transcription = ""
    for chunk in wav_chunks:
        transcribed_text = transcribe_audio(chunk)
        full_transcription += transcribed_text + " "
    
    tfidf_summary = summarize_with_tfidf(full_transcription)
    lda_summary = summarize_with_lda(full_transcription)
    freq_summary = summarize_with_frequency(full_transcription)
    advanced_summary = advanced_summarize_text(full_transcription)
    
    for chunk in wav_chunks:
        os.remove(chunk)
    
    # Compute readability scores and content overlap
    tfidf_readability = readability_score(tfidf_summary)
    lda_readability = readability_score(lda_summary)
    freq_readability = readability_score(freq_summary)
    advanced_readability = readability_score(advanced_summary)
    
    tfidf_overlap = content_overlap(full_transcription, tfidf_summary)
    lda_overlap = content_overlap(full_transcription, lda_summary)
    freq_overlap = content_overlap(full_transcription, freq_summary)
    advanced_overlap = content_overlap(full_transcription, advanced_summary)
    
    print(f"TF-IDF Summary Readability Score: {tfidf_readability:.2f}")
    print(f"LDA Summary Readability Score: {lda_readability:.2f}")
    print(f"Frequency-Based Summary Readability Score: {freq_readability:.2f}")
    print(f"Advanced Summary Readability Score: {advanced_readability:.2f}")
    
    print(f"TF-IDF Summary Content Overlap: {tfidf_overlap:.2f}%")
    print(f"LDA Summary Content Overlap: {lda_overlap:.2f}%")
    print(f"Frequency-Based Summary Content Overlap: {freq_overlap:.2f}%")
    print(f"Advanced Summary Content Overlap: {advanced_overlap:.2f}%")
    
    with open("video_summary.txt", "w") as summary_file:
        summary_file.write(f"TF-IDF Summary:\n{tfidf_summary}\n\n")
        summary_file.write(f"LDA Summary:\n{lda_summary}\n\n")
        summary_file.write(f"Frequency-Based Summary:\n{freq_summary}\n\n")
        summary_file.write(f"Advanced Summary:\n{advanced_summary}\n")
    
    return tfidf_summary, lda_summary, freq_summary, advanced_summary

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            tfidf_summary, lda_summary, freq_summary, advanced_summary = summarize_mp4(file_path)
            return render_template('index.html', 
                                   tfidf_summary=tfidf_summary,
                                   lda_summary=lda_summary,
                                   freq_summary=freq_summary,
                                   advanced_summary=advanced_summary)
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
