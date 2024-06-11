import streamlit as st
import gensim
from gensim import corpora, models
# import openai-whisper
import os
from pytube import YouTube
from pathlib import Path
import pandas as pd


#topic modelling
from sklearn.feature_extraction.text import CountVectorizer #issues
#from umap import UMAP #issues
#from hdbscan import HDBSCAN
from bertopic import BERTopic 
from bertopic.vectorizers import ClassTfidfTransformer


def perform_BERT_topic_modeling(text):
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    # Use sklearn CountVectorizer to remove stopwords after having generated embeddings, and train model
    vectorizer_model = CountVectorizer(stop_words="english")

    #Train BERTopic model
    topic_model = BERTopic(
        language="multilingual",
        ctfidf_model=ctfidf_model,
        # umap_model=umap_model,
        # hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True,
        nr_topics="auto",
        #  low_memory = True,
        top_n_words=6,
        # min_topic_size = 15,
        # diversity=0.5
    )
    # Prep data for modelling
    #docs = df_topic["Review Text"].reset_index().drop(columns="index").to_numpy().ravel()
    #topics, probs = topic_model.fit_transform(text)
    
    text_str = str(text)
    print(text_str)
    text_list = text_str.split()
    print(text_list)
    
    list_version = [text_str]
    
    topics, probs = topic_model.fit_transform(list_version)
    
    # Show topic distribution of largest n topics
    freq = topic_model.get_topic_info()

    
    return freq

def perform_topic_modeling(transcript_text, num_topics=5, num_words=10):
    # Preprocess the transcript text
    # Replace this with your own preprocessing code
    preprocessed_text = preprocess_text(transcript_text)

    # Create a dictionary of all unique words in the transcripts
    dictionary = corpora.Dictionary(preprocessed_text)

    # Convert the preprocessed transcripts into a bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text]

    # Train an LDA model with the specified number of topics
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    # Extract the most probable words for each topic
    topics = []
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        # Extract the top words for each topic and store in a list
        topic_words = [word.split('*')[1].replace('"', '').strip() for word in topic.split('+')]
        topics.append((f"Topic {idx}", topic_words))

    return topics
    
def preprocess_text(text):
    # Replace this with your own preprocessing code
    # This example simply tokenizes the text and removes stop words
    tokens = gensim.utils.simple_preprocess(text)
    stop_words = gensim.parsing.preprocessing.STOPWORDS
    preprocessed_text = [[token for token in tokens if token not in stop_words]]

    return preprocessed_text

@st.cache_resource
# def load_model():
#     model = whisper.load_model("base")
#     return model

def save_video(url, video_filename):
    youtubeObject = YouTube(url)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")
    
    return video_filename

def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    try:
        os.rename(out_file, file_name)
    except WindowsError:
        os.remove(file_name)
        os.rename(out_file, file_name)
    audio_filename = Path(file_name).stem+'.mp3'
    video_filename = save_video(url, Path(file_name).stem+'.mp4')
    print(yt.title + " Has been successfully downloaded")
    return yt.title, audio_filename, video_filename

# def audio_to_transcript(audio_file):
#     model = load_model()
#     result = model.transcribe(audio_file)
#     transcript = result["text"]
#     return transcript
    
st.set_page_config(layout="wide")

choice = st.sidebar.selectbox("Select your choice", ["On Text","Bert", "On Video", "On CSV"])


if choice == "On Text":
    
    st.subheader("Topic Modeling and Labeling on Text")

    # Create a text area widget to allow users to paste transcripts
    text_input = st.text_area("Paste enter text below", height=400)

    if text_input is not None:

        if st.button("Analyze Text"):
            col1, col2= st.columns([1,1])
            with col1:
                st.info("Text is below")
                st.success(text_input)
            with col2:
                # Perform topic modeling on the transcript text
                topics = perform_topic_modeling(text_input)

                # Display the resulting topics in the app
                st.info("Topics in the Text")
                for topic in topics:
                    st.success(f"{topic[0]}: {', '.join(topic[1])}")
                    
elif choice == "Bert":
    
    st.subheader("Bert Topic Modeling and Labeling on Text")

    # Create a text area widget to allow users to paste transcripts
    text_input = st.text_area("Paste enter text below", height=400)

    if text_input is not None:

        if st.button("Analyze Text"):
            col1, col2= st.columns([1,1])
            with col1:
                st.info("Text is below")
                st.success(text_input)
            with col2:
                # Perform topic modeling on the transcript text
                topics = perform_BERT_topic_modeling(text_input)

                # Display the resulting topics in the app
                st.info("Topics in the Text")
                for topic in topics:
                    st.success(f"{topic[0]}: {', '.join(topic[1])}")
    
         
elif choice == "On Video":
    st.subheader("Topic Modeling and Labeling on Video")
    url =  st.text_input('Enter URL of YouTube video:')
    # if upload_video is not None:
    if st.button("Analyze Video"):
        col1, col2 = st.columns([1,1])
        with col1:
            st.info("Video uploaded successfully")
            video_title, audio_filename, video_filename = save_audio(url)
            st.video(video_filename)
        with col2:
            st.info("Transcript is below") 
            print(audio_filename)
            transcript_result = "blachblahballajnaogjbagg"#audio_to_transcript(audio_filename)
            st.success(transcript_result)
                
elif choice == "On CSV":
    st.subheader("Topic Modeling and Labeling on CSV File")
    upload_csv = st.file_uploader("Upload your CSV file", type=['csv'])
    if upload_csv is not None:
        if st.button("Analyze CSV File"):
            col1, col2 = st.columns([1,2])
            with col1:
                st.info("CSV File uploaded")
                csv_file = upload_csv.name
                with open(os.path.join(csv_file),"wb") as f: 
                    f.write(upload_csv.getbuffer()) 
                print(csv_file)
                df = pd.read_csv(csv_file, encoding= 'unicode_escape')
                st.dataframe(df)
            with col2:
                data_list = df['Data'].tolist()
                industry_list = []
                for i in data_list:
                    industry = label_topic(i)
                    industry_list.append(industry)
                df['Industry'] = industry_list
                st.info("Topic Modeling and Labeling")
                st.dataframe(df)
                
            
