import nltk
nltk.download('vader_lexicon')

from googleapiclient.discovery import build
import streamlit as st
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from pymongo import MongoClient
import bcrypt
from keras.preprocessing.text import text_to_word_sequence


# MongoDB connection
client = MongoClient('mongodb+srv://ashbelraj:ashbelraj@cluster-filter.x9df0py.mongodb.net/')  # Replace with your MongoDB connection string
db = client['youtube_sentiment']
users_collection = db['users']
analyses_collection = db['analyses']

# Define functions
def register_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user_data = {'username': username, 'password': hashed_password, 'history': []}
    users_collection.insert_one(user_data)

def login_user(username, password):
    user_data = users_collection.find_one({'username': username})
    if user_data:
        if bcrypt.checkpw(password.encode('utf-8'), user_data['password']):
            return True
    return False

def get_user_history(username):
    user_data = users_collection.find_one({'username': username})
    if user_data:
        return user_data.get('history', [])

def update_user_history(username, analysis_data):
    users_collection.update_one({'username': username}, {'$push': {'history': analysis_data}})

def get_comments(video_id, api_key, max_results=1000):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_results)
    while request is not None:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if item["snippet"]["totalReplyCount"] > 0:
                request_replies = youtube.comments().list(part="snippet", parentId=item["id"], maxResults=100)
                while request_replies is not None:
                    response_replies = request_replies.execute()
                    for item_reply in response_replies["items"]:
                        reply = item_reply["snippet"]["textDisplay"]
                        comments.append(reply)
                    request_replies = youtube.comments().list_next(request_replies, response_replies)
        request = youtube.commentThreads().list_next(request, response)
    return comments

def preprocess(comments):
    preprocessed = []
    for comment in comments:
        # Remove HTML tags and symbols
        comment = re.sub(r'<br\s*\/?>', ' ', comment)
        comment = re.sub(r'<hr\s*\/?>', ' ', comment)
        # Remove URLs
        comment = re.sub(r"http\S+|www\S+|https\S+", "", comment, flags=re.MULTILINE)
        # Remove HTML links and symbols
        comment = re.sub(r'<a\s+(?:[^>]?\s+)?href=(["\'])(.?)\1', '', comment)
        comment = comment.replace("\xa0", " ")
        # Replace "&#39;" with apostrophe
        comment = comment.replace("&#39;", "'")
        # Replace "a href" with empty
        comment=comment.replace("a href", "")
        # Convert to lowercase
        comment = comment.lower()
        # Tokenize text using Keras
        comment = text_to_word_sequence(comment)
        # Join tokens back into a string
        comment = ' '.join(comment)
        preprocessed.append(comment)
    return preprocessed

def analyze(preprocessed):
    vader = SentimentIntensityAnalyzer()
    scores = []
    for comment in preprocessed:
        polarity = vader.polarity_scores(comment)
        compound = polarity["compound"]
        scores.append(compound)
    return scores

def classify(scores):
    labels = []
    for score in scores:
        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        labels.append(label)
    return labels

def count_sentiments(labels):
    positive_count = labels.count('Positive')
    negative_count = labels.count('Negative')
    neutral_count = labels.count('Neutral')
    return positive_count, negative_count, neutral_count

def get_extremes(df):
    df = df.sort_values(by="Score")
    most_negative = df.iloc[0]["Comment"]
    most_negative_score = df.iloc[0]["Score"]
    most_positive = df.iloc[-1]["Comment"]
    most_positive_score = df.iloc[-1]["Score"]
    return most_positive, most_positive_score, most_negative, most_negative_score

# Define function to plot distribution
def plot_distribution(df):
    counts = df["Label"].value_counts()
    plt.bar(counts.index, counts.values)
    plt.title("Distribution of Sentiments")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    st.pyplot()  # Display the plot using Streamlit

# Define function to plot word cloud
def plot_word_cloud(df):
    all_comments = ' '.join(df['Comment'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)
    plt.figure(figsize=(6, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Comments')
    st.pyplot()  # Display the plot using Streamlit

# Define function to plot radial plot
def plot_radial_plot(df):
    r = df['Score']
    theta = np.linspace(0, 2*np.pi, len(df['Score']))

    cmap = plt.cm.get_cmap('coolwarm')
    new_cmap = cmap.reversed()

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.scatter(theta, r, c=r, cmap=new_cmap, s=50, alpha=0.75)
    ax.set_title('Radial Plot of Sentiment Scores')
    st.pyplot()  # Display the plot using Streamlit


# Streamlit app
def main():
    st.title("YouTube Comment Sentiment Analysis")
    page = st.sidebar.selectbox("Select Page", ["Login", "Register"])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if page == "Register":
        st.subheader("Register")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Register"):
            register_user(username, password)
            st.success("Registered successfully!")

    elif page == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login_user(username, password):
                st.success("Logged in successfully!")
                st.session_state.logged_in = True
                st.session_state.username = username
            else:
                st.error("Invalid username or password")

    if 'logged_in' in st.session_state and st.session_state.logged_in:
        st.subheader("Sentiment Analysis")
        youtube_url = st.text_input("Enter YouTube URL:")
        if st.button("Analyze"):
            api_key = "AIzaSyBsnzzxxjKNuyL2QMNtgaCJ4nKJo-V14-I"  # Replace with your actual YouTube API key
            video_id = youtube_url.split("?v=")[-1]
            if video_id:
                st.write("Fetching comments...")
                # Analyze comments and get sentiment analysis results
                comments = get_comments(video_id, api_key)
                st.write("Total no. of Comments")
                st.write(len(comments))
                preprocessed = preprocess(comments)
                scores = analyze(preprocessed)
                labels = classify(scores)
                df = pd.DataFrame({"Comment": preprocessed, "Score": scores, "Label": labels})
                plot_distribution(df)
                plot_word_cloud(df)
                plot_radial_plot(df)
                positive_count, negative_count, neutral_count = count_sentiments(labels)
                analysis_data = {'video_id': video_id, 'positives': positive_count, 'negatives': negative_count, 'neutrals': neutral_count}
                update_user_history(st.session_state.username, analysis_data)
                st.success("Analysis completed and saved to history.")
                
            else:
                st.write("Invalid YouTube URL. Please enter a valid URL.")

        # Display user's last 5 analyses
        st.subheader("History")
        username = st.session_state.username
        history = get_user_history(username)
        if history:
            st.write("Your last 5 analyses:")
            for analysis in history[-5:]:
                st.write(analysis)
        else:
            st.write("No analysis history.")

    # st.sidebar.text("By Your Name")

if _name_ == "_main_":
    main()
