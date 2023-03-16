import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import STOPWORDS, WordCloud
from nltk.probability import FreqDist



df = pd.read_csv('./sample_dataset/labeled_dataset.csv')
df['word_count'] = df['Prompt'].apply(lambda x: len(x.split()))
sns.set(rc={'axes.facecolor':'#100c44', 'figure.facecolor':'#FFFFFF'})

def _corpus(text):
    text_list = text.split()
    return text_list

def get_corpus(df):
    df['Prompt_lists'] = df['Prompt'].apply(_corpus)
    corpus = []
    for i in len(df.shape[0]):
        corpus += df['Prompt_lists'][i]
    len(corpus)
    return corpus

def most_common_words(corpus):
    mostCommon = Counter(corpus).most_common(10)
    words = []
    freq = []
    for word, count in mostCommon:
        words.append(word)
        freq.append(count)
    return word, freq


image = Image.open('image/southfields_logo.png')
st.image(image)

st.write(""" # South-Fields Analysis """)
selected_sport = st.multiselect("Select het type sport waarvan je een analyze wilt doen",
               max_selections=1,
               options=df.Type_sport.unique(),
               default="Voetbal"
               )


if selected_sport != []:
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    sns.histplot(
        df.loc[df.Type_sport == selected_sport[0]], x='word_count', kde=True, 
        palette="Blues", binwidth = 1, alpha = 0.6, ax=ax1
        )
    ax1.set_title('Total amount of words')

    corpus = get_corpus(df)
    words, freq = most_common_words(corpus)
    sns.barplot(x=freq, y=words, palette="Blues", ax=ax2)
    ax2.set_title('Top 10 Most Frequently Occuring Words')

    wordcloud= WordCloud(max_font_size=60, max_words=100,width=600,height=200, stopwords=STOPWORDS, background_color='#100c44').generate_from_frequencies(
    FreqDist([word for prompt in df.Prompt_lists for word in prompt])
    )
    ax3.imshow(wordcloud)

    st.pyplot(fig)