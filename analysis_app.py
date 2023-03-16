import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud
from nltk.probability import FreqDist
import time

from utils.analysis import get_corpus, most_common_words, Bigrams, Trigrams


df = pd.read_csv('./sample_dataset/labeled_dataset.csv')
df['word_count'] = df['Prompt'].apply(lambda x: len(x.split()))
sns.set(rc={'axes.facecolor':'#FFFFFF', 'figure.facecolor':'#FFFFFF'})


image = Image.open('image/southfields_logo.png')
st.image(image)

st.write(""" # South-Fields Analysis """)
selected_sport = st.multiselect("Selecteer een type sport:",
               max_selections=1,
               options=df.Type_sport.unique(),
               default="Voetbal"
               )

with st.spinner("Just a moment ..."):
    time.sleep(1)

if selected_sport != []:
    fig = plt.figure(figsize=(12,15))
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5)
    plt.subplots_adjust(hspace=0.5)
    sns.histplot(
        df.loc[df.Type_sport == selected_sport[0]], x='word_count', kde=True, 
        color="#100c44", binwidth = 1, alpha = 0.7, ax=ax1
        )
    ax1.set_title('Total amount of words')

    corpus = get_corpus(df)
    words, freq = most_common_words(corpus)
    sns.barplot(x=freq, y=words, color="#100c44", ax=ax2)
    ax2.set_title('Top 10 Most Frequently Occuring Words')

    wordcloud= WordCloud(max_font_size=60, max_words=100,width=1000,height=200, stopwords=STOPWORDS, background_color='#FFFFFF').generate_from_frequencies(
    FreqDist([word for prompt in df.Prompt_lists for word in prompt])
    )
    ax3.imshow(wordcloud)
    ax3.grid(visible=False)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ngram_freq, ngram_type = Bigrams(df)
    sns.barplot(x=ngram_freq['frequency'][:10], y=ngram_freq['ngram'][:10], color="#100c44", ax=ax4)
    ax4.set_title('Top 10 Most Frequently Occuring {}'.format(ngram_type))

    ngram_freq, ngram_type = Trigrams(df)
    sns.barplot(x=ngram_freq['frequency'][:10], y=ngram_freq['ngram'][:10], color="#100c44", ax=ax5)
    ax5.set_title('Top 10 Most Frequently Occuring {}'.format(ngram_type))

    st.pyplot(fig)