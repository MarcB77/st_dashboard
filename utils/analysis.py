import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from tqdm import trange
from wordcloud import STOPWORDS



def _corpus(text):
    #text_list = text.split()
    text_list = [word for word in text.lower().split() if word not in STOPWORDS]
    return text_list

def get_corpus(df):
    df['Prompt_lists'] = df['Prompt'].apply(_corpus)
    corpus = []
    for i in trange(df.shape[0], ncols=150, nrows=10, colour='green', smoothing=0.8):
        corpus += df['Prompt_lists'][i]
    len(corpus)
    return corpus

def most_common_words(corpus, amount_words):
    mostCommon = Counter(corpus).most_common(amount_words)
    words = []
    freq = []
    for word, count in mostCommon:
        words.append(word)
        freq.append(count)
    return words, freq

def Bigrams(df):
    cv = CountVectorizer(ngram_range=(2,2))
    bigrams = cv.fit_transform(df['Prompt'])

    count_values = bigrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse = True))
    ngram_freq.columns = ["frequency", "ngram"]
    return ngram_freq, "Bigrams"

def Trigrams(df):
    cv1 = CountVectorizer(ngram_range=(3,3))
    trigrams = cv1.fit_transform(df['Prompt'])

    count_values = trigrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv1.vocabulary_.items()], reverse = True))
    ngram_freq.columns = ["frequency", "ngram"]
    return ngram_freq, "Trigrams"