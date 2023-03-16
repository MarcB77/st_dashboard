import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

df = pd.read_csv('./sample_dataset/labeled_dataset.csv')
df['word_count'] = df['Prompt'].apply(lambda x: len(x.split()))

image = Image.open('image/southfields_logo.png')
st.image(image)

st.write(""" # South-Fields Analysis """)
selected_sport = st.multiselect("Select het type sport waarvan je een analyze wilt doen",
               max_selections=1,
               options=df.Type_sport.unique(),
               default="Voetbal"
               )

fig = plt.figure(figsize=(10,5))
sns.histplot(
    df.loc[df.Type_sport == selected_sport[0]], x='word_count', kde=True, 
    palette="Blues", binwidth = 1, alpha = 0.6
    )
fig.title('Total amount of words')

st.pyplot(fig)