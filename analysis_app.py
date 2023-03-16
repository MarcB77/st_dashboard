import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image

df = pd.read_csv('./sample_dataset/labeled_dataset.csv')

image = Image.open('image/southfields_logo.png')
st.image(image)

st.write(""" # South-Fields Analysis """)
selected_sport = st.multiselect("Select het type sport waarvan je een analyze wilt doen",
               max_selections=1,
               options=df.Type_sport.unique()
               )

word_count_plot = sns.histplot(
    df.loc[df.Type_sport == selected_sport[0]], x='word_count', kde=True, 
    palette="Blues", binwidth = 1, alpha = 0.6
    )
st.pyplot(word_count_plot)