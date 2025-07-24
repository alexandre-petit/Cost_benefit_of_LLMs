import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

# Sample data
data = pd.read_csv("output_df.csv")

st.title("Category Prediction Viewer")

# 1. Bar chart of category counts
st.subheader("Category Distribution")
category_counts = data['true_label'].value_counts().reset_index()
category_counts.columns = ['category', 'count']
fig = px.bar(category_counts, x='category', y='count', title='Count by Category')
st.plotly_chart(fig)

#ajouter analyse descriptive et wordcould
stopwords = set(STOPWORDS)

def filter_words(word_list: list) -> list:
    output_list = []
    for word in word_list:
        if len(word) > 2 and word not in stopwords:
            output_list.append(word)
    
    return output_list

data["words"] = data["name_description"].str.split(" ")

data["words"] = [filter_words(l) for l in data["words"]]

exploded_words = data.explode("words")

text = " ".join(data['name_description'])

word_counts = exploded_words["words"].value_counts().reset_index()
word_counts.columns = ["Word", "Frequency"]

st.subheader(" ".join(word_counts.columns))

st.subheader("Most common words")
fig_3 = px.bar(word_counts.head(10), y="Word", x="Frequency")
st.plotly_chart(fig_3)

wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      stopwords=stopwords).generate(text)

st.subheader("Wordcloud")
fig_2, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_2)

# 2. First selector - Choose a category
st.subheader("Select item to see model prediction")
selected_category = st.selectbox("Choose a Category", sorted(data['true_label'].unique()))

# 3. Second selector - Choose an item from selected category
filtered_data = data[data['true_label'] == selected_category]
selected_item = st.selectbox("Choose an Item", filtered_data['name_description'].tolist())

# 4. Infobar - Show if prediction matches category
item_row = data[data['name_description'] == selected_item].iloc[0]
actual_category = item_row['true_label']
predicted_category = item_row['zeroshot_pred']

st.markdown("### Prediction Info")
if predicted_category == actual_category:
    st.success(f"✅ Prediction matches the category: {predicted_category}")
else:
    st.error(f"❌ Prediction does not match. Predicted: {predicted_category}, Actual: {actual_category}")
