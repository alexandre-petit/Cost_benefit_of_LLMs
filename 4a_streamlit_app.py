import streamlit as st
import pandas as pd
import plotly.express as px

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
