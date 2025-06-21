import streamlit as st
import pandas as pd
import plotly.express as px

# Sample data
data = pd.read_csv("output.csv")

st.title("Category Prediction Viewer")

# 1. Bar chart of category counts
st.subheader("Category Distribution")
category_counts = data['main_category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']
fig = px.bar(category_counts, x='category', y='count', title='Count by Category')
st.plotly_chart(fig)

# 2. First selector - Choose a category
selected_category = st.selectbox("Choose a Category", sorted(data['main_category'].unique()))

# 3. Second selector - Choose an item from selected category
filtered_data = data[data['main_category'] == selected_category]
selected_item = st.selectbox("Choose an Item", filtered_data['description'].tolist())

# 4. Infobar - Show if prediction matches category
item_row = data[data['description'] == selected_item].iloc[0]
actual_category = item_row['category']
predicted_category = item_row['zeroshot_pred']

st.markdown("### Prediction Info")
if predicted_category == actual_category:
    st.success(f"✅ Prediction matches the category: {predicted_category}")
else:
    st.error(f"❌ Prediction does not match. Predicted: {predicted_category}, Actual: {actual_category}")
