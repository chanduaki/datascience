import streamlit as st
import pandas as pd
from gemini_llm import get_gemini_gen_ai_summarised_data, get_areas_of_improvement

# Load the dataset
df = pd.read_csv('books_data_with_sentiment_score_final.csv')


# Helper functions
def get_filtered_df(df, authors, titles, feedback_type):
    # Filter based on author(s), title(s), and feedback type
    if 'All' not in authors:
        df = df[df['author'].isin(authors)]
    if 'All' not in titles:
        df = df[df['book title'].isin(titles)]
    if feedback_type == ['All']:
        feedback_type = ['Positive', 'Negative', 'Neutral']
    if 'All' not in feedback_type:
        df = df[df['Overall_Sentiment_Label'].isin(feedback_type)]
    return df


def get_metrics(df):
    # Metrics Calculation
    metrics = []
    for author in df['author'].unique():
        author_df = df[df['author'] == author]
        for book in author_df['book title'].unique():
            book_df = author_df[author_df['book title'] == book]
            metrics.append({
                'Author': author,
                'Book Title': book,
                'Positive Count': (book_df['Overall_Sentiment_Label'] == 'Positive').sum(),
                'Negative Count': (book_df['Overall_Sentiment_Label'] == 'Negative').sum(),
                'Neutral Count': (book_df['Overall_Sentiment_Label'] == 'Neutral').sum(),
                'Avg Rating': book_df['Overall_Sentiment_Score'].mean()
            })
    return pd.DataFrame(metrics)


def get_feedback_summary(df, feedback_type):
    if feedback_type and 'All' not in feedback_type:
        feedback_df = df[df['Overall_Sentiment_Label'].isin(feedback_type)]
    else:
        feedback_df = df
    positive_feedback = feedback_df[feedback_df['Overall_Sentiment_Label'] == 'Positive']
    negative_feedback = feedback_df[feedback_df['Overall_Sentiment_Label'] == 'Negative']

    positive_summary = ' '.join(positive_feedback['review description'].tolist())
    negative_summary = ' '.join(negative_feedback['review description'].tolist())

    return positive_summary, negative_summary


# Streamlit UI
st.title('Book Feedback Analysis')

# Filters
all_authors = ['All'] + df['author'].unique().tolist()
all_titles = ['All'] + df['book title'].unique().tolist()
all_feedback_types = ['All', 'Positive', 'Negative', 'Neutral']

# Set default selections
default_author = all_authors[1]  # Select the first author by default
default_title = all_titles[1]  # Select the first book title by default
default_feedback_types = all_feedback_types  # Show all feedback types by default

authors = st.multiselect('Select Author(s):', options=all_authors, default=[default_author])
titles = st.multiselect('Select Book Title(s):', options=all_titles, default=[default_title])
feedback_type = st.multiselect('Select Feedback Type(s):', options=all_feedback_types, default=default_feedback_types)

filtered_df = get_filtered_df(df, authors, titles, feedback_type)

# Main Tabs
tabs = st.tabs(['Metrics Display', 'Positive/Negative Feedback', 'Areas of Improvement'])

with tabs[0]:
    st.subheader('Metrics Display')
    metrics_df = get_metrics(filtered_df)
    st.dataframe(metrics_df, use_container_width=True)  # Display in table format

with tabs[1]:
    feedback_tabs = st.tabs(['Positive Summary', 'Negative Summary'])

    with feedback_tabs[0]:
        st.subheader('Positive Feedback Summary')
        positive_summary, _ = get_feedback_summary(filtered_df, feedback_type)
        st.write(get_gemini_gen_ai_summarised_data(positive_summary))

    with feedback_tabs[1]:
        st.subheader('Negative Feedback Summary')
        _, negative_summary = get_feedback_summary(filtered_df, feedback_type)
        st.write(get_gemini_gen_ai_summarised_data(negative_summary))

with tabs[2]:
    st.subheader('Areas of Improvement')
    negative_feedback = ' '.join(
        filtered_df[filtered_df['Overall_Sentiment_Label'] == 'Negative']['review description'].tolist())
    print(negative_feedback)
    areas_of_improvement = get_areas_of_improvement(negative_feedback)
    st.write(areas_of_improvement)
