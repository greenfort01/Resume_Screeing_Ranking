import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Resume Screening Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom functions for extracting text from various file formats
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(file):
    text = docx2txt.process(file)
    return text

def extract_text(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file)
    elif file_extension == '.docx':
        return extract_text_from_docx(file)
    elif file_extension == '.txt':
        return file.getvalue().decode('utf-8')
    else:
        return None

def preprocess_text(text):
    if text is None:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_similarity(job_description, resume_text):
    documents = [job_description, resume_text]
    
    # Create TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_sim

# Application title and description
st.title("📄 Resume Screening and Ranking Tool")
st.markdown("""
This application helps you screen and rank resumes based on their similarity to a job description.
Upload a job description and multiple resumes to find the best matches!
""")

# Sidebar for job description
st.sidebar.header("Job Description")
job_desc_option = st.sidebar.radio("Choose input method for job description:", 
                                   ["Upload File", "Text Input"])

job_description = ""

if job_desc_option == "Upload File":
    job_desc_file = st.sidebar.file_uploader("Upload Job Description", 
                                            type=["pdf", "docx", "txt"],
                                            key="job_desc")
    if job_desc_file is not None:
        job_description = extract_text(job_desc_file)
        st.sidebar.success("Job Description Uploaded Successfully!")
else:
    job_description = st.sidebar.text_area("Enter Job Description:", 
                                         height=300,
                                         key="job_desc_text")

# Main content area for resume uploads
st.header("Resume Upload Section")
uploaded_resumes = st.file_uploader("Upload Multiple Resumes", 
                                  accept_multiple_files=True,
                                  type=["pdf", "docx", "txt"])

# Process button
process_button = st.button("Screen Resumes")

# Display results if process button is clicked
if process_button:
    if not job_description:
        st.error("Please provide a job description!")
    elif not uploaded_resumes:
        st.error("Please upload at least one resume!")
    else:
        # Preprocess job description
        processed_job_desc = preprocess_text(job_description)
        
        # Process each resume and calculate similarity
        resume_results = []
        
        with st.spinner("Processing resumes..."):
            for resume in uploaded_resumes:
                resume_text = extract_text(resume)
                processed_resume = preprocess_text(resume_text)
                
                if processed_resume:
                    similarity_score = calculate_similarity(processed_job_desc, processed_resume)
                    match_percentage = similarity_score * 100
                    
                    # Extract some preview text (first 200 chars)
                    preview = processed_resume[:200] + "..." if len(processed_resume) > 200 else processed_resume
                    
                    resume_results.append({
                        'Resume': resume.name,
                        'Similarity Score': similarity_score,
                        'Match Percentage': match_percentage,
                        'Preview': preview
                    })
        
        if resume_results:
            # Sort results by similarity score (descending)
            resume_results.sort(key=lambda x: x['Similarity Score'], reverse=True)
            
            # Display results
            st.header("Screening Results")
            
            # Top candidates
            st.subheader("🏆 Top Candidates")
            
            # Create a dataframe for results
            results_df = pd.DataFrame(resume_results)
            
            # Display a table with the results
            st.dataframe(
                results_df[['Resume', 'Match Percentage']].rename(
                    columns={'Match Percentage': 'Match %'}
                ).style.format({'Match %': '{:.2f}%'})
            )
            
            # Visualize results with a bar chart
            st.subheader("📊 Visual Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort for visualization
            chart_data = results_df.sort_values(by='Match Percentage')
            
            # Create horizontal bar chart
            bars = ax.barh(chart_data['Resume'], chart_data['Match Percentage'], color='skyblue')
            ax.set_xlabel('Match Percentage (%)')
            ax.set_title('Resume Ranking by Similarity to Job Description')
            ax.set_xlim(0, 100)
            
            # Add percentage labels to the end of each bar
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}%', ha='left', va='center')
            
            st.pyplot(fig)
            
            # Detailed view of each resume with expandable sections
           # st.subheader("📝 Detailed Resume Analysis")
            
           # for i, result in enumerate(resume_results):
            #    with st.expander(f"{i+1}. {result['Resume']} - {result['Match Percentage']:.2f}%"):
             #       st.markdown(f"**Match Percentage:** {result['Match Percentage']:.2f}%")
              #      st.markdown("**Text Preview:**")
               #     st.text(result['Preview'])
        else:
            st.error("Could not process any of the uploaded resumes. Please check the file formats.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Scikit-learn")