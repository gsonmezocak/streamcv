import streamlit as st
import google.generativeai as genai
import time

# --- Page Setup ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖"
)

# --- Title and Description ---
st.title("🤖 AI Powered CV - Job Posting Matcher (MVP)")
st.markdown("This application analyzes the compatibility between a CV and a job posting using.")

# --- API Key Authentication ---
# Securely fetching the API key using Streamlit's secrets feature.
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("API Key not found or invalid. Please check your secrets.toml file.")
    st.stop() # Stop the app if there's an error

# --- Configure Gemini Model ---
# We are using the 'flash' model which is optimized for speed.
model = genai.GenerativeModel('models/gemini-flash-latest') # <-- FAST MODEL

# --- Prompt Design ---
def create_prompt(cv, job_post):
    return f"""
    You are a senior Human Resources (HR) specialist, and your task is to compare a CV with a job posting.
    Analyze the following CV text and JOB POSTING text in detail.

    Follow these steps in your analysis:
    1.  **Overall Compatibility Score:** Rate the CV's suitability for the job posting on a scale of 100.
    2.  **Strengths (Pros):** List the top 3-4 strengths of the candidate that meet the job requirements.
    3.  **Weaknesses / Gaps (Cons):** List 3-4 key points mentioned in the job posting that are missing or weak in the CV.
    4.  **Evaluation Summary:** Write a brief 2-3 sentence overall evaluation summary.

    Please provide your answer in **Markdown format** using clear headings.

    ---[CV TEXT]----
    {cv}
    -----------------

    ---[JOB POSTING TEXT]---
    {job_post}
    -----------------
    """

# --- User Interface (UI) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 CV Text")
    cv_text = st.text_area("Paste the candidate's CV text here", height=300, label_visibility="collapsed")

with col2:
    st.subheader("🎯 Job Posting Text")
    ilan_text = st.text_area("Paste the job posting text here", height=300, label_visibility="collapsed")

# --- Button and Logic ---
if st.button("Run Compatibility Analysis", type="primary", use_container_width=True):
    if cv_text and ilan_text:
        # Show a loading spinner while processing
        with st.spinner("We are analyzing the CV and job post... Please wait."):
            try:
                # Create the prompt
                prompt = create_prompt(cv_text, ilan_text)
                
                # Send the request to Gemini API
                response = model.generate_content(prompt)
                
                # Print the result
                st.divider()
                st.subheader("✨ Analysis Result")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("Please fill in both the CV and Job Posting fields.")
