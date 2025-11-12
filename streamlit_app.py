import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
import json
import numpy as np
import re

# --- CUSTOM CSS STYLİNG ---
def load_custom_css():
    st.markdown("""
    <style>
    /* Ana sayfa arka plan ve tipografi */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Başlık stilleri */
    h1 {
        color: white !important;
        text-align: center;
        font-weight: 700 !important;
        font-size: 3rem !important;
        margin-bottom: 1rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    h2 {
        color: #667eea !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        color: #764ba2 !important;
        font-weight: 500 !important;
    }
    
    /* Tab stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        border-radius: 10px;
        padding: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8f9fa;
        border-radius: 8px;
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Kart stilleri */
    [data-testid="stContainer"] {
        background-color: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        border: none !important;
    }
    
    /* Textarea stilleri */
    textarea {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.9rem !important;
    }
    
    textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Button stilleri */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metric kartları */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #666 !important;
    }
    
    /* Expander stilleri */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Divider stilleri */
    hr {
        margin: 2rem 0 !important;
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #667eea, transparent) !important;
    }
    
    /* Success/Warning/Error mesaj stilleri */
    .stSuccess, .stWarning, .stError {
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    
    /* Input field stilleri */
    input {
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Spinner stilleri */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS'i yükle
load_custom_css()

# --- 1. FIREBASE BAĞLANTISI ---
@st.cache_resource
def init_firebase():
    try:
        creds_dict = dict(st.secrets["firebase_credentials"])
        creds_dict["private_key"] = creds_dict["private_key"].replace(r'\n', '\n')
        creds = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(creds)
    except ValueError:
        pass
    except Exception as e:
        st.error(f"🔥 FİREBASE BAŞLATMA HATASI: {e}")
        st.stop()
    return firestore.client()

# --- 2. GEMINI AI BAĞLANTISI ---
@st.cache_resource
def init_gemini():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        analysis_model = genai.GenerativeModel('models/gemini-flash-latest')
        embedding_model = genai.GenerativeModel('models/text-embedding-004')
        return analysis_model, embedding_model
    except Exception as e:
        st.error(f"💎 GEMİNİ BAŞLATMA HATASI: {e}")
        st.stop()

# --- UYGULAMA BAŞLANGICI ---
st.markdown("""
<div style='text-align: center; padding: 1rem 0 2rem 0;'>
    <h1 style='margin-bottom: 0.5rem;'>🤖 AI CV Matching Platform</h1>
    <p style='color: white; font-size: 1.2rem; font-weight: 300;'>Powered by Google Gemini AI & Firebase</p>
</div>
""", unsafe_allow_html=True)

try:
    db = init_firebase()
    gemini_model, embedding_model = init_gemini()
except Exception as e:
    st.error("Uygulama başlatılırken kritik bir hata oluştu. Lütfen 'Secrets' ayarlarınızı kontrol edin.")
    st.stop()

# --- YARDIMCI FONKSİYONLAR ---
@st.cache_data(ttl=300) 
def get_job_postings_with_vectors():
    jobs = []
    try:
        docs = db.collection("job_postings").where("vector", "!=", None).stream()
        for doc in docs:
            job_data = doc.to_dict()
            jobs.append({
                "id": doc.id,
                "title": job_data.get("title", "No Title"),
                "description": job_data.get("description", "No Description"),
                "vector": job_data.get("vector")
            })
        return jobs
    except Exception as e:
        st.error(f"İş ilanları çekilirken hata oluştu: {e}")
        return []

def extract_score_from_text(text):
    match = re.search(r"Overall Compatibility Score:.*?(\d{1,3})", text, re.IGNORECASE | re.DOTALL)
    if match:
        return int(match.group(1))
    else:
        return None

def get_gemini_analysis(cv, job_post):
    prompt = f"""
    You are a senior Human Resources (HR) specialist... (Prompt metni aynı)
    ...
    1.  **Overall Compatibility Score:** Rate the CV's suitability... on a scale of 100.
    ...
    ---[CV TEXT]----
    {cv}
    -----------------
    ---[JOB POSTING TEXT]---
    {job_post}
    -----------------
    """
    try:
        response = gemini_model.generate_content(prompt)
        analysis_text = response.text
        score = extract_score_from_text(analysis_text)
        return analysis_text, score
    except Exception as e:
        return f"An error occurred during analysis: {e}", None

def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Metnin 'parmak izi' alınırken hata oluştu: {e}")
        return None

# --- ANA UYGULAMA ARAYÜZÜ ---
tab1, tab2 = st.tabs(["🎯 Auto-Matcher: Find Jobs for Me", "➕ Add New Job Posting"])

# --- Sekme 1: OTOMATİK CV EŞLEŞTİRİCİ ---
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hero section
    col_hero1, col_hero2, col_hero3 = st.columns([1, 2, 1])
    with col_hero2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea; margin-top: 0;'>🚀 Discover Your Perfect Job Match</h2>
            <p style='color: #666; font-size: 1.1rem; line-height: 1.6;'>
                Paste your CV below and let our advanced AI analyze it against thousands of job postings 
                to find your top 3 most compatible opportunities.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("**📄 Your CV Text**")
        cv_text = st.text_area(
            "Paste your full CV here", 
            height=350, 
            label_visibility="collapsed",
            placeholder="Paste your complete CV text here (education, experience, skills, etc.)..."
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        search_button = st.button("🔍 Find My Perfect Matches", type="primary", use_container_width=True)
    
    if search_button:
        if cv_text:
            with st.spinner("🤖 AI is analyzing your CV and searching our database..."):
                all_jobs = get_job_postings_with_vectors()
                
                if not all_jobs:
                    st.warning("⚠️ No job postings found. Please add jobs in the 'Add New Job' tab first.")
                    st.stop()
                
                cv_vector = get_embedding(cv_text)
                
                if cv_vector:
                    job_vectors = np.array([job['vector'] for job in all_jobs])
                    cv_vector_np = np.array(cv_vector)
                    similarities = np.dot(job_vectors, cv_vector_np)
                    
                    top_indices = np.argsort(similarities)[-3:][::-1]
                    top_scores = [similarities[i] for i in top_indices]

                    st.success(f"✅ Found {len(top_indices)} excellent matches! Analyzing compatibility now...")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Sonuçlar için özel header
                    st.markdown("""
                    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 10px; margin-bottom: 2rem;'>
                        <h3 style='color: white; margin: 0;'>🎯 Your Top 3 Job Matches</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, index in enumerate(top_indices):
                        matched_job = all_jobs[index]
                        rank = i + 1
                        
                        analysis_text, score = get_gemini_analysis(cv_text, matched_job['description'])
                        
                        # Renge göre emoji ve renk seçimi
                        if score and score >= 80:
                            emoji = "🌟"
                            color = "#10b981"
                        elif score and score >= 60:
                            emoji = "⭐"
                            color = "#f59e0b"
                        else:
                            emoji = "💡"
                            color = "#6b7280"
                        
                        with st.container(border=True):
                            # Üst kısım: Rank badge + Başlık
                            st.markdown(f"""
                            <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
                                <div style='background: {color}; color: white; padding: 0.5rem 1rem; 
                                            border-radius: 20px; font-weight: 700; font-size: 1rem;'>
                                    {emoji} Rank #{rank}
                                </div>
                                <h3 style='margin: 0; color: #1f2937;'>{matched_job['title']}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([0.25, 0.75])
                            
                            with col1:
                                st.metric(
                                    label="Match Score",
                                    value=f"{score}%" if score else "N/A",
                                    help="AI-generated compatibility (0-100%)"
                                )
                            
                            with col2:
                                with st.expander("📊 View Detailed AI Analysis", expanded=False):
                                    st.markdown(analysis_text)
                        
                        if i < len(top_indices) - 1:
                            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please paste your CV text to start matching.")

# --- Sekme 2: YENİ İLAN EKLEME ---
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_hero1, col_hero2, col_hero3 = st.columns([1, 2, 1])
    with col_hero2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea; margin-top: 0;'>➕ Add New Job to Database</h2>
            <p style='color: #666; font-size: 1.1rem; line-height: 1.6;'>
                When you save a job posting, our AI automatically generates its semantic fingerprint 
                for intelligent matching with future candidates.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.form("new_job_form", clear_on_submit=True):
        with st.container(border=True):
            st.markdown("**📌 Job Title**")
            job_title = st.text_input(
                "Enter the job title", 
                label_visibility="collapsed",
                placeholder="e.g., Senior Software Engineer, Data Scientist, Product Manager..."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("**📝 Job Description**")
            job_description = st.text_area(
                "Enter the full job description", 
                height=350, 
                label_visibility="collapsed",
                placeholder="Paste the complete job description including responsibilities, requirements, qualifications, etc."
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_submit1, col_submit2, col_submit3 = st.columns([1, 1, 1])
        with col_submit2:
            submitted = st.form_submit_button("💾 Save Job & Generate AI Fingerprint", use_container_width=True)
        
        if submitted:
            if job_title and job_description:
                with st.spinner("🧠 Generating AI fingerprint (vector) for this job..."):
                    job_vector = get_embedding(f"Title: {job_title}\n\nDescription: {job_description}")
                
                if job_vector:
                    try:
                        doc_ref = db.collection("job_postings").document()
                        doc_ref.set({
                            "title": job_title,
                            "description": job_description,
                            "created_at": firestore.SERVER_TIMESTAMP,
                            "vector": job_vector
                        })
                        st.success(f"✅ Successfully added '{job_title}' with AI fingerprint to database!")
                        st.balloons()
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"❌ Error saving to Firebase: {e}")
                else:
                    st.error("❌ Could not generate AI fingerprint. Job not saved.")
            else:
                st.warning("⚠️ Please fill in both Job Title and Job Description.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem; font-size: 0.9rem;'>
    <p>Built with ❤️ using Streamlit, Google Gemini AI & Firebase</p>
    <p style='opacity: 0.8;'>v2.5 - Visual Edition</p>
</div>
""", unsafe_allow_html=True)
