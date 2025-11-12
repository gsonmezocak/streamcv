import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
import json
import numpy as np
import re

# --- ULTRA MODERN GLASSMORPHISM CSS ---
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Dark animated background */
    .main {
        background: linear-gradient(125deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', sans-serif;
        padding: 0;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism container */
    [data-testid="stContainer"] {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    [data-testid="stContainer"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Neon glow titles */
    h1 {
        background: linear-gradient(90deg, #00F5FF, #FF00FF, #00F5FF);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: neonFlow 3s linear infinite;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 0.5rem !important;
        filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.5));
    }
    
    @keyframes neonFlow {
        to { background-position: 200% center; }
    }
    
    h2 {
        color: #00F5FF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
        margin-top: 0 !important;
    }
    
    h3 {
        color: #FF00FF !important;
        font-weight: 600 !important;
        text-shadow: 0 0 8px rgba(255, 0, 255, 0.4);
    }
    
    /* Cyberpunk tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(15, 12, 41, 0.6);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 12px;
        border: 1px solid rgba(0, 245, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        color: #00F5FF;
        font-weight: 700;
        font-size: 1.1rem;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 245, 255, 0.1);
        border: 1px solid rgba(0, 245, 255, 0.5);
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.2), rgba(255, 0, 255, 0.2));
        border: 1px solid #00F5FF;
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.6), inset 0 0 15px rgba(255, 0, 255, 0.3);
        color: white !important;
    }
    
    /* Futuristic textarea */
    textarea {
        background: rgba(15, 12, 41, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(0, 245, 255, 0.3) !important;
        border-radius: 15px !important;
        color: #E0E0E0 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }
    
    textarea:focus {
        border: 2px solid #00F5FF !important;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.5) !important;
        background: rgba(15, 12, 41, 0.9) !important;
    }
    
    textarea::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }
    
    /* Neon buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.2), rgba(255, 0, 255, 0.2));
        backdrop-filter: blur(15px);
        color: white;
        border: 2px solid #00F5FF;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.6), inset 0 0 20px rgba(255, 0, 255, 0.2);
        transition: all 0.4s ease;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 0 50px rgba(0, 245, 255, 0.9), 0 0 100px rgba(255, 0, 255, 0.5);
        border-color: #FF00FF;
        background: linear-gradient(135deg, rgba(255, 0, 255, 0.3), rgba(0, 245, 255, 0.3));
    }
    
    /* Holographic metrics */
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #00F5FF, #FF00FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 15px rgba(0, 245, 255, 0.7));
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.7) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Glass expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(0, 245, 255, 0.2);
        font-weight: 600;
        color: #00F5FF !important;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(0, 245, 255, 0.1);
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.3);
    }
    
    /* Cyber divider */
    hr {
        margin: 2.5rem 0 !important;
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(0, 245, 255, 0.5) 20%, 
            rgba(255, 0, 255, 0.5) 50%, 
            rgba(0, 245, 255, 0.5) 80%, 
            transparent
        ) !important;
        box-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }
    
    /* Alert messages */
    .stSuccess, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 12px !important;
        border-left: 4px solid #00F5FF !important;
        padding: 1rem !important;
    }
    
    /* Input fields */
    input {
        background: rgba(15, 12, 41, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(0, 245, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #E0E0E0 !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus {
        border-color: #00F5FF !important;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.5) !important;
    }
    
    /* Floating particles background effect */
    .main::before {
        content: '';
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(0, 245, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(0, 245, 255, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Text colors */
    p, span, label {
        color: rgba(255, 255, 255, 0.85) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00F5FF !important;
        filter: drop-shadow(0 0 10px rgba(0, 245, 255, 0.8));
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI CV - Internship Matcher",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# --- HEADER ---
st.markdown("""
<div style='text-align: center; padding: 3rem 0 2rem 0;'>
    <h1>⚡ Internship & CV MATCHER</h1>
    <p style='color: rgba(255, 255, 255, 0.7); font-size: 1.3rem; font-weight: 300; letter-spacing: 3px;'>
        POWERED BY GEMINI AI • NEURAL NETWORK MATCHING
    </p>
</div>
""", unsafe_allow_html=True)

try:
    db = init_firebase()
    gemini_model, embedding_model = init_gemini()
except Exception as e:
    st.error("⚠️ System initialization failed. Check your credentials.")
    st.stop()

# --- HELPER FUNCTIONS (unchanged) ---
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
    You are a senior Human Resources (HR) specialist...
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
        st.error(f"Embedding error: {e}")
        return None

# --- MAIN UI ---
tab1, tab2 = st.tabs(["🎯 NEURAL MATCHER", "⚡ ADD JOB POSTING"])

# --- TAB 1: CV MATCHER ---
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='margin-bottom: 1rem;'>🧠 NEURAL NETWORK CV ANALYSIS</h2>
            <p style='color: rgba(255,255,255,0.7); font-size: 1.1rem; line-height: 1.8;'>
                Our AI scans <span style='color: #00F5FF; font-weight: 700;'>1000+ job postings</span> 
                in <span style='color: #FF00FF; font-weight: 700;'>milliseconds</span> to find your perfect match
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        cv_text = st.text_area(
            "CV_INPUT", 
            height=320, 
            label_visibility="collapsed",
            placeholder=">>> PASTE YOUR COMPLETE CV HERE\n\n[EDUCATION] [EXPERIENCE] [SKILLS] [PROJECTS]..."
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("⚡ SCAN", type="primary", use_container_width=True)
    
    if analyze_btn:
        if cv_text:
            with st.spinner("🔮 Quantum processors analyzing your profile..."):
                all_jobs = get_job_postings_with_vectors()
                
                if not all_jobs:
                    st.warning("⚠️ Database empty. Add jobs in 'ADD JOB POSTING' tab.")
                    st.stop()
                
                cv_vector = get_embedding(cv_text)
                
                if cv_vector:
                    job_vectors = np.array([job['vector'] for job in all_jobs])
                    cv_vector_np = np.array(cv_vector)
                    similarities = np.dot(job_vectors, cv_vector_np)
                    
                    top_indices = np.argsort(similarities)[-3:][::-1]

                    st.success("✅ Neural scan complete! Top matches identified.")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style='text-align: center; padding: 1.5rem; 
                                background: rgba(0, 245, 255, 0.1); 
                                backdrop-filter: blur(15px);
                                border: 2px solid rgba(0, 245, 255, 0.3);
                                border-radius: 15px; 
                                margin-bottom: 2.5rem;
                                box-shadow: 0 0 30px rgba(0, 245, 255, 0.3);'>
                        <h2 style='margin: 0; color: white; text-shadow: 0 0 20px rgba(0, 245, 255, 0.8);'>
                            🎯 TOP 3 MATCHES
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, index in enumerate(top_indices):
                        matched_job = all_jobs[index]
                        rank = i + 1
                        
                        analysis_text, score = get_gemini_analysis(cv_text, matched_job['description'])
                        
                        # Score-based styling
                        if score and score >= 80:
                            icon = "🔥"
                            color = "#00F5FF"
                            glow = "rgba(0, 245, 255, 0.6)"
                        elif score and score >= 60:
                            icon = "⚡"
                            color = "#FF00FF"
                            glow = "rgba(255, 0, 255, 0.6)"
                        else:
                            icon = "💫"
                            color = "#7B68EE"
                            glow = "rgba(123, 104, 238, 0.6)"
                        
                        with st.container(border=True):
                            st.markdown(f"""
                            <div style='margin-bottom: 1.5rem;'>
                                <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
                                    <div style='background: {color}; 
                                                color: black; 
                                                padding: 0.6rem 1.2rem; 
                                                border-radius: 25px; 
                                                font-weight: 900; 
                                                font-size: 1.1rem;
                                                box-shadow: 0 0 20px {glow};'>
                                        {icon} RANK #{rank}
                                    </div>
                                </div>
                                <h3 style='margin: 0; color: white; font-size: 1.5rem;'>
                                    {matched_job['title']}
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col_a, col_b = st.columns([0.3, 0.7])
                            
                            with col_a:
                                st.metric(
                                    label="COMPATIBILITY",
                                    value=f"{score}%" if score else "N/A",
                                    help="AI Neural Network Score"
                                )
                            
                            with col_b:
                                with st.expander("🔬 DETAILED ANALYSIS", expanded=False):
                                    st.markdown(analysis_text)
                        
                        if i < len(top_indices) - 1:
                            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ CV input required to initiate scan")

# --- TAB 2: ADD JOB ---
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2>⚡ REGISTER NEW JOB POSTING</h2>
            <p style='color: rgba(255,255,255,0.7); font-size: 1.1rem;'>
                AI will auto-generate semantic embeddings for intelligent matching
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("job_form", clear_on_submit=True):
            st.markdown("**📍 JOB TITLE**")
            job_title = st.text_input(
                "title", 
                label_visibility="collapsed",
                placeholder="e.g., Senior Full-Stack Engineer | AI Research Scientist"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**📋 FULL JOB DESCRIPTION**")
            job_description = st.text_area(
                "desc", 
                height=320, 
                label_visibility="collapsed",
                placeholder=">>> PASTE COMPLETE JOB DESCRIPTION\n\n[RESPONSIBILITIES] [REQUIREMENTS] [QUALIFICATIONS]..."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit = st.form_submit_button("💾 SAVE & GENERATE EMBEDDINGS", use_container_width=True)
            
            if submit:
                if job_title and job_description:
                    with st.spinner("🧬 Generating neural embeddings..."):
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
                            st.success(f"✅ '{job_title}' registered in neural database!")
                            st.balloons()
                            st.cache_data.clear()
                        except Exception as e:
                            st.error(f"❌ Database error: {e}")
                    else:
                        st.error("❌ Embedding generation failed")
                else:
                    st.warning("⚠️ Both fields required")

# --- FOOTER ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; 
            padding: 2rem; 
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            border: 1px solid rgba(0, 245, 255, 0.2);
            margin-top: 3rem;'>
    <p style='color: rgba(255,255,255,0.8); font-size: 1rem; margin: 0;'>
        POWERED BY <span style='color: #00F5FF; font-weight: 700;'>GOOGLE GEMINI AI</span> 
        × <span style='color: #FF00FF; font-weight: 700;'>FIREBASE QUANTUM DB</span>
    </p>
    <p style='color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;'>
        QUANTUM EDITION v3.0 | NEURAL NETWORK ARCHITECTURE
    </p>
</div>
""", unsafe_allow_html=True)
