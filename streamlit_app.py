import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, auth
import json
import numpy as np
import re
import pyrebase 
import time
import concurrent.futures
from datetime import datetime
import pandas as pd # (YENİ) Toplu yükleme için eklendi
import io # (YENİ) Toplu yükleme için eklendi

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN PROFESYONEL TASARIM ---
def load_custom_css():
    st.markdown("""
        <style>
        /* Import Modern Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main Background */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem 1rem;
        }
        
        [data-testid="stSidebar"] h1 {
            color: white;
            font-weight: 700;
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdown"] {
            color: rgba(255, 255, 255, 0.8);
        }
        
        /* Sidebar Metrics */
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        [data-testid="stSidebar"] [data-testid="stMetric"]:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }
        
        [data-testid="stSidebar"] [data-testid="stMetric"] label {
            color: rgba(255, 255, 255, 0.9) !important;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: white !important;
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        /* Logout Button */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background: rgba(239, 68, 68, 0.2) !important;
            color: white !important;
            border: 1px solid rgba(239, 68, 68, 0.5) !important;
            transition: all 0.3s ease;
        }
        
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(239, 68, 68, 0.3) !important;
            transform: scale(1.02);
        }
        
        /* Main Content Container */
        .block-container {
            padding: 2rem 3rem;
            max-width: 1400px;
        }
        
        /* Main Title */
        h1 {
            color: white;
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.95);
            padding: 0.5rem;
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: black; /* Düzeltildi */
            font-weight: 500;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(102, 126, 234, 0.1);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }
        
        /* Cards/Containers */
        [data-testid="stVerticalBlock"] > div {
            background: white;
            color: black; /* Düzeltildi */
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        [data-testid="stVerticalBlock"] > div:hover {
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        
        /* Headers */
        h2 {
            color: black; /* Düzeltildi */
            font-weight: 600;
            font-size: 1.75rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }
        
        h3 {
            color: black; /* Düzeltildi */
            font-weight: 600;
            font-size: 1.25rem;
        }
        
        /* Text Areas */
        textarea {
            border: 2px solid #e5e7eb !important;
            border-radius: 12px !important;
            font-size: 0.95rem !important;
            transition: all 0.3s ease !important;
        }
        
        textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Input Fields */
        input {
            border: 2px solid #e5e7eb !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Primary Buttons */
        button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        }
        
        button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
        }
        
        /* Secondary Buttons */
        button[kind="secondary"] {
            background: white !important;
            color: #667eea !important;
            border: 2px solid #667eea !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }
        
        button[kind="secondary"]:hover {
            background: #667eea !important;
            color: white !important;
        }
        
        /* Match Score Cards */
        .match-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 5px solid #667eea;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }
        
        .match-card:hover {
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
            transform: translateX(5px);
        }
        
        /* Score Badge */
        .score-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.2rem;
        }
        
        /* Success Messages */
        .stSuccess {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            border: none !important;
        }
        
        /* Warning Messages */
        .stWarning {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            border: none !important;
        }
        
        /* Error Messages */
        .stError {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            border: none !important;
        }
        
        /* Info Messages */
        .stInfo {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            border: none !important;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            border-radius: 10px !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(102, 126, 234, 0.05) !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Divider */
        hr {
            margin: 2rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
        
        /* Login Page Metrics */
        .login-metrics {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        /* Login Page Title */
        .login-title {
            color: white;
            text-align: center;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        .login-subtitle {
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: rgba(102, 126, 234, 0.05);
            border: 2px dashed #667eea;
            border-radius: 12px;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            background: rgba(102, 126, 234, 0.1);
            border-color: #764ba2;
        }
        
        /* Form */
        [data-testid="stForm"] {
            background: rgba(102, 126, 234, 0.02);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(102, 126, 234, 0.1);
        }
        
        /* Metric Delta */
        [data-testid="stMetricDelta"] {
            font-weight: 600;
        }
        
        /* (YENİ) CSS Sınıfları (Login Kartı) */
        .login-card {
            background: white;
            border-radius: 24px;
            padding: 3rem 4rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            margin: 3rem auto;
            text-align: center;
        }
        .login-card h2 { color: #333; font-size: 1.8rem; font-weight: 600; }
        .login-card p { color: #666; font-size: 1rem; margin-bottom: 2rem; }
        .toggle-container { display: flex; background-color: #f0f2f6; border-radius: 12px; padding: 5px; margin-bottom: 2rem; }
        .toggle-btn { flex: 1; padding: 0.75rem 0.5rem; border: none; border-radius: 9px; background-color: transparent; color: #666; font-size: 1rem; font-weight: 500; cursor: pointer; transition: all 0.2s ease-in-out; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .toggle-btn.active { background: white; color: #764ba2; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08); }
        .login-card input[type="text"], .login-card input[type="password"] { border: 1px solid #e0e0e0 !important; border-radius: 12px !important; padding: 0.8rem 1rem !important; font-size: 1rem !important; background-color: #f7f7f7 !important; }
        .login-card input[type="text"]:focus, .login-card input[type="password"]:focus { border-color: #764ba2 !important; box-shadow: 0 0 0 3px rgba(118, 75, 162, 0.1) !important; background-color: white !important; }
        .login-card [data-testid="stTextInput"] label { color: #333 !important; font-weight: 500 !important; margin-bottom: 0.5rem !important; display: block; text-align: left; }
        .login-card button[kind="primary"] { margin-top: 1.5rem; width: 100%; padding: 1rem 2rem !important; font-size: 1.1rem !important; border-radius: 14px !important; }
        .signup-link-container { margin-top: 2rem; font-size: 0.95rem; color: #666; }
        .signup-link { color: #764ba2 !important; font-weight: 600 !important; text-decoration: none !important; }
        .signup-link:hover { text-decoration: underline !important; }
        
        </style>
    """, unsafe_allow_html=True)

# --- 1. FIREBASE ADMIN BAĞLANTISI ---
@st.cache_resource
def init_firebase_admin():
    try:
        creds_dict = dict(st.secrets["firebase_credentials"])
        creds_dict["private_key"] = creds_dict["private_key"].replace(r'\n', '\n')
        creds = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(creds)
    except ValueError:
        pass 
    except Exception as e:
        st.error(f"🔥 Firebase Admin Error: {e}")
        st.stop()
    return firestore.client()

# --- 2. FIREBASE AUTH BAĞLANTISI ---
@st.cache_resource
def init_firebase_auth():
    try:
        firebase_config = {
            "apiKey": st.secrets["FIREBASE_WEB_API_KEY"],
            "authDomain": f"{st.secrets['firebase_credentials']['project_id']}.firebaseapp.com",
            "projectId": st.secrets['firebase_credentials']['project_id'],
            "storageBucket": f"{st.secrets['firebase_credentials']['project_id']}.appspot.com",
            "databaseURL": f"https{st.secrets['firebase_credentials']['project_id']}-default-rtdb.firebaseio.com",
        }
        firebase = pyrebase.initialize_app(firebase_config)
        return firebase.auth()
    except Exception as e:
        st.error(f"🔥 Firebase Auth Error: {e}")
        st.stop()

# --- 3. GEMINI AI BAĞLANTISI ---
@st.cache_resource
def init_gemini():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        analysis_model = genai.GenerativeModel('models/gemini-flash-latest', generation_config=generation_config)
        embedding_model = genai.GenerativeModel('models/text-embedding-004')
        return analysis_model, embedding_model
    except Exception as e:
        st.error(f"💎 Gemini Connection Error: {e}")
        st.stop()

# --- UYGULAMA BAŞLANGICI ---
try:
    db = init_firebase_admin()
    auth_client = init_firebase_auth()
    gemini_model, embedding_model = init_gemini()
except Exception as e:
    st.error("Critical error during application startup.")
    st.stop()

# --- OTURUM YÖNETİMİ ---
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = None
if 'user_token' not in st.session_state:
    st.session_state['user_token'] = None
if 'user_role' not in st.session_state: # (YENİ) Kullanıcı rolü
    st.session_state['user_role'] = 'job_seeker' # Varsayılan
if 'show_signup' not in st.session_state: # (YENİ) Kayıt formu
    st.session_state['show_signup'] = False

# --- YARDIMCI FONKSİYONLAR ---
@st.cache_data(ttl=300) 
def get_platform_stats():
    try:
        job_docs = db.collection("job_postings").stream()
        total_jobs = sum(1 for _ in job_docs)
        profile_docs = db.collection("user_profiles").stream()
        total_profiles = sum(1 for _ in profile_docs)
        return total_jobs, total_profiles
    except Exception as e: 
        return 0, 0

@st.cache_data(ttl=3600) 
def get_total_user_count():
    try:
        page = auth.list_users()
        all_users = list(page.iterate_all())
        return len(all_users)
    except Exception as e: 
        return 0

@st.cache_data(ttl=300) 
def get_job_postings_with_vectors():
    jobs = []
    try:
        docs = db.collection("job_postings").stream()
        for doc in docs:
            job_data = doc.to_dict()
            if 'vector' in job_data: 
                jobs.append({
                    "id": doc.id,
                    "title": job_data.get("title", "No Title"),
                    "description": job_data.get("description", "No Description"),
                    "vector": job_data.get("vector")
                })
        return jobs
    except Exception as e:
        st.error(f"Error fetching job postings: {e}")
        return []

def get_gemini_analysis(cv, job_post):
    prompt = f"""
    You are a senior Human Resources (HR) specialist.
    Analyze the following CV and JOB POSTING.
    Your response MUST be a valid JSON object with the following exact structure:
    {{
        "score": <number from 0-100>,
        "pros": ["<strength 1>", "<strength 2>", "<strength 3>"],
        "cons": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],
        "summary": "<A 2-3 sentence evaluation summary>"
    }}
    ---[CV TEXT]----
    {cv}
    -----------------
    ---[JOB POSTING TEXT]---
    {job_post}
    -----------------
    """
    try:
        response = gemini_model.generate_content(prompt)
        clean_json_text = re.sub(r"^```json\n", "", response.text)
        clean_json_text = re.sub(r"\n```$", "", clean_json_text).strip()
        analysis_data = json.loads(clean_json_text)
        return analysis_data
    except Exception as e:
        print(f"JSON Parse Error: {e}")
        print(f"AI Raw Response: {response.text}")
        return None 

def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def get_user_cv(user_id):
    try:
        doc_ref = db.collection("user_profiles").document(user_id).get()
        if doc_ref.exists:
            return doc_ref.to_dict().get("cv_text", "")
        return ""
    except Exception as e:
        st.error(f"Error fetching CV from profile: {e}")
        return ""

def logout_callback():
    st.session_state['user_email'] = None
    st.session_state['user_token'] = None
    st.session_state['show_signup'] = False # (YENİ) Çıkışta kayıt formunu gizle
    st.rerun()

# --- ANA UYGULAMA FONKSİYONU ---
def main_app():
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"### 👋 Welcome Back!")
        st.markdown(f"**{st.session_state['user_email'].split('@')[0].capitalize()}**")
        st.markdown(f"`{st.session_state['user_email']}`")
        st.button("🚪 Logout", use_container_width=True, on_click=logout_callback, key="logout_btn")
        
        st.markdown("---")
        
        st.markdown("### 📊 Platform Statistics")
        with st.spinner("Loading..."):
            total_jobs, total_profiles = get_platform_stats()
            total_users = get_total_user_count()
        
        st.metric(label="👥 Registered Users", value=f"{total_users:,}")
        st.metric(label="💼 Available Jobs", value=f"{total_jobs:,}")
        st.metric(label="📄 Active CVs", value=f"{total_profiles:,}")
        
        st.markdown("---")
        st.markdown("### ⚡ Quick Tips")
        st.info("💡 Keep your CV updated for better matches!")
        st.info("🎯 Check new jobs daily for opportunities!")
    
    # --- Main Title ---
    st.markdown("<h1>🎯 AI-Powered CV Matching</h1>", unsafe_allow_html=True)
    st.markdown("Find your perfect job match with intelligent AI analysis")
    
    user_id = auth_client.get_account_info(st.session_state['user_token'])['users'][0]['localId']

    # (YENİ) Kullanıcı rolüne göre sekmeleri ayarla
    if st.session_state.get('user_role') == 'recruiter':
        st.info("🧑‍💼 Recruiter Mode: Manage your job postings.")
        tab1, tab2 = st.tabs(["💼 Job Management", "👤 My Profile"])
        
        # --- İK Sekme 1: İlan Yönetimi ---
        with tab1:
            st.markdown("## 💼 Job Posting Management")
            st.markdown("Add new job opportunities to the platform")
            
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                with st.container():
                    st.markdown("### ➕ Add Single Job")
                    with st.form("new_job_form", clear_on_submit=True):
                        job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
                        job_description = st.text_area("Job Description", height=200, placeholder="Enter detailed job requirements...")
                        submitted = st.form_submit_button("💾 Save Job", use_container_width=True, type="primary")
                        
                        if submitted:
                            if job_title and job_description:
                                with st.spinner("🤖 Generating AI fingerprint..."):
                                    job_vector = get_embedding(f"Title: {job_title}\n\nDescription: {job_description}")
                                if job_vector:
                                    try:
                                        db.collection("job_postings").document().set({
                                            "title": job_title,
                                            "description": job_description,
                                            "created_at": firestore.SERVER_TIMESTAMP,
                                            "vector": job_vector,
                                            "added_by": st.session_state['user_email']
                                        })
                                        st.success(f"✅ Successfully added '{job_title}'!")
                                        st.cache_data.clear()
                                    except Exception as e: 
                                        st.error(f"❌ Error saving to Firebase: {e}")
                                else: 
                                    st.error("❌ Could not generate AI fingerprint.")
                            else: 
                                st.warning("⚠️ Please fill in both fields.")
            
            with col_right:
                with st.container():
                    st.markdown("### 📦 Bulk Upload")
                    st.markdown("Upload multiple jobs from CSV/Excel file")
                    st.markdown("**Required columns:** `title`, `description`")
                    
                    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], label_visibility="collapsed")
                    
                    if uploaded_file is not None:
                        try:
                            if uploaded_file.name.endswith('.csv'):
                                df = pd.read_csv(uploaded_file)
                            else:
                                df = pd.read_excel(uploaded_file)
                            
                            if 'title' not in df.columns or 'description' not in df.columns:
                                st.error("❌ File must contain 'title' and 'description' columns!")
                            else:
                                st.success(f"✅ Found {len(df)} jobs in file")
                                st.dataframe(df.head(3), use_container_width=True)
                                
                                if st.button("📤 Upload All Jobs", use_container_width=True, type="primary"):
                                    progress_bar = st.progress(0, text="Uploading jobs...")
                                    success_count = 0
                                    
                                    for idx, row in df.iterrows():
                                        try:
                                            job_title = str(row['title'])
                                            job_desc = str(row['description'])
                                            
                                            job_vector = get_embedding(f"Title: {job_title}\n\nDescription: {job_desc}")
                                            
                                            if job_vector:
                                                db.collection("job_postings").document().set({
                                                    "title": job_title,
                                                    "description": job_desc,
                                                    "created_at": firestore.SERVER_TIMESTAMP,
                                                    "vector": job_vector,
                                                    "added_by": st.session_state['user_email']
                                                })
                                                success_count += 1
                                            
                                            progress_bar.progress((idx + 1) / len(df), text=f"Uploading... {idx + 1}/{len(df)}")
                                        except Exception as e:
                                            st.warning(f"⚠️ Skipped job at row {idx + 1}: {e}")
                                    
                                    progress_bar.empty()
                                    st.success(f"✅ Successfully uploaded {success_count}/{len(df)} jobs!")
                                    st.cache_data.clear()
                                    st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error reading file: {e}")
            
            st.markdown("---")
            
            # İK'cının kendi ilanlarını listele
            st.markdown("### 📋 My Current Job Postings")
            jobs = get_job_postings_with_vectors() # (Gelecekte burayı 'added_by' == user_email ile filtrele)
            
            if jobs:
                st.info(f"📊 Total Jobs: **{len(jobs)}**")
                
                for idx, job in enumerate(jobs[:10]):  # Show first 10
                    with st.expander(f"💼 {job['title']}", expanded=False):
                        st.markdown(f"**Description:**")
                        st.write(job['description'][:300] + "..." if len(job['description']) > 300 else job['description'])
                        st.caption(f"Job ID: `{job['id']}`")
                
                if len(jobs) > 10:
                    st.info(f"📌 Showing 10 of {len(jobs)} jobs. All jobs are available for matching.")
            else:
                st.warning("⚠️ No job postings yet. Add some jobs to get started!")

        # --- İK Sekme 2: Profil ---
        with tab2:
            st.markdown("## 👤 Recruiter Profile")
            st.markdown("Manage your company and profile information")
            with st.container(border=True):
                st.info("Recruiter profile page is under construction.")
                st.markdown(f"**Email:** `{st.session_state['user_email']}`")
                st.markdown(f"**User ID:** `{user_id[:8]}...`")

    else: # (YENİ) Rol "job_seeker" (varsayılan) ise
        st.info("💼 Job Seeker Mode: Find your next opportunity.")
        tab1, tab2 = st.tabs(["🚀 Auto-Matcher", "👤 My Profile"])

        # --- Aday Sekme 1: Auto-Matcher ---
        with tab1:
            st.markdown("## 🚀 Intelligent Job Matching")
            st.markdown("Upload your CV and let our AI find the best opportunities for you")
            
            saved_cv = get_user_cv(user_id)
            
            with st.container(border=True):
                st.markdown("### 📄 Your CV")
                cv_text = st.text_area(
                    "Paste your CV text here (or use the one saved in your profile)",
                    value=saved_cv,
                    height=300,
                    placeholder="Enter your complete CV here...",
                    label_visibility="collapsed"
                )
            
            CANDIDATE_POOL_SIZE = 10 
            TOP_N_RESULTS = 5       
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(f"🎯 Find My Top {TOP_N_RESULTS} Matches", type="primary", use_container_width=True):
                    if cv_text:
                        start_time = time.time() 
                        
                        with st.spinner(f"🔍 Scanning {CANDIDATE_POOL_SIZE} job opportunities..."):
                            all_jobs = get_job_postings_with_vectors()
                            if not all_jobs:
                                st.warning("⚠️ No job postings found. Please add jobs first.")
                                st.stop()
                            
                            cv_vector = get_embedding(cv_text)
                            if not cv_vector:
                                st.error("❌ Could not generate CV fingerprint. Please try again.")
                                st.stop()
                                
                            job_vectors = np.array([job['vector'] for job in all_jobs])
                            cv_vector_np = np.array(cv_vector)
                            similarities = np.dot(job_vectors, cv_vector_np)
                            
                            pool_size = min(len(all_jobs), CANDIDATE_POOL_SIZE)
                            top_candidate_indices = np.argsort(similarities)[-pool_size:][::-1]

                        analysis_results = []
                        progress_bar = st.progress(0, text=f"🤖 AI analyzing candidates... 0%")

                        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
                            future_to_job = {}
                            for index in top_candidate_indices:
                                matched_job = all_jobs[index]
                                future = executor.submit(get_gemini_analysis, cv_text, matched_job['description'])
                                future_to_job[future] = matched_job
                            
                            completed_count = 0
                            for future in concurrent.futures.as_completed(future_to_job):
                                matched_job = future_to_job[future]
                                try:
                                    analysis_data = future.result() 
                                    if analysis_data and analysis_data.get("score") is not None:
                                        analysis_results.append({
                                            "job": matched_job,
                                            "data": analysis_data,
                                            "score": int(analysis_data.get("score", 0))
                                        })
                                except Exception as e:
                                    st.error(f"❌ Error analyzing job '{matched_job['title']}': {e}")
                                
                                completed_count += 1
                                percent_complete = completed_count / pool_size
                                progress_bar.progress(percent_complete, text=f"🤖 AI analyzing... {int(percent_complete * 100)}%")
                        
                        progress_bar.empty()

                        if not analysis_results:
                            st.error("❌ AI analysis failed for all candidates. Please try again.")
                            st.stop()

                        sorted_results = sorted(analysis_results, key=lambda x: x["score"], reverse=True)
                        
                        end_time = time.time()
                        st.success(f"✅ Found your top {TOP_N_RESULTS} matches in {end_time - start_time:.2f} seconds!")
                        st.balloons()
                        
                        st.markdown("---")
                        st.markdown(f"## 🏆 Your Top {TOP_N_RESULTS} Job Matches")

                        for i, result in enumerate(sorted_results[:TOP_N_RESULTS]):
                            rank = i + 1
                            job_title = result["job"]["title"]
                            score = result["score"]
                            analysis_data = result["data"]
                            
                            if score >= 80: medal = "🥇"
                            elif score >= 60: medal = "🥈"
                            else: medal = "🥉"
                            
                            st.markdown(f"""
                            <div class="match-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                    <div>
                                        <span style="font-size: 1.5rem;">{medal}</span>
                                        <span style="font-size: 1.3rem; font-weight: 600; color: #1e3c72; margin-left: 0.5rem;">
                                            #{rank} {job_title}
                                        </span>
                                    </div>
                                    <div class="score-badge" style="background: {'#10b981' if score >= 80 else ('#3b82f6' if score >= 60 else '#f59e0b')};">
                                        {score}% Match
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("📊 View Detailed AI Analysis", expanded=(rank == 1)):
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.markdown("### ✅ Strengths")
                                    pros = analysis_data.get("pros", [])
                                    if pros:
                                        for pro in pros: st.markdown(f"✓ {pro}")
                                    else:
                                        st.write("No specific strengths identified")
                                
                                with col_b:
                                    st.markdown("### ⚠️ Areas for Improvement")
                                    cons = analysis_data.get("cons", [])
                                    if cons:
                                        for con in cons: st.markdown(f"• {con}")
                                    else:
                                        st.write("No specific weaknesses identified")
                                
                                st.markdown("---")
                                st.markdown("### 📝 Summary")
                                st.info(analysis_data.get("summary", "No summary available"))
                            
                            if rank < TOP_N_RESULTS:
                                st.markdown("<br>", unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Please paste your CV text to find matches.")

        # --- Aday Sekme 2: Profilim ---
        with tab2:
            st.markdown("## 👤 My Profile")
            st.markdown("Manage your CV and profile information")
            
            current_cv = get_user_cv(user_id)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.container(border=True):
                    st.markdown("### 📄 Your CV")
                    with st.form("profile_form"):
                        new_cv_text = st.text_area(
                            "CV Text", 
                            value=current_cv, 
                            height=400,
                            placeholder="Paste your complete CV here for better job matching...",
                            label_visibility="collapsed"
                        )
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            submitted = st.form_submit_button("💾 Save CV", use_container_width=True, type="primary")
                        
                        if submitted:
                            if new_cv_text.strip():
                                try:
                                    with st.spinner("🤖 Generating AI fingerprint for your CV..."):
                                        cv_vector = get_embedding(new_cv_text)
                                    
                                    if cv_vector:
                                        db.collection("user_profiles").document(user_id).set({
                                            "email": st.session_state['user_email'],
                                            "cv_text": new_cv_text,
                                            "cv_vector": cv_vector,
                                            "updated_at": firestore.SERVER_TIMESTAMP
                                        }, merge=True)
                                        st.success("✅ Your CV has been successfully saved!")
                                        st.balloons()
                                        st.cache_data.clear()
                                    else:
                                        st.error("❌ Could not generate AI fingerprint for your CV.")
                                except Exception as e:
                                    st.error(f"❌ An error occurred while saving: {e}")
                            else:
                                st.warning("⚠️ Please enter your CV text before saving.")
            
            with col2:
                with st.container(border=True):
                    st.markdown("### ℹ️ Profile Info")
                    st.markdown(f"**Email:** `{st.session_state['user_email']}`")
                    st.markdown(f"**User ID:** `{user_id[:8]}...`")
                    
                    if current_cv:
                        st.markdown(f"**CV Length:** {len(current_cv)} characters")
                        st.markdown(f"**Words:** ~{len(current_cv.split())} words")
                        
                        try:
                            doc = db.collection("user_profiles").document(user_id).get()
                            if doc.exists and 'updated_at' in doc.to_dict():
                                updated = doc.to_dict()['updated_at']
                                st.markdown(f"**Last Updated:** {updated.strftime('%Y-%m-%d %H:%M') if updated else 'N/A'}")
                        except:
                            pass
                    else:
                        st.warning("⚠️ No CV saved yet")
                    
                    st.markdown("---")
                    st.markdown("### 💡 Tips")
                    st.info("✓ Include relevant skills")
                    st.info("✓ Highlight experience")
                    st.info("✓ Keep it updated")

# --- (YENİ) LOGIN PAGE (Figma Tasarımı) ---
def login_page():
    # Sayfa arka planını koru
    st.markdown('<style>.stApp {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}</style>', unsafe_allow_html=True)
    
    # "Job Seeker" / "Recruiter" rolünü yönet
    if 'user_role' not in st.session_state:
        st.session_state['user_role'] = 'job_seeker'
    
    # Toggle butonların durumunu güncellemek için callback
    def set_role(role):
        st.session_state['user_role'] = role
        st.session_state['show_signup'] = False # Rol değiştirince formu sıfırla

    # --- ANA KART ---
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        
        # Logo ve başlık
        # st.image("logo.png", width=70) # Eğer 'logo.png' dosyanız varsa bu satırı kullanın
        st.markdown("<h2>TalentMatch</h2>", unsafe_allow_html=True)
        st.markdown("<p>Connect through your digital avatar</p>", unsafe_allow_html=True)

        # Toggle Butonlar
        # Streamlit'in butonlarını CSS ile 'toggle' gibi göster
        toggle_cols = st.columns(2)
        with toggle_cols[0]:
            job_seeker_class = "toggle-btn active" if st.session_state['user_role'] == 'job_seeker' else "toggle-btn"
            if st.button("💼 Job Seeker", use_container_width=True, key="job_seeker_btn_css"):
                set_role('job_seeker')
                st.rerun() # Sadece bu butona stil vermek için yeniden çalıştırmak zorundayız

        with toggle_cols[1]:
            recruiter_class = "toggle-btn active" if st.session_state['user_role'] == 'recruiter' else "toggle-btn"
            if st.button("🧑‍💼 Recruiter", use_container_width=True, key="recruiter_btn_css"):
                set_role('recruiter')
                st.rerun()

        # Streamlit butonlarına özel CSS sınıflarını uygula
        st.markdown(f"""
            <style>
                button[data-testid="stButton"][key="job_seeker_btn_css"] > div {{ {job_seeker_class} }}
                button[data-testid="stButton"][key="recruiter_btn_css"] > div {{ {recruiter_class} }}
                
                /* Streamlit butonunun kendi arka planını ve kenarlığını gizle */
                button[data-testid="stButton"][key="job_seeker_btn_css"],
                button[data-testid="stButton"][key="recruiter_btn_css"] {{
                    background: transparent !important;
                    border: none !important;
                    padding: 0 !important;
                    margin: 0 !important;
                }}
            </style>
        """, unsafe_allow_html=True)
        
        # --- Form Alanı ---
        if st.session_state['show_signup']:
            # --- Kayıt Olma Formu ---
            st.markdown("<h3 style='margin-top: 2rem;'>✨ Create Your Account</h3>", unsafe_allow_html=True)
            new_email = st.text_input("Email", key="signup_email", placeholder="your@email.com")
            new_password = st.text_input("Password", type="password", key="signup_pass", placeholder="min. 6 characters")
            
            if st.button("Create Account", type="primary", key="signup_button_new", use_container_width=True):
                if new_email and new_password:
                    try:
                        user = auth_client.create_user_with_email_and_password(new_email, new_password)
                        st.success("✅ Account created! Please log in to continue.")
                        st.session_state['show_signup'] = False # Login formuna geri dön
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        error_message = str(e)
                        if "WEAK_PASSWORD" in error_message:
                            st.warning("⚠️ Password should be at least 6 characters.")
                        elif "EMAIL_EXISTS" in error_message:
                            st.warning("⚠️ Account already exists. Please log in.")
                        elif "INVALID_EMAIL" in error_message:
                            st.warning("⚠️ Please enter a valid email address.")
                        else:
                            st.error("❌ An error occurred during sign up.")
                else:
                    st.warning("⚠️ Please enter both email and password.")

            # Giriş yap linki
            if st.button("Already have an account? Log in", key="goto_login_btn", use_container_width=True):
                st.session_state['show_signup'] = False
                st.rerun()
                
        else:
            # --- Giriş Formu ---
            st.markdown("<h3 style='margin-top: 2rem;'>🔐 Log In</h3>", unsafe_allow_html=True)
            email = st.text_input("Email", key="login_email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••••")
            
            if st.button("Start Matching", type="primary", key="login_button_new", use_container_width=True):
                if email and password:
                    try:
                        user = auth_client.sign_in_with_email_and_password(email, password)
                        st.session_state['user_email'] = user['email']
                        st.session_state['user_token'] = user['idToken']
                        st.session_state['user_role'] = st.session_state.get('user_role', 'job_seeker') # Seçili rolü kaydet
                        st.rerun() 
                    except Exception as e:
                        st.error("❌ Invalid email or password. Please try again.")
                else:
                    st.warning("⚠️ Please enter both email and password.")

            # Kayıt ol linki
            if st.button("Don't have an account? Sign up", key="goto_signup_btn", use_container_width=True):
                st.session_state['show_signup'] = True
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True) # .login-card div'i kapat

# --- ANA MANTIK ---
load_custom_css()

if st.session_state['user_email']:
    main_app()
else:
    login_page()
