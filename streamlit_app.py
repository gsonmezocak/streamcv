import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, auth # auth'u buraya ekledik
import json
import numpy as np
import re
import pyrebase 

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖",
    layout="wide"
)

# --- 1. FIREBASE ADMIN BAĞLANTISI (Veritabanı için) ---
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
        st.error(f"🔥 FİREBASE ADMİN HATASI: {e}")
        st.stop()
    return firestore.client()

# --- 2. FIREBASE AUTH BAĞLANTISI (Login için) ---
@st.cache_resource
def init_firebase_auth():
    try:
        firebase_config = {
            "apiKey": st.secrets["FIREBASE_WEB_API_KEY"],
            "authDomain": f"{st.secrets['firebase_credentials']['project_id']}.firebaseapp.com",
            "projectId": st.secrets['firebase_credentials']['project_id'],
            "storageBucket": f"{st.secrets['firebase_credentials']['project_id']}.appspot.com",
            "databaseURL": f"https.://{st.secrets['firebase_credentials']['project_id']}-default-rtdb.firebaseio.com",
        }
        firebase = pyrebase.initialize_app(firebase_config)
        return firebase.auth()
    except Exception as e:
        st.error(f"🔥 FİREBASE AUTH HATASI: {e}")
        st.stop()

# --- 3. GEMINI AI BAĞLANTISI ---
@st.cache_resource
def init_gemini():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        analysis_model = genai.GenerativeModel('models/gemini-flash-latest')
        embedding_model = genai.GenerativeModel('models/text-embedding-004')
        return analysis_model, embedding_model
    except Exception as e:
        st.error(f"💎 GEMİNİ BAĞLATMA HATASI: {e}")
        st.stop()

# --- UYGULAMA BAŞLANGICI ---
try:
    db = init_firebase_admin()
    auth_client = init_firebase_auth()
    gemini_model, embedding_model = init_gemini()
except Exception as e:
    st.error("Uygulama başlatılırken kritik bir hata oluştu.")
    st.stop()

# --- OTURUM YÖNETİMİ (Session State) ---
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = None
if 'user_token' not in st.session_state:
    st.session_state['user_token'] = None

# --- YARDIMCI FONKSİYONLAR ---

@st.cache_data(ttl=300) # 5 dakika önbellek
def get_platform_stats():
    """
    Dashboard'da gösterilecek temel istatistikleri çeker.
    """
    try:
        job_docs = db.collection("job_postings").stream()
        total_jobs = sum(1 for _ in job_docs)
        
        profile_docs = db.collection("user_profiles").stream()
        total_profiles = sum(1 for _ in profile_docs)
        
        return total_jobs, total_profiles
    except Exception as e:
        st.error(f"İstatistikler çekilirken hata: {e}")
        return 0, 0

@st.cache_data(ttl=3600) # 1 saat önbellek
def get_total_user_count():
    """
    Firebase Authentication'daki toplam kayıtlı kullanıcı sayısını çeker.
    """
    try:
        page = auth.list_users()
        all_users = list(page.iterate_all())
        return len(all_users)
    except Exception as e:
        st.error(f"Toplam kullanıcı sayısı çekilirken hata: {e}")
        return 0

# (Diğer yardımcı fonksiyonlar...)
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
        st.error(f"İş ilanları çekilirken hata oluştu: {e}")
        return []

def extract_score_from_text(text):
    match = re.search(r"Overall Compatibility Score:.*?(\d{1,3})", text, re.IGNORECASE | re.DOTALL)
    if match: return int(match.group(1))
    return None

def get_gemini_analysis(cv, job_post):
    prompt = f"""
    You are a senior Human Resources (HR) specialist...
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

def get_user_cv(user_id):
    try:
        doc_ref = db.collection("user_profiles").document(user_id).get()
        if doc_ref.exists:
            return doc_ref.to_dict().get("cv_text", "")
        return ""
    except Exception as e:
        st.error(f"Profilinizden CV'niz çekilirken hata oluştu: {e}")
        return ""

# --- ANA UYGULAMA FONKSİYONU (Dashboard'u içeriyor) ---
def main_app():
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🤖 AI CV Matching Platform")
    with col2:
        st.write(f"Logged in as: `{st.session_state['user_email']}`")
        if st.button("Logout", use_container_width=True):
            st.session_state['user_email'] = None
            st.session_state['user_token'] = None
            st.rerun() 
            
    st.markdown("---") 

    # --- Dashboard Metrikleri ---
    with st.spinner("Loading platform stats..."):
        total_jobs, total_profiles = get_platform_stats()
        total_users = get_total_user_count()
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric(label="👥 Total Registered Users", value=total_users)
    with stat_col2:
        st.metric(label="🎯 Total Jobs in Pool", value=total_jobs)
    with stat_col3:
        st.metric(label="👤 Saved CV Profiles", value=total_profiles, help="Number of users who have saved their CV.")

    st.markdown("---")
    
    user_id = auth_client.get_account_info(st.session_state['user_token'])['users'][0]['localId']

    tab1, tab2, tab3 = st.tabs(["🚀 Auto-Matcher", "📝 Add New Job Posting", "👤 My Profile"])

    # (Sekme 1)
    with tab1:
        st.header("Find the Best Jobs for Your CV")
        saved_cv = get_user_cv(user_id)
        
        with st.container(border=True):
            cv_text = st.text_area("📄 Your CV Text:", value=saved_cv, height=350)
        
        if st.button("Find My Matches", type="primary", use_container_width=True):
            if cv_text:
                with st.spinner("Analyzing your CV and searching..."):
                    all_jobs = get_job_postings_with_vectors()
                    if not all_jobs:
                        st.warning("No job postings found. Please add jobs first.")
                        st.stop()
                    
                    cv_vector = get_embedding(cv_text)
                    if cv_vector:
                        job_vectors = np.array([job['vector'] for job in all_jobs])
                        cv_vector_np = np.array(cv_vector)
                        similarities = np.dot(job_vectors, cv_vector_np)
                        top_indices = np.argsort(similarities)[-3:][::-1]

                        st.success(f"Found {len(top_indices)} great matches!")
                        st.markdown("---")
                        
                        for i, index in enumerate(top_indices):
                            matched_job = all_jobs[index]
                            rank = i + 1
                            analysis_text, score = get_gemini_analysis(cv_text, matched_job['description'])
                            
                            with st.container(border=True):
                                col_metric, col_details = st.columns([0.2, 0.8])
                                with col_metric:
                                    st.metric(label=f"Rank #{rank} Match", value=f"{score}%" if score else "N/A")
                                with col_details:
                                    st.subheader(matched_job['title'])
                                    with st.expander("Click to see detailed AI analysis"):
                                        st.markdown(analysis_text)
                            st.divider()
            else:
                st.warning("Please paste your CV text to find matches.")

    # (Sekme 2)
    with tab2:
        st.header("Add a New Job Posting to the Database")
        with st.form("new_job_form", clear_on_submit=True):
            job_title = st.text_input("Job Title")
            job_description = st.text_area("Job Description", height=300)
            submitted = st.form_submit_button("Save Job & Generate Vector")
            
            if submitted:
                if job_title and job_description:
                    with st.spinner("Generating AI fingerprint (vector)..."):
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
                            st.success(f"Successfully added '{job_title}'!")
                            st.cache_data.clear()
                        except Exception as e: st.error(f"Error saving to Firebase: {e}")
                    else: st.error("Could not generate AI fingerprint.")
                else: st.warning("Please fill in both fields.")

    # (Sekfme 3)
    with tab3:
        st.header("My Profile")
        st.markdown("Save your CV here so you don't have to paste it every time.")
        
        current_cv = get_user_cv(user_id)
        
        with st.form("profile_form"):
            new_cv_text = st.text_area("Your CV Text", value=current_cv, height=400)
            submitted = st.form_submit_button("Save CV to Profile")
            
            if submitted:
                try:
                    with st.spinner("Generating AI fingerprint for your CV..."):
                        cv_vector = get_embedding(new_cv_text)
                    
                    if cv_vector:
                        db.collection("user_profiles").document(user_id).set({
                            "email": st.session_state['user_email'],
                            "cv_text": new_cv_text,
                            "cv_vector": cv_vector,
                            "updated_at": firestore.SERVER_TIMESTAMP
                        }, merge=True)
                        st.success("Your CV has been successfully saved to your profile!")
                    else:
                        st.error("Could not generate AI fingerprint for your CV. Not saved.")
                except Exception as e:
                    st.error(f"An error occurred while saving your profile: {e}")

# --- LOGIN SAYFASI FONKSİYONU (GÜNCELLENDİ) ---
def login_page():
    st.title("🤖 AI CV Matching Platform")
    
    st.markdown("Welcome! Log in or sign up to find your perfect job match.")
    st.markdown("---")

    # --- (YENİ) Login Sayfasına İstatistikleri Ekleme ---
    with st.spinner("Loading platform stats..."):
        total_jobs, total_profiles = get_platform_stats()
        total_users = get_total_user_count()
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric(label="👥 Total Registered Users", value=total_users)
    with stat_col2:
        st.metric(label="🎯 Total Jobs in Pool", value=total_jobs)
    with stat_col3:
        st.metric(label="👤 Saved CV Profiles", value=total_profiles)

    st.markdown("---")
    
    # (Devamı)
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
    
    with login_tab:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", type="primary", key="login_button"):
            if email and password:
                try:
                    user = auth_client.sign_in_with_email_and_password(email, password)
                    st.session_state['user_email'] = user['email']
                    st.session_state['user_token'] = user['idToken']
                    st.rerun() 
                except Exception as e:
                    st.warning("Login failed. Please check your email and password.")
            else:
                st.warning("Please enter both email and password.")
                
    with signup_tab:
        st.subheader("Create a New Account")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pass")
        
        if st.button("Sign Up", type="primary", key="signup_button"):
            if new_email and new_password:
                try:
                    user = auth_client.create_user_with_email_and_password(new_email, new_password)
                    st.success("Account created successfully! Please go to the 'Login' tab to log in.")
                except Exception as e:
                    error_message = str(e)
                    if "WEAK_PASSWORD" in error_message:
                        st.warning("Password should be at least 6 characters.")
                    elif "EMAIL_EXISTS" in error_message:
                        st.warning("An account with this email already exists. Please log in.")
                    elif "INVALID_EMAIL" in error_message:
                        st.warning("Please enter a valid email address.")
                    else:
                        st.error("An unknown error occurred during sign up.")
            else:
                st.warning("Please enter both email and password.")

# --- ANA MANTIK ---
if st.session_state['user_email']:
    main_app()
else:
    login_page()
