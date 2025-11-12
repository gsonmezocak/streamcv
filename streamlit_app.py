import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, auth
import json
import numpy as np
import re
import pyrebase 
import time # (YENİ) Analiz süresini göstermek için

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖",
    layout="wide"
)

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
        st.error(f"🔥 FİREBASE ADMİN HATASI: {e}")
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
            "databaseURL": f"https://{st.secrets['firebase_credentials']['project_id']}-default-rtdb.firebaseio.com",
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
        
        # (YENİ) Gemini'yi JSON modunda çalışacak şekilde yapılandır
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        analysis_model = genai.GenerativeModel(
            'models/gemini-flash-latest',
            generation_config=generation_config # JSON modunu uygula
        )
        
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

# --- OTURUM YÖNETİMİ ---
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = None
if 'user_token' not in st.session_state:
    st.session_state['user_token'] = None

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
        st.error(f"İş ilanları çekilirken hata oluştu: {e}")
        return []

def get_gemini_analysis(cv, job_post):
    """
    (GÜNCELLENDİ) Artık metin değil, doğrudan bir Python sözlüğü (dict) döndürüyor.
    """
    
    # (YENİ) Prompt'u JSON formatı istemek için güncelledik
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
        
        # (YENİ) Yanıtı doğrudan JSON olarak yükle
        # Gemini'nin JSON modu bazen '```json\n{...}\n```' ile sarmalayabilir.
        # Her ihtimale karşı temizliyoruz.
        clean_json_text = re.sub(r"^```json\n", "", response.text)
        clean_json_text = re.sub(r"\n```$", "", clean_json_text).strip()
        
        analysis_data = json.loads(clean_json_text)
        return analysis_data
        
    except json.JSONDecodeError:
        st.error(f"AI analizi sırasında JSON hatası oluştu. AI'ın ham yanıtı: {response.text}")
        return None # Hata durumunda None döndür
    except Exception as e:
        st.error(f"AI analizi sırasında bilinmeyen bir hata oluştu: {e}")
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

# --- ANA UYGULAMA FONKSİYONU ---
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

    tab1, tab2, tab3 = st.tabs(["🚀 Auto-Matcher", "📝 Job Management", "👤 My Profile"])

    # --- (TAMAMEN YENİDEN YAZILDI) Sekme 1: Auto-Matcher ---
    with tab1:
        st.header("Find the Best Jobs for Your CV")
        st.markdown("We will use the CV saved in your 'My Profile' tab. If it's empty, please paste your CV below.")
        
        saved_cv = get_user_cv(user_id)
        
        with st.container(border=True):
            cv_text = st.text_area("📄 Your CV Text:", value=saved_cv, height=350)
        
        # (YENİ) Gösterilecek ve taranacak ilan sayıları
        CANDIDATE_POOL_SIZE = 10 # Kaç ilanı analiz edeceğiz (daha yüksek = daha doğru ama yavaş)
        TOP_N_RESULTS = 5       # Kullanıcıya kaçını göstereceğiz
        
        if st.button(f"Find My Top {TOP_N_RESULTS} Matches", type="primary", use_container_width=True):
            if cv_text:
                start_time = time.time() # Zamanlayıcıyı başlat
                
                # --- Adım 1: Hızlı Filtreleme (Vektör Arama) ---
                with st.spinner(f"Step 1/3: Searching all jobs for the top {CANDIDATE_POOL_SIZE} candidates..."):
                    all_jobs = get_job_postings_with_vectors()
                    if not all_jobs:
                        st.warning("No job postings found. Please add jobs first.")
                        st.stop()
                    
                    cv_vector = get_embedding(cv_text)
                    if not cv_vector:
                        st.error("Could not generate fingerprint for your CV. Aborting.")
                        st.stop()
                        
                    job_vectors = np.array([job['vector'] for job in all_jobs])
                    cv_vector_np = np.array(cv_vector)
                    similarities = np.dot(job_vectors, cv_vector_np)
                    
                    # Havuz boyutunu, toplam ilan sayısından az olacak şekilde ayarla
                    pool_size = min(len(all_jobs), CANDIDATE_POOL_SIZE)
                    top_candidate_indices = np.argsort(similarities)[-pool_size:][::-1]

                # --- Adım 2: Detaylı Analiz (AI Çağrıları) ---
                analysis_results = []
                progress_bar = st.progress(0, text=f"Step 2/3: Analyzing {pool_size} candidates...")

                for i, index in enumerate(top_candidate_indices):
                    matched_job = all_jobs[index]
                    
                    # AI'ı çağır ve JSON sonucunu al
                    analysis_data = get_gemini_analysis(cv_text, matched_job['description'])
                    
                    if analysis_data and analysis_data.get("score") is not None:
                        analysis_results.append({
                            "job": matched_job,
                            "data": analysis_data,
                            "score": int(analysis_data.get("score", 0)) # Skoru al
                        })
                    
                    progress_bar.progress((i + 1) / pool_size, text=f"Step 2/3: Analyzing '{matched_job['title'][:30]}...'")
                
                progress_bar.empty()

                # --- Adım 3: Yeniden Sırala ve Göster ---
                with st.spinner(f"Step 3/3: Ranking results and showing the Top {TOP_N_RESULTS}..."):
                    if not analysis_results:
                        st.error("AI analysis failed for all candidates. Please try again.")
                        st.stop()

                    # (YENİ) Listeyi, metin benzerliğine göre değil, AI SKORUNA göre sırala
                    sorted_results = sorted(analysis_results, key=lambda x: x["score"], reverse=True)
                    
                    end_time = time.time()
                    st.success(f"Done! Found and ranked your Top {TOP_N_RESULTS} matches in {end_time - start_time:.2f} seconds.")
                    st.markdown("---")

                    # Sadece en iyi 5'i göster
                    for i, result in enumerate(sorted_results[:TOP_N_RESULTS]):
                        rank = i + 1
                        job_title = result["job"]["title"]
                        score = result["score"]
                        analysis_data = result["data"]
                        
                        with st.container(border=True):
                            col_metric, col_details = st.columns([0.2, 0.8])
                            
                            with col_metric:
                                # Skoru yazdır
                                st.metric(label=f"Rank #{rank} Match", value=f"{score}%")
                            
                            with col_details:
                                st.subheader(job_title)
                                with st.expander("Click to see detailed AI analysis"):
                                    # Sonuçları JSON'dan manuel olarak yazdır
                                    st.subheader("Summary")
                                    st.write(analysis_data.get("summary", "N/A"))
                                    
                                    st.subheader("Strengths (Pros)")
                                    pros = analysis_data.get("pros", [])
                                    if pros:
                                        for pro in pros: st.markdown(f"* {pro}")
                                    else:
                                        st.write("N/A")
                                        
                                    st.subheader("Weaknesses (Cons)")
                                    cons = analysis_data.get("cons", [])
                                    if cons:
                                        for con in cons: st.markdown(f"* {con}")
                                    else:
                                        st.write("N/A")
                        st.divider()
            else:
                st.warning("Please paste your CV text to find matches.")

    # (Sekme 2: İlan Ekleme. Değişiklik yok)
    with tab2:
        st.header("Job Management")
        
        with st.form("new_job_form", clear_on_submit=True):
            st.subheader("Add a Single Job Posting")
            job_title = st.text_input("Job Title")
