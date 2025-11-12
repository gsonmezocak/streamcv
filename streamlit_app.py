import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
import json
import numpy as np # (YENİ) Matematiksel hesaplama için

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖",
    layout="wide"
)

# --- 1. FIREBASE BAĞLANTISI ---
@st.cache_resource
def init_firebase():
    try:
        creds_dict = dict(st.secrets["firebase_credentials"])
        creds_dict["private_key"] = creds_dict["private_key"].replace(r'\n', '\n')
        creds = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(creds)
    except ValueError:
        pass # Uygulama zaten başlatılmış
    except Exception as e:
        st.error(f"🔥 FİREBASE BAŞLATMA HATASI: {e}")
        st.stop()
    return firestore.client()

# --- 2. GEMINI AI BAĞLANTISI ---
@st.cache_resource
def init_gemini():
    """
    (YENİ) Artık İKİ model başlatıyoruz:
    1. 'flash' -> Analiz için
    2. 'embedding' -> Parmak izi/Vektör için
    """
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # Model A: Analiz Modeli
        analysis_model = genai.GenerativeModel('models/gemini-flash-latest')
        
        # Model B: Parmak İzi (Embedding) Modeli
        embedding_model = genai.GenerativeModel('models/text-embedding-004') # (YENİ)
        
        return analysis_model, embedding_model
    except Exception as e:
        st.error(f"💎 GEMİNİ BAŞLATMA HATASI: {e}")
        st.stop()

# --- UYGULAMA BAŞLANGICI ---
st.title("🤖 AI CV Matching Platform (v2 - Vector Search)")

try:
    db = init_firebase()
    gemini_model, embedding_model = init_gemini() # (YENİ) Artık 2 model alıyoruz
except Exception as e:
    st.error("Uygulama başlatılırken kritik bir hata oluştu. Lütfen 'Secrets' ayarlarınızı kontrol edin.")
    st.stop()


# --- YARDIMCI FONKSİYONLAR ---

@st.cache_data(ttl=300) 
def get_job_postings_with_vectors(): # (YENİ İSİM)
    """
    Firestore'dan ilanları ve 'vector' (parmak izi) alanlarını çeker.
    """
    jobs = []
    try:
        # Sadece 'vector' alanı olan ilanları getir (parmak izi olmayanları atla)
        docs = db.collection("job_postings").where("vector", "!=", None).stream()
        for doc in docs:
            job_data = doc.to_dict()
            jobs.append({
                "id": doc.id,
                "title": job_data.get("title", "No Title"),
                "description": job_data.get("description", "No Description"),
                "vector": job_data.get("vector") # (YENİ) Vektörü de al
            })
        return jobs
    except Exception as e:
        st.error(f"İş ilanları çekilirken hata oluştu: {e}")
        return []

def get_gemini_analysis(cv, job_post):
    """
    Model A'ya (flash) analiz prompt'unu gönderir. (Bu fonksiyon değişmedi)
    """
    prompt = f"""
    You are a senior Human Resources (HR) specialist... (Prompt metni aynı)
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
        return response.text
    except Exception as e:
        return f"An error occurred during analysis: {e}"

def get_embedding(text):
    """
    (YENİ) Model B'ye (embedding) bir metin gönderir ve parmak izini (vektör) alır.
    """
    try:
        result = genai.embed_content(
            model="models/text-embedding-004", # Hangi modelin kullanılacağı
            content=text,
            task_type="RETRIEVAL_DOCUMENT" # Görev tipi: Belge arama
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Metnin 'parmak izi' alınırken hata oluştu: {e}")
        return None

# --- ANA UYGULAMA ARAYÜZÜ ---

tab1, tab2 = st.tabs(["🚀 Auto-Matcher (Find Jobs for Me)", "📝 Add New Job Posting"])

# --- Sekme 1: OTOMATİK CV EŞLEŞTİRİCİ (YENİDEN YAZILDI) ---
with tab1:
    st.header("Find the Best Jobs for Your CV")
    st.markdown("Paste your CV below, and our AI will search our entire database to find the top 3 most compatible job postings for you.")
    
    with st.container(border=True):
        cv_text = st.text_area("📄 Paste your full CV text here:", height=350, label_visibility="collapsed")
    
    if st.button("Find My Matches", type="primary", use_container_width=True):
        if cv_text:
            with st.spinner("Analyzing your CV and searching thousands of jobs..."):
                # 1. Tüm ilanları ve vektörlerini veritabanından çek
                all_jobs = get_job_postings_with_vectors()
                
                if not all_jobs:
                    st.warning("No job postings with 'vectors' found in database. Please add jobs in the 'Add New Job' tab first.")
                    st.stop()
                
                # 2. CV'nin parmak izini (vektörünü) al
                cv_vector = get_embedding(cv_text)
                
                if cv_vector:
                    # 3. Matematik: CV vektörü ile tüm ilan vektörleri arasındaki benzerliği hesapla
                    job_vectors = np.array([job['vector'] for job in all_jobs])
                    cv_vector_np = np.array(cv_vector)
                    
                    # 'Dot product' (iç çarpım) en hızlı benzerlik ölçümüdür
                    similarities = np.dot(job_vectors, cv_vector_np)
                    
                    # 4. En iyi 3 eşleşmenin indekslerini bul
                    top_3_indices = np.argsort(similarities)[-3:][::-1] # En yüksek 3 skoru al
                    
                    st.success(f"Found {len(top_3_indices)} great matches for you! Analyzing them now...")
                    
                    # 5. Sadece en iyi 3 ilan için detaylı analiz yap
                    for index in top_3_indices:
                        matched_job = all_jobs[index]
                        st.subheader(f"Rank #{list(top_3_indices).index(index) + 1}: {matched_job['title']}")
                        
                        # Model A'yı (flash) çağır
                        analysis_result = get_gemini_analysis(cv_text, matched_job['description'])
                        
                        with st.expander("Click to see detailed analysis"):
                            st.markdown(analysis_result)
        else:
            st.warning("Please paste your CV text to find matches.")

# --- Sekme 2: YENİ İLAN EKLEME (GÜNCELLENDİ) ---
with tab2:
    st.header("Add a New Job Posting to the Database")
    st.markdown("When you save a job, the AI will automatically generate its 'semantic fingerprint' (vector) and save it for future matching.")
    
    with st.form("new_job_form", clear_on_submit=True):
        job_title = st.text_input("Job Title")
        job_description = st.text_area("Job Description (Paste the full text)", height=300)
        
        submitted = st.form_submit_button("Save Job & Generate Vector")
        
        if submitted:
            if job_title and job_description:
                with st.spinner("Generating AI fingerprint (vector) for this job..."):
                    # 1. (YENİ) İlanın parmak izini al
                    job_vector = get_embedding(f"Title: {job_title}\n\nDescription: {job_description}")
                
                if job_vector:
                    # 2. (YENİ) Vektör ile birlikte Firebase'e kaydet
                    try:
                        doc_ref = db.collection("job_postings").document()
                        doc_ref.set({
                            "title": job_title,
                            "description": job_description,
                            "created_at": firestore.SERVER_TIMESTAMP,
                            "vector": job_vector # (YENİ) Vektörü buraya ekle
                        })
                        st.success(f"Successfully added '{job_title}' with its AI fingerprint!")
                        st.cache_data.clear() # Cache'i temizle
                    except Exception as e:
                        st.error(f"An error occurred while saving to Firebase: {e}")
                else:
                    st.error("Could not generate AI fingerprint. Job not saved.")
            else:
                st.warning("Please fill in both the Job Title and Job Description.")
