import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pyrebase
import json
import numpy as np
import re
import time
import concurrent.futures
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
import io

# --- 0. SAYFA AYARLARI ---
# HTML dosyalarınızdaki fontları ve iconları ekliyoruz
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        /* Streamlit'in radio butonunu sign_up.html'deki toggle'a benzetme */
        div[role="radiogroup"] {
            background-color: #F0F0F0;
            border-radius: 0.5rem;
            padding: 0.25rem;
            display: flex;
        }
        div[role="radiogroup"] label {
            background-color: transparent;
            color: #6B7280;
            flex-grow: 1;
            text-align: center;
            padding: 0.5rem;
            border-radius: 0.375rem;
            transition: all 0.2s ease-in-out;
        }
        /* Seçili olan radio butonu */
        div[role="radiogroup"] input:checked + div {
            background-color: #FFFFFF;
            color: #111827;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        }
        .material-symbols-outlined {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 48;
            vertical-align: middle;
        }
    </style>
""", unsafe_allow_html=True)


# --- 1. FIREBASE ADMIN BAĞLANTISI ---
@st.cache_resource
def init_firebase_admin():
    try:
        # st.secrets'ten Firebase credentials'ı al
        creds_dict = dict(st.secrets["firebase_credentials"])
        
        # private_key'deki \n karakterlerini düzelt (Streamlit secrets için önemli)
        creds_dict["private_key"] = creds_dict["private_key"].replace(r'\n', '\n')
        
        creds = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(creds)
    except ValueError:
        # Uygulama zaten başlatılmışsa (hot reload) hata verme
        pass  
    except Exception as e:
        st.error(f"🔥 FİREBASE ADMİN HATASI: {e}")
        st.stop()
    return firestore.client()

# --- 2. FIREBASE AUTH BAĞLANTISI ---
@st.cache_resource
def init_firebase_auth():
    try:
        # Gerekli anahtarları st.secrets'ten al
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
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        analysis_model = genai.GenerativeModel('models/gemini-flash-latest', generation_config=generation_config)
        embedding_model = genai.GenerativeModel('models/text-embedding-004')
        return analysis_model, embedding_model
    except Exception as e:
        st.error(f"💎 GEMİNİ BAĞLANTI HATASI: {e}")
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
        # Yanıttan JSON bloğunu temizle (bazen ```json ... ``` ile dönebiliyor)
        clean_json_text = re.sub(r"^```json\n", "", response.text)
        clean_json_text = re.sub(r"\n```$", "", clean_json_text).strip()
        analysis_data = json.loads(clean_json_text)
        return analysis_data
    except Exception as e:
        print(f"JSON Parse Hatası: {e}")
        print(f"AI Ham Yanıtı: {response.text}")
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

def get_user_profile(user_id):
    """(GÜNCELLENDİ) Sadece CV'yi değil, tüm profili çeker."""
    try:
        doc_ref = db.collection("user_profiles").document(user_id).get()
        if doc_ref.exists:
            return doc_ref.to_dict()
        return {} # Boş bir sözlük döndür
    except Exception as e:
        st.error(f"Profiliniz çekilirken hata oluştu: {e}")
        return {}

def parse_cv_file(file_bytes, file_name):
    """(YENİ) Yüklenen PDF veya DOCX dosyasını metne çevirir."""
    text = ""
    try:
        if file_name.endswith('.pdf'):
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
        elif file_name.endswith('.docx'):
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            # Diğer dosya türlerini (örn. .txt) basitçe okumayı dene
            text = file_bytes.decode('utf-8')
            
        return text
    except Exception as e:
        st.error(f"Dosya okunurken hata oluştu: {e}")
        return None

# --- ANA UYGULAMA FONKSİYONU ---
def main_app():
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🤖 AI CV Matching Platform")
    with col2:
        st.write(f"`{st.session_state['user_email']}`")
        if st.button("Logout", use_container_width=True):
            st.session_state['user_email'] = None
            st.session_state['user_token'] = None
            st.rerun()  
            
    st.markdown("---") 

    with st.spinner("Platform istatistikleri yükleniyor..."):
        total_jobs, total_profiles = get_platform_stats()
        total_users = get_total_user_count()
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric(label="👥 Toplam Kayıtlı Kullanıcı", value=total_users)
    with stat_col2:
        st.metric(label="🎯 Toplam İş İlanı", value=total_jobs)
    with stat_col3:
        st.metric(label="👤 Kayıtlı CV Profili", value=total_profiles, help="CV'sini kaydeden kullanıcı sayısı.")

    st.markdown("---")
    
    user_id = auth_client.get_account_info(st.session_state['user_token'])['users'][0]['localId']

    tab1, tab2, tab3 = st.tabs(["🚀 Auto-Matcher", "📝 İlan Yönetimi", "👤 Profilim"])

    # --- (GÜNCELLENDİ) Sekme 1: Auto-Matcher ---
    with tab1:
        st.header("CV'niz için En İyi İşleri Bulun")
        
        # CV'yi profilden çek
        profile = get_user_profile(user_id)
        cv_text = profile.get("cv_text")
        
        if not cv_text:
            st.warning("Henüz kayıtlı bir CV'niz bulunmuyor.")
            st.info("Lütfen önce '👤 Profilim' sekmesine gidin ve CV'nizi yükleyin.")
            st.stop()
            
        st.success("Harika! 'Profilim' sekmesinde kayıtlı olan CV'niz kullanılacak.")
        st.markdown(f"> **Profil Başlığınız:** `{profile.get('headline', 'Belirtilmemiş')}`")
        
        CANDIDATE_POOL_SIZE = 10 
        TOP_N_RESULTS = 5       
        
        if st.button(f"En İyi {TOP_N_RESULTS} Eşleşmeyi Bul", type="primary", use_container_width=True):
            start_time = time.time() 
            
            # --- Adım 1: Hızlı Filtreleme (Vektör Arama) ---
            with st.spinner(f"Adım 1/3: Tüm ilanlar taranıyor..."):
                all_jobs = get_job_postings_with_vectors()
                if not all_jobs:
                    st.warning("Hiç iş ilanı bulunamadı. Lütfen önce ilan ekleyin.")
                    st.stop()
                
                # CV vektörünü profilden al, yoksa oluştur (normalde profilde olmalı)
                cv_vector = profile.get("cv_vector")
                if not cv_vector:
                    st.info("Profilinizde CV 'parmak izi' bulunamadı, şimdi oluşturuluyor...")
                    cv_vector = get_embedding(cv_text)
                    if cv_vector:
                         db.collection("user_profiles").document(user_id).update({"cv_vector": cv_vector})
                    else:
                        st.error("CV'niz için 'parmak izi' oluşturulamadı. İşlem iptal edildi.")
                        st.stop()
                        
                job_vectors = np.array([job['vector'] for job in all_jobs])
                cv_vector_np = np.array(cv_vector)
                similarities = np.dot(job_vectors, cv_vector_np)
                
                pool_size = min(len(all_jobs), CANDIDATE_POOL_SIZE)
                top_candidate_indices = np.argsort(similarities)[-pool_size:][::-1]

            # --- Adım 2: Paralel Analiz ---
            analysis_results = []
            progress_bar = st.progress(0, text=f"Adım 2/3: En iyi {pool_size} aday analiz ediliyor... (0%)") 

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
                        st.error(f"'{matched_job['title']}' ilanı analiz edilirken hata: {e}")
                    
                    completed_count += 1
                    percent_complete = completed_count / pool_size
                    progress_bar.progress(percent_complete, text=f"Adım 2/3: Analiz ediliyor... {int(percent_complete * 100)}% tamamlandı") 
            
            progress_bar.empty()

            # --- Adım 3: Yeniden Sırala ve Göster ---
            with st.spinner(f"Adım 3/3: Sonuçlar sıralanıyor..."):
                if not analysis_results:
                    st.error("AI analizi tüm adaylar için başarısız oldu. Lütfen tekrar deneyin.")
                    st.stop()

                sorted_results = sorted(analysis_results, key=lambda x: x["score"], reverse=True)
                
                end_time = time.time()
                st.success(f"İşlem tamam! En iyi {TOP_N_RESULTS} eşleşme {end_time - start_time:.2f} saniyede bulundu.")
                st.balloons() 
                
                st.markdown("---")

                for i, result in enumerate(sorted_results[:TOP_N_RESULTS]):
                    rank = i + 1
                    job_title = result["job"]["title"]
                    score = result["score"]
                    analysis_data = result["data"]
                    
                    with st.container(border=True):
                        col_metric, col_details = st.columns([0.2, 0.8])
                        with col_metric:
                            st.metric(label=f"#{rank} Eşleşme", value=f"{score}%")
                        with col_details:
                            st.subheader(job_title)
                            with st.expander("Detaylı AI analizini görmek için tıklayın"):
                                st.subheader("Özet")
                                st.write(analysis_data.get("summary", "N/A"))
                                st.subheader("Güçlü Yönler (Artılar)")
                                pros = analysis_data.get("pros", [])
                                if pros:
                                    for pro in pros: st.markdown(f"* {pro}")
                                else:
                                    st.write("N/A") 
                                st.subheader("Zayıf Yönler (Eksiler)")
                                cons = analysis_data.get("cons", [])
                                if cons:
                                    for con in cons: st.markdown(f"* {con}")
                                else:
                                    st.write("N/A")
                    st.divider()

    # --- Sekme 2: İlan Yönetimi (Toplu Yükleme dahil) ---
    with tab2:
        st.header("Job Management")
        
        # Tekli ilan formu
        with st.form("new_job_form", clear_on_submit=True):
            st.subheader("Tek İş İlanı Ekle")
            job_title = st.text_input("İş Başlığı")
            job_description = st.text_area("İş Tanımı", height=200)
            submitted = st.form_submit_button("İlanı Kaydet & Vektör Oluştur")
            
            if submitted:
                if job_title and job_description:
                    with st.spinner("AI 'parmak izi' (vektör) oluşturuluyor..."):
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
                            st.success(f"'{job_title}' başarıyla eklendi!")
                            st.cache_data.clear() 
                        except Exception as e: st.error(f"Firebase'e kaydederken hata: {e}")
                    else: st.error("AI 'parmak izi' oluşturulamadı.")
                else: st.warning("Lütfen her iki alanı da doldurun.")

        st.divider()
        
        # Toplu ilan yükleme
        st.subheader("VEYA... CSV/Excel ile Toplu İlan Yükle")
        st.markdown("**'title'** ve **'description'** sütunlarını içeren bir dosya yükleyin.")
        
        uploaded_file = st.file_uploader("Bir CSV veya Excel dosyası seçin", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                if 'title' not in df.columns or 'description' not in df.columns:
                    st.error("Hata: Dosya 'title' ve 'description' sütunlarını içermelidir.")
                else:
                    st.success(f"'{uploaded_file.name}' dosyası okundu. {len(df)} ilan bulundu.")
                    st.dataframe(df.head())
                    
                    if st.button(f"{len(df)} İlanı İşle ve Yükle", type="primary"):
                        st.info("Toplu yükleme başlıyor... Bu işlem birkaç dakika sürebilir.")
                        progress_bar_bulk = st.progress(0, text="Başlatılıyor...")
                        success_count = 0
                        batch = db.batch()
                        
                        for index, row in df.iterrows():
                            title = str(row['title'])
                            description = str(row['description'])
                            
                            progress_text = f"İşleniyor ({index + 1}/{len(df)}): {title[:30]}..."
                            progress_bar_bulk.progress((index + 1) / len(df), text=progress_text)
                            
                            job_vector = get_embedding(f"Title: {title}\n\nDescription: {description}")
                            
                            if job_vector:
                                doc_ref = db.collection("job_postings").document()
                                batch.set(doc_ref, {
                                    "title": title,
                                    "description": description,
                                    "created_at": firestore.SERVER_TIMESTAMP,
                                    "vector": job_vector,
                                    "added_by": f"bulk_upload_{st.session_state['user_email']}"
                                })
                                success_count += 1
                        
                        batch.commit()
                        st.success(f"Tamamlandı! {len(df)} ilandan {success_count} tanesi başarıyla işlendi ve yüklendi.")
                        st.cache_data.clear()
                        
            except Exception as e:
                st.error(f"Dosya işlenirken bir hata oluştu: {e}")


    # --- (GÜNCELLENDİ) Sekme 3: Profilim (code.html'e benzetildi) ---
    with tab3:
        st.header("Profilim")
        st.markdown("`code.html` tasarımınıza uygun olarak, profil bilgilerinizi ve CV'nizi buradan yönetebilirsiniz.")
        
        # Mevcut profili yükle
        current_profile = get_user_profile(user_id)
        
        with st.form("profile_form"):
            st.subheader("1. Profesyonel Detaylar")
            st.markdown("`Auto-Matcher` sekmesinin çalışması için bu bilgileri doldurmanız gerekmektedir.")
            
            # code.html'den yeni alanlar
            full_name = st.text_input(
                "Tam Adınız (Full Name)", 
                value=current_profile.get("full_name", "")
            )
            headline = st.text_input(
                "Mesleki Başlık / Ünvan (Headline)", 
                value=current_profile.get("headline", ""),
                placeholder="Örn: Kıdemli Yazılım Geliştirici"
            )
            
            st.divider()
            
            st.subheader("2. CV'niz")
            st.markdown("AI eşleşmesi için CV'nizi yükleyin veya metin olarak yapıştırın.")

            # sign_up.html'den dosya yükleyici
            uploaded_cv_file = st.file_uploader(
                "CV'nizi Yükleyin (PDF veya DOCX)",
                type=["pdf", "docx", "txt"]
            )
            
            new_cv_text_area = st.text_area(
                "VEYA CV Metnini Buraya Yapıştırın", 
                value=current_profile.get("cv_text", ""), 
                height=300,
                help="Eğer bir dosya yüklerseniz, bu alan dikkate alınmayacaktır."
            )
            
            submitted = st.form_submit_button("Profili Kaydet", type="primary", use_container_width=True)
            
            if submitted:
                final_cv_text = ""
                
                with st.spinner("Profil kaydediliyor..."):
                    # Adım 1: CV Metnini Belirle
                    if uploaded_cv_file is not None:
                        st.info(f"'{uploaded_cv_file.name}' dosyası işleniyor...")
                        file_bytes = uploaded_cv_file.getvalue()
                        parsed_text = parse_cv_file(file_bytes, uploaded_cv_file.name)
                        if parsed_text:
                            final_cv_text = parsed_text
                        else:
                            st.error("Dosya okunamadı. Kayıt durduruldu.")
                            st.stop()
                    else:
                        final_cv_text = new_cv_text_area
                        
                    if not final_cv_text:
                        st.warning("Kaydedilecek bir CV metni bulunamadı. Lütfen bir dosya yükleyin veya metin yapıştırın.")
                        st.stop()
                        
                    # Adım 2: CV Vektörünü Oluştur
                    st.info("CV'niz için AI 'parmak izi' oluşturuluyor...")
                    cv_vector = get_embedding(final_cv_text)
                    
                    if not cv_vector:
                        st.error("CV'niz için 'parmak izi' oluşturulamadı. Profil kaydedilmedi.")
                        st.stop()
                        
                    # Adım 3: Firestore'a Kaydet
                    try:
                        db.collection("user_profiles").document(user_id).set({
                            "email": st.session_state['user_email'],
                            "full_name": full_name,
                            "headline": headline,
                            "cv_text": final_cv_text,
                            "cv_vector": cv_vector,
                            "updated_at": firestore.SERVER_TIMESTAMP
                        }, merge=True)
                        st.success("Profiliniz başarıyla kaydedildi!")
                        st.cache_data.clear() # Diğer sekmelerin veriyi yeniden çekmesi için
                    except Exception as e:
                        st.error(f"Profiliniz kaydedilirken bir hata oluştu: {e}")

# --- (GÜNCELLENDİ) LOGIN SAYFASI (sign_up.html'e benzetildi) ---
def login_page():
    
    # sign_up.html'deki gibi ortalanmış bir kutu oluştur
    with st.container():
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: center;
                padding-top: 2rem;
            ">
                <div style="
                    width: 100%;
                    max-width: 448px; /* max-w-sm */
                    background-color: white;
                    padding: 2rem; /* p-8 */
                    border-radius: 0.75rem; /* rounded-xl */
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                ">
            """, 
            unsafe_allow_html=True
        )

        # sign_up.html'den Başlık
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: center; gap: 0.75rem; margin-bottom: 1.5rem;">
                <span class="material-symbols-outlined" style="color: #005A9C; font-size: 2.2rem;">hub</span>
                <span style="font-size: 1.5rem; font-weight: 700; color: #111827;">TalentAI</span>
            </div>
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <p style="font-size: 1.875rem; font-weight: 700; color: #111827;">Welcome Back</p>
                <p style="font-size: 0.875rem; color: #6B7280;">Sign in or create an account to continue.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # sign_up.html'den Toggle
        auth_mode = st.radio(
            "Seçiminiz:",
            ["Log In", "Sign Up"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # LOG IN FORMU
        if auth_mode == "Log In":
            st.subheader("Login")
            email = st.text_input("Email", key="login_email", placeholder="Email Address")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Password")
            
            if st.button("Continue (Login)", type="primary", key="login_button", use_container_width=True):
                if email and password:
                    try:
                        user = auth_client.sign_in_with_email_and_password(email, password)
                        st.session_state['user_email'] = user['email']
                        st.session_state['user_token'] = user['idToken']
                        st.rerun() 
                    except Exception as e:
                        st.warning("Giriş başarısız. Lütfen email ve şifrenizi kontrol edin.")
                else:
                    st.warning("Lütfen email ve şifrenizi girin.")
        
        # SIGN UP FORMU
        if auth_mode == "Sign Up":
            st.subheader("Create a New Account")
            new_email = st.text_input("Email", key="signup_email", placeholder="Email Address")
            new_password = st.text_input("Password", type="password", key="signup_pass", placeholder="Password")
            
            # (NOT: HTML'deki CV Yükleme alanı kasten buraya eklenmedi.
            # En iyi pratik, kullanıcının önce hesabını oluşturması, 
            # sonra giriş yapıp profilini doldurmasıdır.)
            
            if st.button("Continue (Sign Up)", type="primary", key="signup_button", use_container_width=True):
                if new_email and new_password:
                    try:
                        user = auth_client.create_user_with_email_and_password(new_email, new_password)
                        st.success("Hesap başarıyla oluşturuldu! Lütfen 'Log In' sekmesinden giriş yapın.")
                    except Exception as e:
                        error_message = str(e)
                        if "WEAK_PASSWORD" in error_message:
                            st.warning("Şifre en az 6 karakter olmalıdır.")
                        elif "EMAIL_EXISTS" in error_message:
                            st.warning("Bu email ile kayıtlı bir hesap zaten var. Lütfen giriş yapın.")
                        elif "INVALID_EMAIL" in error_message:
                            st.warning("Lütfen geçerli bir email adresi girin.")
                        else:
                            st.error("Kayıt sırasında bilinmeyen bir hata oluştu.")
                else:
                    st.warning("Lütfen email ve şifre girin.")
        
        # Footer
        st.markdown(
            """
            <p style="font-size: 0.75rem; color: #6B7280; text-align: center; margin-top: 1.5rem;">
                By continuing, you agree to our <a href="#" target="_blank">Terms</a> and <a href="#" target="_blank">Privacy Policy</a>.
            </p>
            </div></div>
            """,
            unsafe_allow_html=True
        )

# --- ANA MANTIK ---
if st.session_state['user_email']:
    main_app()
else:
    login_page()
