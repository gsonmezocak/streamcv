import streamlit as st
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import uuid # Benzersiz ID oluşturmak için

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Destekli CV Eşleştirme",
    page_icon="🤖"
)

# --- GOOGLE SHEETS BAĞLANTISI ---
@st.cache_resource
def connect_to_google_sheets():
    try:
        # Streamlit Secrets'tan kimlik bilgilerini al
        creds_dict = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
        scopes = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        # E-Tabloyu adıyla aç
        sheet_name = st.secrets["GOOGLE_SHEET_NAME"]
        spreadsheet = client.open(sheet_name)
        # Google E-Tablonuzdaki ilk sekmenin adının 'Sheet1' olduğundan emin olun!
        # Değilse, E-tabloya gidip adını 'Sheet1' olarak değiştirin.
        worksheet = spreadsheet.worksheet("Sheet1") 
        return worksheet
    except gspread.exceptions.WorksheetNotFound:
        st.error("Google E-Tablonuzda 'Sheet1' adında bir çalışma sayfası bulunamadı. Lütfen sekme adını kontrol edin.")
        return None
    except Exception as e:
        st.error(f"Google Sheets'e bağlanırken hata oluştu: {e}")
        return None

# --- GEMINI AI FONKSİYONLARI ---
def configure_gemini():
    try:
        # API Anahtarını DOĞRU İSİMLE çağırıyoruz
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Çalıştığı kanıtlanmış DOĞRU MODELİ kullanıyoruz
        return genai.GenerativeModel('gemini-1.5-pro-latest')
    except Exception as e:
        st.error("Gemini API Anahtarı bulunamadı veya geçersiz. Lütfen Streamlit Secrets'ı kontrol edin.")
        st.stop()

def get_gemini_analysis(cv, ilan):
    model = configure_gemini()
    if model is None:
        return "Model yüklenemediği için analiz yapılamadı."

    prompt = f"""
    Sen kıdemli bir İnsan Kaynakları (İK) uzmanısın ve görevin bir CV ile bir iş ilanını karşılaştırmak.
    Aşağıdaki CV metni ile İŞ İLANI metnini detaylıca analiz et.

    Analizini yaparken şu adımları izle:
    1.  **Genel Uyum Skoru:** CV'nin ilana uygunluğunu 100 üzerinden puanla.
    2.  **Güçlü Yönler (Artılar):** Adayın ilandaki gereksinimleri karşılayan en güçlü 3-4 yönünü listele.
    3.  **Zayıf Yönler / Eksiklikler (Eksiler):** İlanda aranan ancak CV'de bulunmayan veya zayıf olan 3-4 noktayı listele.
    4.  **Değerlendirme Özeti:** 2-3 cümlelik kısa bir genel değerlendirme yazısı yaz.

    Lütfen cevabını net başlıklar kullanarak **Markdown formatında** ver.

    ---[CV METNİ]----
    {cv}
    -----------------

    ---[İŞ İLANI METNİ]---
    {ilan}
    -----------------
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Analiz sırasında bir hata oluştu: {e}"

# --- UYGULAMA ARAYÜZÜ ---

st.title("🤖 AI Destekli CV - İş İlanı Eşleştirme (MVP)")
st.markdown("Bu uygulama, Gemini AI kullanarak bir CV metni ile bir iş ilanı metni arasındaki uyumu analiz eder.")

# Veritabanına (Google Sheets) bağlan
worksheet = connect_to_google_sheets()
if worksheet is None:
    st.info("Google Sheets bağlantısı kurulamadı. Lütfen 'Secrets' ayarlarınızı ve E-Tablo sekme adınızı kontrol edin.")
    st.stop()

# Arayüzü iki sekmeye ayıralım: Biri adaylar, biri IK'cılar için
tab1, tab2 = st.tabs(["🤖 CV Eşleştirme (Aday Görünümü)", "📝 Yeni İlan Ekle (İK Görünümü)"])

# --- TAB 1: CV EŞLEŞTİRME ---
with tab1:
    st.header("CV'nizi Mevcut İlanlarla Eşleştirin")
    
    # İlanları Google Sheet'ten çek ve DataFrame'e dönüştür
    with st.spinner("İş ilanları yükleniyor..."):
        try:
            records = worksheet.get_all_records()
            ilan_df = pd.DataFrame.from_records(records)
            
            # Eğer DataFrame boş değilse devam et
            if not ilan_df.empty and 'ilan_basligi' in ilan_df.columns and 'ilan_detayi' in ilan_df.columns:
                
                ilan_dict = pd.Series(ilan_df.ilan_detayi.values, index=ilan_df.ilan_basligi).to_dict()
                
                selected_ilan_basligi = st.selectbox(
                    "Eşleştirmek istediğiniz iş ilanını seçin:",
                    options=ilan_dict.keys()
                )
                
                if selected_ilan_basligi:
                    # Seçilen ilanın detayını göster
                    selected_ilan_detayi = ilan_dict[selected_ilan_basligi]
                    with st.expander("Seçilen İlanın Detayları"):
                        st.text(selected_ilan_detayi)
                    
                    # CV metin alanı
                    cv_text = st.text_area("CV metninizi buraya yapıştırın", height=250, key="cv_text_tab1")
                    
                    # Analiz butonu
                    if st.button("Uyum Analizi Yap", type="primary", use_container_width=True, key="analiz_button_tab1"):
                        if cv_text:
                            with st.spinner("Gemini AI, CV ve ilanı analiz ediyor... Lütfen bekleyin."):
                                analiz_sonucu = get_gemini_analysis(cv_text, selected_ilan_detayi)
                                st.divider()
                                st.subheader("✨ Analiz Sonucu")
                                st.markdown(analiz_sonucu)
                        else:
                            st.warning("Lütfen CV metninizi girin.")
                else:
                    st.info("Henüz sisteme eklenmiş bir iş ilanı bulunmuyor.")
            else:
                st.info("Henüz sisteme eklenmiş bir iş ilanı bulunmuyor veya E-Tablo sütunları ('ilan_basligi', 'ilan_detayi') yanlış.")
        
        except gspread.exceptions.APIError as e:
            st.error(f"Google Sheets API hatası: {e.response.json().get('error', {}).get('message', 'Bilinmeyen API hatası')}")
        except Exception as e:
            st.error(f"İlanlar yüklenirken bir hata oluştu: {e}")


# --- TAB 2: YENİ İLAN EKLEME ---
with tab2:
    st.header("Sisteme Yeni İş İlanı Ekleyin")
    
    with st.form("ilan_formu", clear_on_submit=True):
        ilan_basligi = st.text_input("İlan Başlığı (Örn: Kıdemli Python Geliştirici)")
        ilan_detayi = st.text_area("İlanın Tam Metni (Gereksinimler, iş tanımı vs.)", height=300)
        
        submitted = st.form_submit_button("İlanı Kaydet")
        
        if submitted:
            if ilan_basligi and ilan_detayi:
                try:
                    # Yeni satır için verileri hazırla
                    yeni_id = str(uuid.uuid4()) # Benzersiz bir ID oluştur
                    yeni_ilan_satiri = [yeni_id, ilan_basligi, ilan_detayi]
                    
                    # Google Sheet'e yeni satırı ekle
                    worksheet.append_row(yeni_ilan_satiri)
                    
                    st.success(f"'{ilan_basligi}' başlıklı ilan başarıyla sisteme eklendi!")
                except Exception as e:
                    st.error(f"İlan kaydedilirken bir hata oluştu: {e}")
            else:
                st.warning("Lütfen hem ilan başlığını hem de detayını girin.")
