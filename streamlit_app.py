import streamlit as st

st.set_page_config(page_title="Hata Ayıklayıcı", layout="wide")
st.title("🕵️ Streamlit Secrets Hata Ayıklayıcısı")
st.markdown("---")

st.header("Streamlit Gerçekte Hangi Sırları Görüyor?")

# st.secrets'taki tüm anahtarları (key) listele
all_secrets = st.secrets.keys()

st.info(f"Streamlit'in gördüğü tüm anahtar isimleri: **{list(all_secrets)}**")

st.markdown("---")
st.header("Test Sonuçları:")

# 1. Test: `firebase_credentials` (Bizim aradığımız)
if "firebase_credentials" in all_secrets:
    st.success("✅ 'firebase_credentials' (küçük harf) anahtarı bulundu. Sorun bu değil.")
else:
    st.error("❌ 'firebase_credentials' (küçük harf) anahtarı BULUNAMADI.")
    st.warning("Lütfen Secrets bölümündeki anahtarın adının `[firebase_credentials]` (küçük harf, köşeli parantezli) olduğundan emin olun.")

# 2. Test: `GEMINI_API_KEY`
if "GEMINI_API_KEY" in all_secrets:
    st.success("✅ 'GEMINI_API_KEY' anahtarı bulundu.")
else:
    st.error("❌ 'GEMINI_API_KEY' anahtarı BULUNAMADI.")

# 3. Test: `FIREBASE_WEB_API_KEY`
if "FIREBASE_WEB_API_KEY" in all_secrets:
    st.success("✅ 'FIREBASE_WEB_API_KEY' anahtarı bulundu.")
else:
    st.error("❌ 'FIREBASE_WEB_API_KEY' anahtarı BULUNAMADI.")
