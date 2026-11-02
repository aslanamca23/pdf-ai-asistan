from google import genai
from google.genai import types
import edge_tts
import asyncio
import os
import io
import pypdf
import streamlit as st

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Sesli PDF & Web Asistanı", page_icon="🎧", layout="wide")
st.title("🎧 Sesli PDF & Web Araştırma Asistanı")

# --- API Anahtarı ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        api_key = st.text_input("Google Gemini API Key", type="password")
        st.markdown("[Ücretsiz API Key Al](https://aistudio.google.com/app/apikey)")

if not api_key:
    st.warning("Lütfen Gemini API Key'ini tanımlayın.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- PDF Metin Çıkarma (Sadece bir kez, dosya yüklendiğinde) ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        pdf_stream = io.BytesIO(uploaded_file.getvalue())
        reader = pypdf.PdfReader(pdf_stream)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        st.error(f"PDF okunurken hata: {e}")
    return text

# --- Sidebar: Dosya Yükleme ---
with st.sidebar:
    st.divider()
    st.markdown("### 📂 Dosya Yükleme")
    uploaded_file = st.file_uploader("Bir PDF dosyası yükle", type=["pdf"], key="pdf_uploader")

pdf_context = ""
if uploaded_file is not None:
    with st.spinner("PDF okunuyor..."):
        pdf_context = extract_text_from_pdf(uploaded_file)
    st.sidebar.success(f"'{uploaded_file.name}' hafızaya alındı!")

# --- Sohbet Geçmişi ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Edge TTS ile Ses Üretme Fonksiyonu ---
async def generate_audio(text, voice="tr-TR-AhmetNeural"):
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

# --- Kullanıcı Girişi ---
if prompt := st.chat_input("Bir şeyler sor (örn: 'Bu PDF'i sesli olarak özetle')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yapay zeka düşünüyor ve ses hazırlıyor..."):
            try:
                system_instruction = (
                    "Sen gelişmiş bir araştırma ve PDF analiz asistanısın. "
                    "Kullanıcının sorularına hem yüklediği PDF belgesindeki bilgileri baz alarak "
                    "hem de internet araması yaparak en doğru, güncel ve yapılandırılmış yanıtı ver. "
                    "Yanıtlarını sesli okunacağı için kısa ve doğal paragraflar halinde, Türkçe olarak ver."
                )

                # PDF varsa ve soru PDF ile ilgiliyse bağlamı ekle, yoksa sadece prompt gönder
                if pdf_context and any(kw in prompt.lower() for kw in ["pdf", "özet", "oku", "belge", "metin"]):
                    # PDF metnini çok uzun tutmamak için ilk 15000 karakterle sınırla
                    full_prompt = f"PDF İçeriği:\n---\n{pdf_context[:15000]}\n---\n\nKullanıcı İsteği: {prompt}"
                else:
                    full_prompt = prompt

                # Model: gemini-2.5-flash (Ücretsiz katmanda Google Arama destekler)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=[{"google_search": {}}], # Arama aracı
                    ),
                )

                answer = response.text
                st.markdown(answer)

                # Edge TTS ile sesi oluştur (asenkron fonksiyonu çalıştır)
                audio_bytes = asyncio.run(generate_audio(answer))
                
                # Ses oynatıcı
                st.audio(audio_bytes, format="audio/mp3")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
