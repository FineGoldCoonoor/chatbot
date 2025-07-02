# app.py

import streamlit as st
from dotenv import load_dotenv

# --- UPDATED IMPORT: Using the simple and free deep-translator ---
from deep_translator import GoogleTranslator

# --- Local Imports (No changes here) ---
from src.rag_chain import load_vector_db, create_effective_rag_chain

# --- Load Environment Variables (e.g., for GROQ_API_KEY) ---
load_dotenv()

# --- NEW, SIMPLIFIED TRANSLATION FUNCTION ---
def translate_text(text: str, target_language: str) -> str:
    """
    Translates text using the free deep-translator library.
    No API key or authentication is needed.
    
    Args:
        text (str): The text to translate.
        target_language (str): The target language code (e.g., 'en', 'ta').

    Returns:
        str: The translated text.
    """
    if not text:
        return ""
    try:
        # The library automatically detects the source language.
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        # If translation fails for any reason, show an error and return the original text.
        st.error(f"Translation failed due to an error: {e}")
        return text

# --- UI Text Configuration (No changes here) ---
UI_TEXT = {
    "en": {
        "title": "Police Assistance Cell",
        "welcome": "Welcome! I am the Thoothukudi District Police Assistance bot. How can I help you?",
        "placeholder": "Type your question here...",
        "buttons": ["Emergency contacts", "Police stations", "How to file a complaint?"],
    },
    "ta": {
        "title": "காவல்துறை உதவி செயலி",
        "welcome": "வணக்கம்! தூத்துக்குடி மாவட்ட காவல்துறை உதவி செயலிக்கு உங்களை வரவேற்கிறோம். நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        "placeholder": "உங்கள் கேள்வியை இங்கு தட்டச்சு செய்யவும்...",
        "buttons": ["அவசர உதவி எண்கள்", "காவல் நிலையங்கள்", "புகார் அளிப்பது எப்படி?"],
    }
}

# --- Main App Logic ---
def main():
    st.set_page_config(page_title="Thoothukudi Police Bot", page_icon="🚨")

    # --- Language Selection Sidebar ---
    st.sidebar.title("Language / மொழி")
    language = st.sidebar.radio("Choose Language", ('English', 'Tamil'), label_visibility="collapsed")
    lang_code = "ta" if language == "Tamil" else "en"

    # --- Initialize Chat State and RAG Chain ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_chain" not in st.session_state:
        with st.spinner("Bot is warming up... Please wait."):
            vector_db = load_vector_db()
            if vector_db:
                st.session_state.rag_chain = create_effective_rag_chain(vector_db)
                st.success("Bot is ready!")
            else:
                st.error("Could not load the knowledge base. Please run 'python build_index.py' first.")
                st.stop()

    # --- Main Chat Interface ---
    st.markdown(f"<h3 style='text-align: center;'>{UI_TEXT[lang_code]['title']}</h3>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": UI_TEXT[lang_code]['welcome']})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Quick Action Buttons ---
    cols = st.columns(len(UI_TEXT[lang_code]['buttons']))
    for i, button_text in enumerate(UI_TEXT[lang_code]['buttons']):
        if cols[i].button(button_text):
            st.session_state.user_input_from_button = button_text
            st.rerun()

    # --- Handle User Input ---
    prompt = st.chat_input(UI_TEXT[lang_code]['placeholder'])
    if "user_input_from_button" in st.session_state and st.session_state.user_input_from_button:
        prompt = st.session_state.user_input_from_button
        del st.session_state.user_input_from_button

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            # Translate user's question to English if they are typing in Tamil
            input_for_llm = translate_text(prompt, 'en') if lang_code == 'ta' else prompt
            
            response = st.session_state.rag_chain.invoke({"input": input_for_llm})
            response_text = response.get("answer", "Sorry, I encountered an issue.")
            
            # Translate the English answer back to Tamil if needed
            # We add a check to avoid translating the "not found" message
            default_fallback_en = "The answer is not available in the provided documents."
            if response_text == default_fallback_en:
                if lang_code == 'ta':
                    final_response = translate_text(default_fallback_en, 'ta')
                else:
                    final_response = default_fallback_en
            else:
                if lang_code == 'ta':
                    final_response = translate_text(response_text, 'ta')
                else:
                    final_response = response_text            
        
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.rerun()

if __name__ == "__main__":
    main()