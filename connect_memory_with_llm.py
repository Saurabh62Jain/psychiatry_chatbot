import os
import streamlit as st
import streamlit.components.v1 as components
import requests
from typing import Any, List, Optional
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Updated to a valid HuggingFace model repo
REPO_ID = "Qwen/Qwen2.5-7B-Instruct" 
DB_FAISS_PATH = "vectorstore/db_faiss"

class SimpleHFChatModel(BaseChatModel):
    repo_id: str
    hf_token: str
    temperature: float = 0.5
    max_new_tokens: int = 512

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        formatted_messages = []
        for msg in messages:
            role = "user"
            if msg.type == "human" or msg.type == "user":
                role = "user"
            elif msg.type == "ai" or msg.type == "assistant":
                role = "assistant"
            elif msg.type == "system":
                role = "system"
            formatted_messages.append({"role": role, "content": msg.content})

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "model": self.repo_id,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens
        }
        url = "https://router.huggingface.co/v1/chat/completions"
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise ValueError(f"Hugging Face Router API Error ({response.status_code}): {response.text}")
        res_data = response.json()
        content = res_data["choices"][0]["message"]["content"]
        ai_msg = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _llm_type(self) -> str:
        return "simple_hf_chat"

@st.cache_resource
def get_vectorstore():
    """Loads the FAISS database locally and caches it for Streamlit."""
    if not os.path.exists(DB_FAISS_PATH):
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    """Defines the strict instructions for the LLM to prevent hallucinations."""
    # Fixed the variable from {question} to {input} to match create_retrieval_chain defaults
    custom_prompt_template = """Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't provide anything out of the given context.
    
    Context: {context}
    Question: {input}
    
    Start the answer directly. No small talk please.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "input"])
    return prompt

def load_llm(hf_token):
    """Connects to the LLM via Hugging Face API."""
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Please set it in your .env file or in the sidebar.")
        
    chat_model = SimpleHFChatModel(
        repo_id=REPO_ID,
        hf_token=hf_token,
        temperature=0.5,
        max_new_tokens=512
    )
    return chat_model

def add_custom_style(image_path="background.jpg"):
    import base64
    
    # Custom CSS base
    css = """<style>
/* Hide Streamlit default UI elements (Deploy button, MainMenu, Header, Footer) */
header {visibility: hidden !important; height: 0px !important;}
footer {visibility: hidden !important;}
#MainMenu {visibility: hidden !important;}
.stAppDeployButton {display: none !important;}

/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');

/* Global styles */
.stApp {
    font-family: 'Outfit', sans-serif !important;
}

/* Calming titles */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    color: #1A3E35 !important; /* Elegant Forest Green */
    text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.9);
}

/* Glassmorphic Chat Messages */
.stChatMessage {
    background-color: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border-radius: 18px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
}

/* User chat messages distinct styling */
.stChatMessage[data-testid="stChatMessageUser"] {
    background-color: rgba(230, 240, 235, 0.85) !important; /* Sage hint for user */
    border: 1px solid rgba(46, 90, 80, 0.15) !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: rgba(245, 247, 245, 0.95) !important;
    backdrop-filter: blur(5px) !important;
    border-right: 1px solid rgba(0, 0, 0, 0.05) !important;
}

/* Input field container styling */
.stChatInputContainer {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border-radius: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
}

/* Custom style for source documents */
.stMarkdown hr {
    border-color: rgba(46, 90, 80, 0.2) !important;
}

.stMarkdown em, .stMarkdown strong {
    color: #2E5A50 !important;
}
</style>"""
    
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            bin_str = base64.b64encode(data).decode()
            bg_css = f"""<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{bin_str}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>"""
            css += bg_css
        except Exception as e:
            pass
    else:
        fallback_css = """<style>
.stApp {
    background: linear-gradient(135deg, #E6ECE9 0%, #D8E4DF 100%) !important;
}
</style>"""
        css += fallback_css
        
    st.markdown(css, unsafe_allow_html=True)

def render_speaker_button(text: str, key: str):
    """Renders a custom glassmorphic HTML/JS speaker button that reads text aloud using Web Speech API."""
    import json
    # Use json.dumps to safely escape characters (quotes, backslashes, newlines) for Javascript
    safe_text = json.dumps(text)
    
    html_code = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap');
    
    body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: 'Outfit', sans-serif;
        overflow: hidden;
    }}
    
    .tts-container {{
        display: flex;
        align-items: center;
        margin-top: 4px;
        margin-bottom: 4px;
    }}
    
    .speaker-btn {{
        background: rgba(46, 90, 80, 0.08);
        border: 1px solid rgba(46, 90, 80, 0.15);
        color: #2E5A50;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s ease-in-out;
        outline: none;
        user-select: none;
    }}
    
    .speaker-btn:hover {{
        background: rgba(46, 90, 80, 0.16);
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(46, 90, 80, 0.08);
    }}
    
    .speaker-btn:active {{
        transform: translateY(0);
    }}
    
    .speaker-btn.playing {{
        background: #2E5A50;
        color: white;
        border-color: #2E5A50;
    }}
    
    .speaker-btn.playing:hover {{
        background: #244840;
    }}
    
    /* Subtle pulsing soundwave icon when playing */
    .soundwave {{
        display: none;
        align-items: center;
        gap: 2px;
        height: 10px;
        width: 12px;
    }}
    
    .speaker-btn.playing .soundwave {{
        display: flex;
    }}
    
    .speaker-btn.playing .static-icon {{
        display: none;
    }}
    
    .bar {{
        width: 2px;
        height: 100%;
        background-color: white;
        border-radius: 1px;
        animation: bounce 0.8s ease-in-out infinite alternate;
    }}
    .bar:nth-child(2) {{ animation-delay: 0.25s; }}
    .bar:nth-child(3) {{ animation-delay: 0.5s; }}
    
    @keyframes bounce {{
        0% {{ height: 3px; }}
        100% {{ height: 10px; }}
    }}
    </style>
    
    <div class="tts-container">
        <button id="btn" class="speaker-btn" onclick="toggleSpeech()">
            <span class="static-icon">🔊</span>
            <div class="soundwave">
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
            </div>
            <span id="label">Read Aloud</span>
        </button>
    </div>
    
    <script>
    let synth = window.speechSynthesis || window.parent.speechSynthesis;
    let utterance = null;
    let isPlaying = false;
    
    function toggleSpeech() {{
        const btn = document.getElementById('btn');
        const label = document.getElementById('label');
        const textToSpeak = {safe_text};
        
        if (!synth) {{
            alert("Text-to-speech is not supported in this browser.");
            return;
        }}
        
        if (isPlaying) {{
            synth.cancel();
            setStopState();
        }} else {{
            synth.cancel();
            
            utterance = new SpeechSynthesisUtterance(textToSpeak);
            utterance.rate = 0.95;
            utterance.pitch = 1.05;
            
            utterance.onend = function() {{
                setStopState();
            }};
            
            utterance.onerror = function() {{
                setStopState();
            }};
            
            synth.speak(utterance);
            isPlaying = true;
            btn.classList.add('playing');
            label.textContent = "Stop Reading";
        }}
    }}
    
    function setStopState() {{
        const btn = document.getElementById('btn');
        const label = document.getElementById('label');
        isPlaying = false;
        btn.classList.remove('playing');
        label.textContent = "Read Aloud";
    }}
    
    window.addEventListener('unload', () => {{
        if (isPlaying && synth) {{
            synth.cancel();
        }}
    }});
    </script>
    """
    components.html(html_code, height=36)

def main():
    add_custom_style("background.jpg")
    st.title("AI Psychiatry Therapy Session")

    # Access HF_TOKEN from environment, or let user input it in sidebar
    token_to_use = HF_TOKEN
    if not token_to_use:
        st.sidebar.subheader("Hugging Face API Settings")
        token_to_use = st.sidebar.text_input("Enter HF Token (required if not in .env):", type="password")

    # Initialize chat history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                speak_text = message.get("speak_content")
                if not speak_text:
                    parts = message["content"].split("\n\n---\n**Source Documents:**")
                    speak_text = parts[0]
                render_speaker_button(speak_text, key=f"speak_{idx}")

    # Chat Input
    prompt = st.chat_input("Ask a medical question based on the textbook...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Setup RAG Pipeline
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error(f"Failed to load the vector store. Ensure '{DB_FAISS_PATH}' exists and run the creation script first.")
                return

            llm = load_llm(token_to_use)
            qa_prompt = set_custom_prompt()

            # Create the QA Chain
            combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
            qa_chain = create_retrieval_chain(
                retriever=vectorstore.as_retriever(search_kwargs={'k': 4}), 
                combine_docs_chain=combine_docs_chain
            )
            
            with st.spinner("Analyzing medical texts..."):
                response = qa_chain.invoke({"input": prompt})
                
            result = response["answer"]
            source_docs = response["context"]

            # Format the output to show source metadata
            result_to_show = f"{result}\n\n---\n**Source Documents:**\n"
            for i, doc in enumerate(source_docs, 1):
                page_num = doc.metadata.get('page', 'Unknown Page')
                # Optional: adding the source file name if available
                source_file = doc.metadata.get('source', 'Unknown Source')
                file_name = os.path.basename(source_file)
                result_to_show += f"* Source {i}: {file_name} (Page {page_num})\n"

            # Display response
            with st.chat_message("assistant"):
                st.markdown(result_to_show)
                render_speaker_button(result, key=f"speak_{len(st.session_state.messages)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": result_to_show,
                "speak_content": result
            })

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()