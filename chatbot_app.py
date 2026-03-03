import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your token

# --- FIX HF TOKEN (Optional) ---
# os.environ["HF_TOKEN"] = "your_huggingface_token_here"

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Support Chatbot", layout="wide")
st.title("🤖 Intelligent Customer Support Chatbot")

# Load Embedding Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 2. KNOWLEDGE BASE (FAQ) ---
faq_data = {
    "question": [
        "How do I reset my password?",
        "Where is my order?",
        "What is your return policy?",
        "How can I contact support?",
        "Do you offer refunds?",
        "What are your business hours?"
    ],
    "answer": [
        "You can reset your password by clicking 'Forgot Password' on the login page.",
        "You can track your order using the tracking number sent to your email.",
        "We accept returns within 30 days of purchase if the item is unused.",
        "You can contact us via email at support@example.com or call 1-800-123-4567.",
        "Yes, full refunds are available within the 30-day return window.",
        "Our support team is available Monday to Friday, 9 AM to 5 PM EST."
    ]
}

df_faq = pd.DataFrame(faq_data)

# --- 3. PREPROCESSING ---
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower()

# --- 4. CHAT LOGIC ---
def get_response(user_query, context_history):
    clean_query = clean_text(user_query)
    query_embedding = model.encode([clean_query])
    faq_embeddings = model.encode(df_faq['question'].tolist())
    similarities = cosine_similarity(query_embedding, faq_embeddings)[0]
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]
    
    if best_score > 0.65:
        return df_faq['answer'].iloc[best_match_idx], best_score
    else:
        return "I'm sorry, I didn't understand that. Would you like to speak to a human agent?", 0.0

# --- 5. SESSION STATE MANAGEMENT ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- 6. UI LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Chat with Support")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Type your query here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, score = get_response(prompt, st.session_state.messages)
                st.markdown(response)
                
                if score > 0.65:
                    st.caption(f"Confidence: {score:.2%}")
                else:
                    st.caption("Low Confidence - Fallback Triggered")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.header("📊 System Info")
    st.write(f"**Total FAQs:** {len(df_faq)}")
    st.write(f"**Model:** Sentence Transformer (MiniLM)")
    st.write(f"**Threshold:** 0.65")
    
    st.divider()
    st.subheader("🛠️ Admin Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.subheader("📚 Knowledge Base Preview")
    st.dataframe(df_faq.head(5), width='stretch')  # ✅ FIXED LINE

# --- 7. EVALUATION & FEEDBACK ---
st.divider()
st.subheader("📈 Performance Metrics")
st.write("Track accuracy by monitoring fallback rates.")
st.write("In production, integrate a feedback loop (Thumbs Up/Down).")