import streamlit as st
import sys
import os

# Ensure the src directory is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline
from src.config.settings import settings

st.set_page_config(page_title="AmazonHelp Support Agent", page_icon="🤖", layout="wide")

# Initialize Pipeline only once using Streamlit caching
@st.cache_resource
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()

# Title and Description
st.title("AmazonHelp Customer Support RAG Agent")
st.markdown("""
This dashboard simulates the backend decision engine for the support agent.
It will analyze your query, check for safety, classify the intent, and either **AUTO-HANDLE** with retrieved evidence, or **ESCALATE** to a human expert.
""")

# Sidebar settings
with st.sidebar:
    st.header("Session Settings")
    session_id = st.text_input("Session ID", value="demo_user_123")
    is_new_session = st.checkbox("New Session (Clear Context)", value=False)
    
    st.markdown("---")
    st.markdown("**Taxonomy Intents:**")
    st.markdown("- Order & Delivery Issues\n- Prime & Membership\n- Refunds & Financials\n- Returns & Exchanges\n- Account Security & Private Support\n- Other / Human Review")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (UI only, backend tracks its own history in SQLite)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if query := st.chat_input("How can we help you today?"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process via Pipeline
    with st.spinner("Agent is analyzing request..."):
        try:
            result = pipeline.process(session_id=session_id, query=query, is_new_session=is_new_session)
            
            # Formulate response for UI
            decision = result.get("decision", "ESCALATE")
            intent = result.get("intent", "Unknown")
            confidence = result.get("confidence", 0.0)
            reason = result.get("reason", "")
            answer = result.get("answer", "")
            grounded = result.get("grounded", False)
            
            # Display Agent Thought Process in an expander
            with st.expander("🔍 Agent Decision Process", expanded=True):
                st.write(f"**Predicted Intent:** {intent} (Confidence: {confidence:.2f})")
                
                if decision == "AUTO_HANDLE":
                    st.success("✅ **Decision: AUTO_HANDLE**")
                    st.write(f"**Grounded in evidence:** {grounded}")
                    st.write(f"**Evidence Used:** {result.get('evidence_ids', [])}")
                else:
                    st.error(f"⚠️ **Decision: ESCALATE**")
                    st.write(f"**Escalation Reason:** {reason}")
            
            # Display Final Output
            with st.chat_message("assistant"):
                if decision == "AUTO_HANDLE":
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    escalation_msg = f"*I'm escalating this to a human specialist because: {reason}*"
                    st.markdown(escalation_msg)
                    st.session_state.messages.append({"role": "assistant", "content": escalation_msg})
                    
        except Exception as e:
            st.error(f"Pipeline Error: {str(e)}")
