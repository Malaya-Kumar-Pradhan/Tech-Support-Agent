import shutil
import os
import streamlit as st

try:
    from chromadb.api.shared_system_client import SharedSystemClient
except ImportError:  # older chromadb versions
    from chromadb.api.client import SharedSystemClient


def wipe_vector_db(path):
    """Release Chroma's cached client first, then delete the folder."""
    try:
        SharedSystemClient.clear_system_cache()
    except Exception as e:
        print(f"Could not clear Chroma cache: {e}")
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def reset_application():
    st.cache_resource.clear()
    for key in ["messages", "vector_db_ready", "chat_summary", "turn_buffer"]:
        st.session_state.pop(key, None)

    if os.path.exists("document"):
        shutil.rmtree("document", ignore_errors=True)

    wipe_vector_db("tech_db")
