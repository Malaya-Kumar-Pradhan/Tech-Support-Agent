import shutil
import os
import streamlit as st

def reset_application():
    st.cache_resource.clear()
    for key in ["messages","vector_db_ready","chat_summary","turn_buffer"]:
        st.session_state.pop(key,None)

    if os.path.exists("document"):
        shutil.rmtree("document",ignore_errors=True)

    if os.path.exists("tech_db"):
        shutil.rmtree("tech_db",ignore_errors=True)