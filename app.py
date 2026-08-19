import os
import time
import streamlit as st
from data_processing import extract_text_from_pdf,extract_text_from_txt,build_vector_database,chunk_document_text
from helper import reset_application
from techbot import ask_local_techrag,get_llm,get_embeddings
from memory import init_memory,get_memory_context,update_memory
import shutil

DB_DIR = "./tech_db"
DOC_DIR = "document"

if "vector_db_ready" not in st.session_state:
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        st.session_state["vector_db_ready"] = True
    else:
        st.session_state["vector_db_ready"] = False

def check_for_existing_files(folder_path = DOC_DIR):
    return os.path.exists(folder_path) and len(os.listdir(folder_path)) > 0

if st.session_state["vector_db_ready"] and not check_for_existing_files():
    st.session_state["vector_db_ready"] = False
    shutil.rmtree(DB_DIR)

st.set_page_config(page_title = "Tech Support Agent",page_icon ="👩🏻‍💻")
st.title("Tech Support Agent 👩🏻‍💻")

with st.sidebar:
    st.header("Document Management")
    if not st.session_state["vector_db_ready"] and check_for_existing_files():
        st.info("Existing documents detected! Syncing indexes....")
        with st.spinner("Rebuilding localized vector database"):
            all_chunks = []
            for filename in os.listdir(DOC_DIR):
                file_path = os.path.join(DOC_DIR,filename)
                if filename.endswith(".pdf"):
                    raw_text = extract_text_from_pdf(file_path)
                elif filename.endswith(".txt"):
                    raw_text = extract_text_from_txt(file_path)
                else:
                    continue
                document_chunks = chunk_document_text(raw_text,filename)
                all_chunks.extend(document_chunks)
            if all_chunks:
                build_vector_database(all_chunks,database_folder=DB_DIR,embeddings=get_embeddings())
                st.session_state["vector_db_ready"] = True
                st.success("Database linked!")
                st.rerun()
    uploaded_files = st.file_uploader("Upload Tech related documents",type = ["pdf","txt"],accept_multiple_files = True)
    upload_clicked = st.button("Upload files")
    if uploaded_files and upload_clicked:
        os.makedirs(DOC_DIR,exist_ok=True)
        new_files_to_process = []
        for file in uploaded_files:
            target_path = os.path.join(DOC_DIR,file.name)
            if not os.path.exists(target_path):
                with open(target_path,"wb") as f:
                    f.write(file.getvalue())
                new_files_to_process.append(file.name)

        if new_files_to_process:
            with st.spinner("Analyzing and embedding new tech files..."):
                all_chunks = []
                for filename in new_files_to_process:
                    file_path = os.path.join(DOC_DIR,filename)
                    if filename.endswith(".pdf"):
                        raw_text = extract_text_from_pdf(file_path)
                    elif filename.endswith(".txt"):
                        raw_text = extract_text_from_txt(file_path)
                    else:
                        continue
                    document_chunks = chunk_document_text(raw_text,filename)
                    all_chunks.extend(document_chunks)
                if all_chunks:
                    build_vector_database(all_chunks,database_folder=DB_DIR,embeddings=get_embeddings())
                st.session_state["vector_db_ready"] = True
                st.success("Database updated successfully")
                time.sleep(3)
                st.rerun()
    st.markdown("----")
    st.subheader("System Actions")
    if st.button("🔄 Clear All & Reset", type="primary", use_container_width=True):
        with st.spinner("Wiping secure local system vectors..."):
            reset_application()
        st.rerun()
if "messages" not in st.session_state:
    st.session_state["messages"] = []
init_memory(st.session_state)

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View referenced query Documents"):
                for source in msg["sources"]:
                    st.caption(f"📁 {source}")
if st.session_state["vector_db_ready"]:
    user_question = st.chat_input(placeholder="Ask a question about the technology related documents....")
    if user_question and user_question.strip():
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state["messages"].append({"role":"user","content":user_question})

        # Pull memory context (running summary + last few turns) BEFORE answering,
        # so the current question is never duplicated inside its own history.
        chat_summary, recent_turns_text = get_memory_context(st.session_state)

        with st.spinner("Analyzing document contexts via Truptishree..."):
            result_dict = ask_local_techrag(
                user_question,
                chat_summary=chat_summary,
                recent_turns_text=recent_turns_text,
                database_folder=DB_DIR
            )
            answer = result_dict["answer"]
            sources = result_dict["sources"]
        with st.chat_message("assistant"):
            st.write(answer)

            if sources:
                with st.expander("🔍 View Referenced Query Documents", expanded=False):
                    for source in sources:
                        st.caption(f"📁 {source}")

        st.session_state["messages"].append({
            "role":"assistant",
            "content":answer,
            "sources":sources
        })

        # Update memory AFTER answering: append this turn to the recent window,
        # summarizing older turns into the running summary if the window overflows.
        update_memory(st.session_state, user_question, answer, get_llm())
else:
    st.info("Please upload Tech documents via the sidebar panel to initialize the workspace assistant")