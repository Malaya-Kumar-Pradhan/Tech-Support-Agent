# import re
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama
# from langchain_core.messages import SystemMessage, HumanMessage
# import streamlit as st

# SYSTEM_PROMPT = """You will be acting as a Tech Support Agent named Truptishree, created by the company MKP limited. Your goal is to resolve technology related queries. You will be replying to users who are on the MKP site and who will be confused if you don't respond in the character of Truptishree.

# You should maintain a friendly customer service tone.

# Here are some important rules for the interaction:
# - Always stay in character, as Truptishree, an AI from MKP limited.
# - If you are unsure how to respond, say exactly: "Sorry, I didn't understand that. Could you rephrase your question?"
# - If someone asks something irrelevant, say exactly: "Sorry, I am Truptishree and I resolve technology related queries. Do you have a technology related problem today I can help you with?"
# - Respond to only the single most recent user question. Do not invent further user turns, do not continue the conversation on your own, and do not apologize for or "correct" earlier answers unless the user actually points out a problem in their latest message.
# - Put your entire answer inside a single pair of <response></response> tags, with nothing before the opening tag and nothing after the closing tag. Never output more than one <response> block.

# Here is an example of how to respond in a standard interaction:
# <example>
# Customer: Hi, how were you created and what do you do?
# Truptishree: Hello! My name is Truptishree, and I was created by MKP site to resolve technology related queries. What can I help you with today?
# </example>"""

# RESPONSE_TAG_RE = re.compile(r"<response>(.*?)(?:</response>|$)", re.DOTALL)


# def extract_response_text(raw_text):
#     """
#     Pull out only the content between <response> and </response>.
#     Falls back to the raw (stripped) text if no tags are present at all,
#     so we degrade gracefully instead of showing an empty message.
#     """
#     if not raw_text:
#         return raw_text
#     match = RESPONSE_TAG_RE.search(raw_text)
#     if match:
#         return match.group(1).strip()
#     return raw_text.strip()


# def _message_text(result):
#     """Normalize a ChatOllama result (AIMessage or plain string) to plain text."""
#     return result.content if hasattr(result, "content") else str(result)


# @st.cache_resource
# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2"
#     )

# @st.cache_resource
# def get_llm():
#     # ChatOllama applies Llama 3.1's real chat template (unlike the raw OllamaLLM
#     # completion wrapper), so the model actually understands turn boundaries.
#     # The stop sequence is belt-and-suspenders: generation halts the instant
#     # </response> would be produced, so the model can't ramble into fake
#     # follow-up turns or "corrected" re-answers.
#     return ChatOllama(model="llama3.1", temperature=0.0, stop=["</response>"])

# def get_vector_db(database_folder="./tech_db"):
#     embeddings = get_embeddings()
#     return Chroma(
#         persist_directory=database_folder,
#         embedding_function=embeddings
#     )

# def ask_local_techrag(query,chat_summary,recent_turns_text,database_folder="./tech_db",k=3):
#     vector_db = get_vector_db(database_folder)
#     llm = get_llm()

#     try:
#         hyde_prompt = f"Write a short, paragraph-long tech abstract or definition answering this question to help find matching clauses in a tech document: '{query}'"
#         print("Expanding tech query concepts... 🧠")
#         hypothetical_answer = _message_text(llm.invoke(hyde_prompt))
#     except Exception as e:
#         print(f"HyDE expansion failed, falling back to raw query: {e}")
#         hypothetical_answer = query

#     print("Searching database for matching contexts...")
#     matching_docs=vector_db.similarity_search(hypothetical_answer, k=k)
#     facts = "\n\n".join([f"Source: {doc.metadata.get('source','Unknown')} - Content: {doc.page_content}" for doc in matching_docs])

#     human_prompt = f"""Here is the facts that you have used to answer the query:
#         {facts}

#         Here is a summary of the conversation prior to the recent turns below. It may say "(no earlier context yet)" if the conversation hasn't gone on long enough to need summarizing:
#         <summary>
#         {chat_summary if chat_summary else "(no earlier context yet)"}
#         </summary>

#         Here are the most recent conversational turns between the user and you, in chronological order. It could be empty if there is no history yet:
#         <recent_history>
#         {recent_turns_text if recent_turns_text else "(no prior turns yet)"}
#         </recent_history>

#         Here is the user's question:
#         <question>
#         {query}
#         </question>

#         Think about your answer first, then respond following the system rules. Put your entire answer inside a single <response></response> block.
#         """

#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT),
#         HumanMessage(content=human_prompt),
#     ]

#     print("Consulting Truptishree ....")
#     try:
#         raw_response = _message_text(llm.invoke(messages))
#     except Exception as e:
#         print(f"LLM call failed: {e}")
#         raw_response = ("Sorry, I'm having trouble reaching my brain right now (the local Ollama model "
#                     "may not be running). Could you check that Ollama is up and try again?")

#     answer = extract_response_text(raw_response)

#     unique_sources = set()
#     for doc in matching_docs:
#         source_name = doc.metadata.get('source','Unknown Document')
#         unique_sources.add(source_name)

#     return{
#         "answer":answer,
#         "sources":list(unique_sources)
#     }
import os
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st

# The model actually running the chat + HyDE calls. gpt-oss-20b is Groq's
# recommended fast/cheap replacement for the deprecated llama-3.1-8b-instant.
GROQ_MODEL = "openai/gpt-oss-20b"


def _get_groq_api_key():
    """
    Look for the Groq API key in Streamlit secrets first (how it's supplied on
    Streamlit Community Cloud), then fall back to a plain environment variable
    (useful for local testing / other hosts).
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")

SYSTEM_PROMPT = """You will be acting as a Tech Support Agent named Truptishree, created by the company MKP limited. Your goal is to resolve technology related queries. You will be replying to users who are on the MKP site and who will be confused if you don't respond in the character of Truptishree.

You should maintain a friendly customer service tone.

Here are some important rules for the interaction:
- Always stay in character, as Truptishree, an AI from MKP limited.
- If you are unsure how to respond, say exactly: "Sorry, I didn't understand that. Could you rephrase your question?"
- If someone asks something irrelevant, say exactly: "Sorry, I am Truptishree and I resolve technology related queries. Do you have a technology related problem today I can help you with?"
- Respond to only the single most recent user question. Do not invent further user turns, do not continue the conversation on your own, and do not apologize for or "correct" earlier answers unless the user actually points out a problem in their latest message.
- Put your entire answer inside a single pair of <response></response> tags, with nothing before the opening tag and nothing after the closing tag. Never output more than one <response> block.

Here is an example of how to respond in a standard interaction:
<example>
Customer: Hi, how were you created and what do you do?
Truptishree: Hello! My name is Truptishree, and I was created by MKP site to resolve technology related queries. What can I help you with today?
</example>"""

RESPONSE_TAG_RE = re.compile(r"<response>(.*?)(?:</response>|$)", re.DOTALL)


def extract_response_text(raw_text):
    """
    Pull out only the content between <response> and </response>.
    Falls back to the raw (stripped) text if no tags are present at all,
    so we degrade gracefully instead of showing an empty message.
    """
    if not raw_text:
        return raw_text
    match = RESPONSE_TAG_RE.search(raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def _message_text(result):
    """Normalize a ChatOllama result (AIMessage or plain string) to plain text."""
    return result.content if hasattr(result, "content") else str(result)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_llm():
    # ChatGroq gives us a hosted, fast Llama-family model (no local Ollama
    # process needed) - required for deployment on platforms like Streamlit
    # Community Cloud that can't run Ollama themselves.
    # The stop sequence is belt-and-suspenders: generation halts the instant
    # </response> would be produced, so the model can't ramble into fake
    # follow-up turns or "corrected" re-answers.
    api_key = _get_groq_api_key()
    if not api_key:
        st.error(
            "No GROQ_API_KEY found. Add it under Settings → Secrets in "
            "Streamlit Community Cloud (or as a local environment variable) "
            "to enable the chat model."
        )
        st.stop()
    return ChatGroq(model=GROQ_MODEL, temperature=0.0, api_key=api_key, stop=["</response>"])

def get_vector_db(database_folder="./tech_db"):
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=database_folder,
        embedding_function=embeddings
    )

def ask_local_techrag(query,chat_summary,recent_turns_text,database_folder="./tech_db",k=3):
    vector_db = get_vector_db(database_folder)
    llm = get_llm()

    try:
        hyde_prompt = f"Write a short, paragraph-long tech abstract or definition answering this question to help find matching clauses in a tech document: '{query}'"
        print("Expanding tech query concepts... 🧠")
        hypothetical_answer = _message_text(llm.invoke(hyde_prompt))
    except Exception as e:
        print(f"HyDE expansion failed, falling back to raw query: {e}")
        hypothetical_answer = query

    print("Searching database for matching contexts...")
    matching_docs=vector_db.similarity_search(hypothetical_answer, k=k)
    facts = "\n\n".join([f"Source: {doc.metadata.get('source','Unknown')} - Content: {doc.page_content}" for doc in matching_docs])

    human_prompt = f"""Here is the facts that you have used to answer the query:
        {facts}

        Here is a summary of the conversation prior to the recent turns below. It may say "(no earlier context yet)" if the conversation hasn't gone on long enough to need summarizing:
        <summary>
        {chat_summary if chat_summary else "(no earlier context yet)"}
        </summary>

        Here are the most recent conversational turns between the user and you, in chronological order. It could be empty if there is no history yet:
        <recent_history>
        {recent_turns_text if recent_turns_text else "(no prior turns yet)"}
        </recent_history>

        Here is the user's question:
        <question>
        {query}
        </question>

        Think about your answer first, then respond following the system rules. Put your entire answer inside a single <response></response> block.
        """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    print("Consulting Truptishree ....")
    try:
        raw_response = _message_text(llm.invoke(messages))
    except Exception as e:
        print(f"LLM call failed: {e}")
        raw_response = ("Sorry, I'm having trouble reaching my brain right now (the Groq API "
                    "may be unreachable or rate-limited). Please try again in a moment.")

    answer = extract_response_text(raw_response)

    unique_sources = set()
    for doc in matching_docs:
        source_name = doc.metadata.get('source','Unknown Document')
        unique_sources.add(source_name)

    return{
        "answer":answer,
        "sources":list(unique_sources)
    }