"""
Advanced conversation memory manager.

Keeps the last MAX_TURNS conversational turns verbatim (the "recent window")
and condenses anything older into a single running LLM-generated summary.
This bounds prompt size regardless of how long the conversation gets, while
still preserving important context from earlier in the chat.

This is a hand-rolled equivalent of LangChain's ConversationSummaryBufferMemory,
built without the (deprecated) langchain.memory module.
"""

MAX_TURNS = 6  # number of most-recent user+assistant turn pairs kept verbatim

SUMMARY_PROMPT_TEMPLATE = """You are maintaining a running summary of a technical support conversation
between a user and an AI support agent named Truptishree.

Below is the existing summary of the conversation so far (it may say "(none yet)" if this is the first summarization):
<existing_summary>
{existing_summary}
</existing_summary>

Below are the new conversation turns to fold into that summary:
<new_turns>
{new_turns}
</new_turns>

Write an updated, concise summary that preserves all important facts, decisions, unresolved
issues, and context the user has shared (e.g. product names, error messages, troubleshooting
steps already tried). Do not add commentary or preamble - respond with ONLY the updated summary text.
"""


def init_memory(session_state):
    """Ensure memory-related keys exist in session_state. Safe to call repeatedly."""
    if "chat_summary" not in session_state:
        session_state["chat_summary"] = ""
    if "turn_buffer" not in session_state:
        session_state["turn_buffer"] = []  # list of {"user": ..., "assistant": ...}, chronological order


def reset_memory(session_state):
    """Wipe memory state, e.g. on 'Clear All & Reset'."""
    session_state["chat_summary"] = ""
    session_state["turn_buffer"] = []


def _format_turns(turns):
    lines = []
    for turn in turns:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


def get_memory_context(session_state):
    """
    Returns (chat_summary, recent_turns_text) ready to inject into the RAG prompt.
    Both may be empty strings early in the conversation.
    Call this BEFORE generating the answer for the current question, so the
    current question is never duplicated inside its own history.
    """
    init_memory(session_state)
    summary = session_state["chat_summary"]
    recent_text = _format_turns(session_state["turn_buffer"])
    return summary, recent_text


def update_memory(session_state, user_msg, ai_msg, llm):
    """
    Call this AFTER a turn has been answered. Appends the new turn to the
    recent window; if the window exceeds MAX_TURNS, the oldest turn(s) are
    popped and folded into the running summary via one extra LLM call.
    """
    init_memory(session_state)
    session_state["turn_buffer"].append({"user": user_msg, "assistant": ai_msg})

    if len(session_state["turn_buffer"]) > MAX_TURNS:
        overflow_count = len(session_state["turn_buffer"]) - MAX_TURNS
        overflow_turns = session_state["turn_buffer"][:overflow_count]
        session_state["turn_buffer"] = session_state["turn_buffer"][overflow_count:]

        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            existing_summary=session_state["chat_summary"] or "(none yet)",
            new_turns=_format_turns(overflow_turns)
        )
        print("Condensing older turns into running summary... 🧠")
        try:
            raw_summary = llm.invoke(summary_prompt)
            summary_text = raw_summary.content if hasattr(raw_summary, "content") else str(raw_summary)
            session_state["chat_summary"] = summary_text.strip()
        except Exception as e:
            print(f"Summarization failed, keeping previous summary and re-queuing overflow turns: {e}")
            # Put the overflow turns back at the front of the window rather than losing them
            session_state["turn_buffer"] = overflow_turns + session_state["turn_buffer"]