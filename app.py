import os
import streamlit as st
from streamlit_chat import message
from agent import Agent

st.set_page_config(page_title="Assignment 2 Snoop Dogg Voice chat")


def display_messages():
    """
    Displays chat messages from the session state.
    
    Iterates through messages stored in `st.session_state["messages"]`, 
    displaying each using the `message` function from `streamlit_chat`. 
    Prepares an empty container for a loading spinner.
    """
    st.subheader("Chat with snoop dogg")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["spinner"] = st.empty()

def process_input():
    """
    Processes user input from the chat interface.
    
    Checks if the user has entered a non-empty message. If so, it selects
    the appropriate method based on the user's choice, sends the query to the Agent,
    converts the Agent's text response to audio, and updates the session state with the conversation.
    """
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["spinner"], st.spinner(f"Processing"):
            # Use the selected method
            if st.session_state["selected_method"] == "fine_tuned_llama2":
                agent_text = st.session_state["agent"].ask(user_text)
            else:  # 'fine_tuned_GPT'
                agent_text = st.session_state["agent"].ask_GPT(user_text)
            # Converting the text response to audio
            audio_url = st.session_state["agent"].text_to_speech(agent_text)

        st.session_state["user_input"] = ""
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

        # Display the audio player with the generated audio URL
        if audio_url:
            st.audio(audio_url, format='audio/wav', start_time=0)


def is_openai_api_key_set() -> bool:
    """
    Checks if the OpenAI API key is set in the session state.
    
    Returns:
        bool: True if the OpenAI API key is set and non-empty, False otherwise.
    """
    return len(st.session_state["OPENAI_API_KEY"]) > 0

def main():
    """
    Main function to initialize the chat interface and handle user interactions.
    
    Sets up the Streamlit interface for the chat application, initializes session state variables,
    and handles the chat interaction flow.
    """
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        st.session_state["HF_API_KEY"] = st.secrets["HF_API_KEY"]
        st.session_state["HF_URL"] = st.secrets["HF_URL"]
        st.session_state["PLAY_HT_API_KEY"] = st.secrets["PLAY_HT_API_KEY"]
        st.session_state["PLAY_HT_USER_ID"] = st.secrets["PLAY_HT_USER_ID"]
        st.session_state["selected_method"] = "fine_tuned_GPT"  # Default value
        if is_openai_api_key_set():
            # Assuming you also store the Play.ht API key in the session state or environment variable
            st.session_state["agent"] = Agent(llama2_hf_url=st.session_state["HF_URL"],llama2_hf_api_key=st.session_state["HF_API_KEY"] ,openai_api_key=st.session_state["OPENAI_API_KEY"], playht_user_id=st.session_state["PLAY_HT_USER_ID"], playht_api_key=st.session_state["PLAY_HT_API_KEY"])
        else:
            st.session_state["agent"] = None

    st.header("Assignment 2 Snoop dog voice chat")
    
    method_selection = st.selectbox(
        "Choose the Fine tuned LLM to use:",
        options=["fine_tuned_llama2", "fine_tuned_GPT"],
        index=0,  # Default selection; 0 for 'ask', 1 for 'ask_GPT'
        key="selected_method"
    )


    display_messages()
    st.text_input("Chat with snoop", key="user_input", disabled=not is_openai_api_key_set(), on_change=process_input)

    st.divider()


if __name__ == "__main__":
    main()
