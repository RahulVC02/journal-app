import streamlit as st
from thesysJournal import generate_overall_model_response

def main():
    st.markdown("<h1 style='color: #CC7722;'>Journal GPT</h1>", unsafe_allow_html=True)
    st.subheader("Welcome to Journal-GPT. I am your virtual journal.")
    st.subheader("Ask me to remember something or retrieve and remind you about something you've told me before.")
    st.subheader("Type in the information in the input box and press the 'Send' button.")
    
    st.markdown("<br>", unsafe_allow_html=True)  # Adds a blank line

    if 'conversations' not in st.session_state:
        st.session_state.conversations = []
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # Display the conversation in a scrollable container
    with st.container():
        st.markdown(
            """
            <style>
            .scrollable-container {
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 10px;
                background-color: #2d2d2d; /* dark background for better contrast */
            }
            .user-message {
                text-align: right;
                color: white;
                padding: 10px;
                background-color: #1e1e1e; /* slightly darker background */
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .bot-message {
                text-align: left;
                color: black;
                padding: 10px;
                background-color: #A1A1A1; /* slightly lighter background */
                border-radius: 10px;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
        display_conversations(st.session_state.conversations)
        st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.text_area("You:", key="user_input")

    if st.button("Send"):
        if user_input.strip():
            st.session_state.conversations.append({"role": "user", "text": user_input})
            response_text = generate_overall_model_response(user_input)
            st.session_state.conversations.append({"role": "bot", "text": response_text})
            st.experimental_rerun() 

def display_conversations(conversations):
    for convo in conversations:
        if convo["role"] == "user":
            st.markdown(f"<div class='user-message'>{convo['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{convo['text']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
