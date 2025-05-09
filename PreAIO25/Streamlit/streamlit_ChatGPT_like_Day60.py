import streamlit as st
import google.generativeai as genai

# Set the API key for Gemini
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)

# Select the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Title for the app
st.title("Chat với Gemini")

# Initialize message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user input (prompt)
if prompt := st.chat_input("Bạn muốn hỏi gì?"):
    # Display user's message
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate the response using the Gemini model
    try:
        response = model.generate_content(prompt)
        reply = response.text

        # Display the assistant's reply and save it to session state
        st.chat_message("assistant").write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"An error occurred: {e}")
