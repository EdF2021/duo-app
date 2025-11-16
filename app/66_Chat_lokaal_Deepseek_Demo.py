import streamlit as st
import ollama

st.title("Ollama with Deepseek local Chat")

# Initialize chat history in session state
if "deepseek_messages" not in st.session_state:
    st.session_state.deepseek_messages = []

# Display existing deepseek_messages from session state
for message in st.session_state.deepseek_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to session state and display it
    st.session_state.deepseek_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            # Use a placeholder for the streaming response
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response from the model
            stream = ollama.chat(
                # Note: 'deepseek-v3.1:671b-cloud' is not a standard model name.
                # Using a common model like 'llama3' for this example.
                # Replace 'llama3' with your actual model name.
                model="deepseek-v3.1:671b-cloud", 
                deepseek_messages=st.session_state.deepseek_messages,
                stream=True,
            )

            # Append each chunk to the full response and update the placeholder
            for chunk in stream:
                if chunk.get('message', {}).get('thinking'):
                    # Display the thinking process, but don't add it to the final response
                    response_placeholder.markdown(f"🤔 {chunk['message']['thinking']}")

                if chunk['message']['content']:
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(full_response + "▌")
            
            # Display the final complete response
            response_placeholder.markdown(full_response)

            # Add the complete assistant response to the session state
            st.session_state.deepseek_messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

