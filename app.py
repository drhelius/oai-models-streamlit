"""
Streamlit app for an AI chat interface using Azure OpenAI services.
Models are dynamically loaded from environment variables.
"""
import os
import time
import streamlit as st
from openai import AzureOpenAI
from models_config import get_model_names, get_model_info, get_env_variable_keys

# === DEFAULT VALUES ===

DEFAULT_SETTINGS = {
    "system_prompt": "You are a helpful assistant that provides clear and concise information.",
    "use_streaming": True,
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 1000,
    "max_retries": 0,
    "timeout": 60
}

# === SESSION STATE INITIALIZATION ===

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SETTINGS["system_prompt"]
    if "use_streaming" not in st.session_state:
        st.session_state.use_streaming = DEFAULT_SETTINGS["use_streaming"]
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_SETTINGS["temperature"]
    if "top_p" not in st.session_state:
        st.session_state.top_p = DEFAULT_SETTINGS["top_p"]
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = DEFAULT_SETTINGS["max_tokens"]
    if "max_retries" not in st.session_state:
        st.session_state.max_retries = DEFAULT_SETTINGS["max_retries"]
    if "timeout" not in st.session_state:
        st.session_state.timeout = DEFAULT_SETTINGS["timeout"]

def reset_to_defaults():
    """Reset all settings to default values"""
    st.session_state.system_prompt = DEFAULT_SETTINGS["system_prompt"]
    st.session_state.use_streaming = DEFAULT_SETTINGS["use_streaming"]
    st.session_state.temperature = DEFAULT_SETTINGS["temperature"]
    st.session_state.top_p = DEFAULT_SETTINGS["top_p"]
    st.session_state.max_tokens = DEFAULT_SETTINGS["max_tokens"]
    st.session_state.max_retries = DEFAULT_SETTINGS["max_retries"]
    st.session_state.timeout = DEFAULT_SETTINGS["timeout"]

# === MODEL INTERACTION ===

def get_openai_client(model_id):
    """
    Set up the OpenAI client based on the selected model ID.
    Returns client and deployment name.
    """
    try:
        env_keys = get_env_variable_keys(model_id)
        
        endpoint = os.getenv(env_keys["endpoint"])
        api_key = os.getenv(env_keys["api_key"])
        api_version = os.getenv(env_keys["api_version"])
        deployment_name = os.getenv(env_keys["deployment_name"])
        
        if not all([endpoint, api_key, api_version, deployment_name]):
            missing = []
            if not endpoint: missing.append(env_keys["endpoint"])
            if not api_key: missing.append(env_keys["api_key"])
            if not api_version: missing.append(env_keys["api_version"])
            if not deployment_name: missing.append(env_keys["deployment_name"])
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            max_retries=st.session_state.max_retries,
            timeout=st.session_state.timeout
        )
        
        return client, deployment_name
    except Exception as e:
        raise ValueError(f"Error setting up client for {model_id}: {str(e)}")

def generate_response_streaming(client, deployment_name, messages, temperature, max_tokens, top_p, message_placeholder):
    """Generate response using streaming mode"""
    full_response = ""
    
    # First show the thinking indicator in the message area
    message_placeholder.write("Thinking... 🤔")
    
    stream = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True
    )
    
    for chunk in stream:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.write(full_response + "▌")
    
    # Remove the cursor and display final response
    message_placeholder.write(full_response)
    return full_response

def generate_response_non_streaming(client, deployment_name, messages, temperature, max_tokens, top_p, message_placeholder):
    """Generate response without streaming"""
    thinking_text = "Generating response... ⏳"
    message_placeholder.write(thinking_text)
    
    # Visual feedback with progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 70:  # Speed varies to seem more natural
            time.sleep(0.01) 
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    
    # Remove progress bar after completion
    progress_bar.empty()
    
    full_response = response.choices[0].message.content
    message_placeholder.write(full_response)
    return full_response

# === UI COMPONENTS ===

def render_sidebar():
    """Render the sidebar with model selection and parameters"""
    with st.sidebar:
        st.title("Chat Settings")

        # Action buttons
        render_action_buttons()

        # System prompt configuration
        render_system_prompt_section()
        
        # Model selection section
        selected_model_id = render_model_section()
        
        # Generation parameters section
        temperature, top_p, max_tokens = render_generation_parameters()

        return selected_model_id, temperature, top_p, max_tokens

def render_system_prompt_section():
    """Render the system prompt configuration section"""
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=100,
        help="Define the behavior of the AI assistant",
        label_visibility="collapsed"
    )
    st.session_state.system_prompt = system_prompt

def render_model_section():
    """Render the model selection section and return the selected model ID"""
    st.subheader("Model Settings")
    
    model_options = get_model_names()
    if not model_options:
        st.error("No models found! Check your .env file configuration.")
        st.stop()
    
    # Model selection
    model_ids = [m[0] for m in model_options]
    model_display_names = [m[1] for m in model_options]
    
    selected_model_index = st.selectbox(
        "Select Model",
        range(len(model_options)),
        format_func=lambda i: model_display_names[i],
        help="Choose which Azure OpenAI deployment to use"
    )
    selected_model_id = model_ids[selected_model_index]
    
    # Streaming option
    use_streaming = st.checkbox(
        "Use Streaming",
        value=st.session_state.use_streaming,
        help="Toggle to enable or disable streaming of responses"
    )
    st.session_state.use_streaming = use_streaming
    
    return selected_model_id

def render_generation_parameters():
    """Render the generation parameters section and return the values"""
    st.subheader("Generation Parameters")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Higher values make output more random, lower values make it more deterministic"
    )
    st.session_state.temperature = temperature
    
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.top_p,
        step=0.01,
        help="Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered"
    )
    st.session_state.top_p = top_p
    
    max_tokens = st.slider(
        "Max Response Length",
        min_value=100,
        max_value=10000,
        value=st.session_state.max_tokens,
        step=100,
        help="Maximum number of tokens in the response"
    )
    st.session_state.max_tokens = max_tokens
    
    st.subheader("API Settings")
    
    max_retries = st.slider(
        "Max Retries",
        min_value=0,
        max_value=10,
        value=st.session_state.max_retries,
        step=1,
        help="Maximum number of retries for API calls"
    )
    st.session_state.max_retries = max_retries
    
    timeout = st.slider(
        "Timeout (seconds)",
        min_value=10,
        max_value=300,
        value=st.session_state.timeout,
        step=5,
        help="Timeout in seconds for API calls"
    )
    st.session_state.timeout = timeout
    
    return temperature, top_p, max_tokens

def render_action_buttons():
    """Render the action buttons for resetting and clearing chat"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Reset ⚙️", use_container_width=True):
            reset_to_defaults()
            st.rerun()
    
    with col2:
        if st.button("Clear Chat 🧹", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def render_chat_interface(selected_model_id, temperature, top_p, max_tokens):
    """Render the main chat interface"""
    st.title("AI Chat Interface")
    
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Handle new user input
    if prompt := st.chat_input("Type your message here..."):
        process_user_message(prompt, selected_model_id, temperature, top_p, max_tokens)

def process_user_message(prompt, selected_model_id, temperature, top_p, max_tokens):
    """Process a new user message and generate a response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant message with a spinner while processing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Get client and deployment name
            client, deployment_name = get_openai_client(selected_model_id)
            
            # Prepare messages for API call
            messages = [{"role": "system", "content": st.session_state.system_prompt}]
            messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])
            
            # Start measuring response time
            start_time = time.time()
            
            # Generate response based on selected method
            if st.session_state.use_streaming:
                with st.spinner("Thinking..."):
                    full_response = generate_response_streaming(
                        client, deployment_name, messages, temperature, max_tokens, top_p, message_placeholder
                    )
            else:
                full_response = generate_response_non_streaming(
                    client, deployment_name, messages, temperature, max_tokens, top_p, message_placeholder
                )
            
            # Display response metrics
            response_time = time.time() - start_time
            st.caption(f"Response time: {response_time:.2f} seconds")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"Sorry, an error occurred: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# === MAIN APP ===

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="AI Chat Interface",
        page_icon="🤖",
        layout="wide",
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get selected values
    selected_model_id, temperature, top_p, max_tokens = render_sidebar()
    
    # Render main chat interface
    render_chat_interface(selected_model_id, temperature, top_p, max_tokens)
    
    # Footer
    st.markdown("---")
    model_name = get_model_info(selected_model_id)["name"]
    st.caption(f"Using Azure OpenAI services - Current model: {model_name}")

if __name__ == "__main__":
    main()
