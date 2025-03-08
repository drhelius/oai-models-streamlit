"""
Streamlit app for an AI chat interface using Azure OpenAI services.
Models are dynamically loaded from environment variables.
"""
import os
import time
import json
import streamlit as st
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from models_config import get_model_names, get_model_info, get_env_variable_keys

# === DEFAULT VALUES ===

DEFAULT_SETTINGS = {
    "system_prompt": "You are a helpful assistant that provides clear and concise information.",
    "use_streaming": False,
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 1000,
    "max_retries": 0,
    "timeout": 60,
    "selected_model_id": None,  # Will be set on first run
    "api_type": None,  # Will be set based on model selection
    "api_version": None,  # Will be set based on model selection
    "show_debug": False  # New setting for debug toggle
}

# === SESSION STATE INITIALIZATION ===

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize all default settings in session state
    for key, value in DEFAULT_SETTINGS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Set a default model on first run
    if st.session_state.selected_model_id is None:
        model_options = get_model_names()
        if model_options:
            st.session_state.selected_model_id = model_options[0][0]
            # Initialize API details for the selected model
            update_api_details()

def reset_to_defaults():
    """Reset all settings to default values"""
    for key, value in DEFAULT_SETTINGS.items():
        if key not in ['selected_model_id', 'api_type', 'api_version']:  # Keep model selection
            st.session_state[key] = value

# === MODEL INTERACTION ===

def update_api_details():
    """Update API type and version for the currently selected model"""
    try:
        env_keys = get_env_variable_keys(st.session_state.selected_model_id)
        st.session_state.api_type = os.getenv(env_keys["api_type"], "openai").lower()
        st.session_state.api_version = os.getenv(env_keys["api_version"])
        st.session_state.deployment_name = os.getenv(env_keys["deployment_name"])
        st.session_state.endpoint = os.getenv(env_keys["endpoint"])
    except Exception as e:
        st.error(f"Error updating API details: {e}")

def get_client():
    """
    Set up the client based on the selected model ID and API type.
    Returns client based on the current session state.
    """
    try:
        model_id = st.session_state.selected_model_id
        env_keys = get_env_variable_keys(model_id)
        
        endpoint = os.getenv(env_keys["endpoint"])
        api_key = os.getenv(env_keys["api_key"])
        api_version = os.getenv(env_keys["api_version"])
        deployment_name = os.getenv(env_keys["deployment_name"])
        
        # Store these in session state for reference
        st.session_state.api_version = api_version
        st.session_state.deployment_name = deployment_name
        
        if not all([endpoint, api_key, api_version, deployment_name]):
            missing = []
            if not endpoint: missing.append(env_keys["endpoint"])
            if not api_key: missing.append(env_keys["api_key"])
            if not api_version: missing.append(env_keys["api_version"])
            if not deployment_name: missing.append(env_keys["deployment_name"])
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        
        # Create client based on API type
        if st.session_state.api_type == "azure":
            return ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key)
            )
        else:  # Default to OpenAI API
            return AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                max_retries=st.session_state.max_retries,
                timeout=st.session_state.timeout
            )
    except Exception as e:
        raise ValueError(f"Error setting up client for {model_id}: {str(e)}")

def format_messages_for_api(messages):
    """Format messages based on API type from session state"""
    if st.session_state.api_type == "azure":
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                formatted_messages.append(UserMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_messages.append(AssistantMessage(content=msg["content"]))
        return formatted_messages
    else:
        # OpenAI API format (already in the correct format)
        return messages

def generate_response_streaming(client, messages, message_placeholder):
    """Generate response using streaming mode"""
    full_response = ""
    
    # Create a simulated complete response structure that resembles non-streaming
    simulated_response = {
        "id": "simulated-streaming-response",
        "model": st.session_state.deployment_name,
        "created": int(time.time()),
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ""  # Will be filled with full response at the end
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": "unknown",
            "completion_tokens": "unknown",
            "total_tokens": "unknown"
        }
    }
    
    debug_data = {
        "request": {
            "messages": messages,
            "temperature": st.session_state.temperature,
            "max_tokens": st.session_state.max_tokens,
            "top_p": st.session_state.top_p,
            "stream": True
        },
        "response": {},
        "streaming_chunks": []
    }
    
    # First show the thinking indicator in the message area
    message_placeholder.write("Thinking... 🤔")
    
    # Format messages based on API type
    formatted_messages = format_messages_for_api(messages)
    
    if st.session_state.api_type == "azure":
        stream = client.complete(
            messages=formatted_messages,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            top_p=st.session_state.top_p,
            stream=True
        )
        
        for update in stream:
            if update.choices:
                chunk_content = update.choices[0].delta.content
                if chunk_content is not None:
                    full_response += chunk_content
                    message_placeholder.write(full_response + "▌")
                    # Capture sample chunk data for debugging (limit to 3 chunks)
                    if len(debug_data["streaming_chunks"]) < 3:
                        # Convert to dict for better formatting
                        try:
                            chunk_dict = {
                                "id": getattr(update, "id", "unknown"),
                                "created": getattr(update, "created", "unknown"),
                                "choices": [{
                                    "index": update.choices[0].index,
                                    "delta": {
                                        "content": update.choices[0].delta.content
                                    }
                                }]
                            }
                            debug_data["streaming_chunks"].append(chunk_dict)
                        except:
                            # Fallback if can't convert properly
                            debug_data["streaming_chunks"].append({"chunk_data": str(update)})
    else:
        # Original OpenAI API implementation
        stream = client.chat.completions.create(
            model=st.session_state.deployment_name,
            messages=formatted_messages,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            top_p=st.session_state.top_p,
            stream=True
        )
        
        for chunk in stream:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.write(full_response + "▌")
                        # Capture sample chunk data for debugging (limit to 3 chunks)
                        if len(debug_data["streaming_chunks"]) < 3:
                            try:
                                # Convert to dict for better formatting
                                chunk_dict = chunk.model_dump()
                                debug_data["streaming_chunks"].append(chunk_dict)
                            except:
                                # Fallback if model_dump not available
                                debug_data["streaming_chunks"].append({"chunk_data": str(chunk)})
    
    # Remove the cursor and display final response
    message_placeholder.write(full_response)
    
    # Update simulated complete response with full content
    simulated_response["choices"][0]["message"]["content"] = full_response
    debug_data["response"] = simulated_response
    
    return full_response, debug_data

def generate_response_non_streaming(client, messages, message_placeholder):
    """Generate response without streaming"""
    thinking_text = "Generating response... ⏳"
    message_placeholder.write(thinking_text)
    
    debug_data = {
        "request": {
            "messages": messages,
            "temperature": st.session_state.temperature,
            "max_tokens": st.session_state.max_tokens,
            "top_p": st.session_state.top_p,
            "stream": False
        },
        "response": {}
    }
    
    # Visual feedback with progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 70:  # Speed varies to seem more natural
            time.sleep(0.01) 

    # Format messages based on API type
    formatted_messages = format_messages_for_api(messages)
    
    if st.session_state.api_type == "azure":
        response = client.complete(
            messages=formatted_messages,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            top_p=st.session_state.top_p
        )
        full_response = response.choices[0].message.content
        
        # Convert to dict for better formatting in debug data
        try:
            # First try to convert to a dict if possible
            response_dict = {
                "id": getattr(response, "id", "unknown"),
                "created": getattr(response, "created", int(time.time())),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_response
                    },
                    "finish_reason": getattr(response.choices[0], "finish_reason", "stop")
                }]
            }
            debug_data["response"] = response_dict
        except:
            # Fallback to string representation
            debug_data["response"] = {"raw_response": str(response)}
    else:
        # Original OpenAI API implementation
        response = client.chat.completions.create(
            model=st.session_state.deployment_name,
            messages=formatted_messages,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            top_p=st.session_state.top_p
        )
        full_response = response.choices[0].message.content
        
        # Convert response to dict for better formatting in debug data
        try:
            debug_data["response"] = response.model_dump()
        except:
            try:
                debug_data["response"] = json.loads(response.json())
            except:
                debug_data["response"] = {"raw_response": str(response)}

    # Remove progress bar after completion
    progress_bar.empty()
    
    message_placeholder.write(full_response)
    return full_response, debug_data

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
        render_model_section()
        
        # Generation parameters section
        render_generation_parameters()

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
    """Render the model selection section"""
    st.sidebar.subheader("Model Settings")
    
    model_options = get_model_names()
    if not model_options:
        st.error("No models found! Check your .env file configuration.")
        st.stop()
    
    # Model selection
    model_ids = [m[0] for m in model_options]
    model_display_names = [m[1] for m in model_options]
    
    # Use a key for the selectbox to ensure it refreshes properly
    selected_model = st.sidebar.selectbox(
        "Select Model",
        model_ids,
        index=model_ids.index(st.session_state.selected_model_id) if st.session_state.selected_model_id in model_ids else 0,
        format_func=lambda id: next((name for mid, name in model_options if mid == id), id),
        help="Choose which deployment to use",
        key="model_selector"
    )
    
    # If model changed, update selected model and API details
    if selected_model != st.session_state.selected_model_id:
        st.session_state.selected_model_id = selected_model
        update_api_details()
        # Clear chat history when model changes
        st.session_state.messages = []
        # Force a rerun to update the interface with the new model
        st.rerun()
    
    # Streaming option
    use_streaming = st.sidebar.checkbox(
        "Use Streaming",
        value=st.session_state.use_streaming,
        help="Toggle to enable or disable streaming of responses"
    )
    st.session_state.use_streaming = use_streaming
    
    # Debug toggle option
    show_debug = st.sidebar.checkbox(
        "Show Debug Data",
        value=st.session_state.show_debug,
        help="Show request/response debugging information"
    )
    st.session_state.show_debug = show_debug

def render_generation_parameters():
    """Render the generation parameters section"""
    st.sidebar.subheader("Generation Parameters")
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Higher values make output more random, lower values make it more deterministic"
    )
    st.session_state.temperature = temperature
    
    top_p = st.sidebar.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.top_p,
        step=0.01,
        help="Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered"
    )
    st.session_state.top_p = top_p
    
    max_tokens = st.sidebar.slider(
        "Max Response Length",
        min_value=100,
        max_value=10000,
        value=st.session_state.max_tokens,
        step=100,
        help="Maximum number of tokens in the response"
    )
    st.session_state.max_tokens = max_tokens
    
    st.sidebar.subheader("API Settings")
    
    max_retries = st.sidebar.slider(
        "Max Retries",
        min_value=0,
        max_value=10,
        value=st.session_state.max_retries,
        step=1,
        help="Maximum number of retries for API calls (Only for OpenAI API)"
    )
    st.session_state.max_retries = max_retries
    
    timeout = st.sidebar.slider(
        "Timeout (seconds)",
        min_value=10,
        max_value=300,
        value=st.session_state.timeout,
        step=5,
        help="Timeout in seconds for API calls (Only for OpenAI API)"
    )
    st.session_state.timeout = timeout

def render_action_buttons():
    """Render the action buttons for resetting and clearing chat"""
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Reset ⚙️", use_container_width=True):
            reset_to_defaults()
            st.rerun()
    
    with col2:
        if st.button("Clear Chat 🧹", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def render_chat_interface():
    """Render the main chat interface"""
    st.title("GenAI Chat")
    
    # Display existing messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show debug expander if enabled and debug data exists
            if st.session_state.show_debug and "debug_data" in message:
                try:
                    render_debug_data(message["debug_data"])
                except Exception as e:
                    st.error(f"Error rendering debug data: {str(e)}")
    
    # Handle new user input
    if prompt := st.chat_input("Type your message here..."):
        process_user_message(prompt)

def process_user_message(prompt):
    """Process a new user message and generate a response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant message with a spinner while processing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        debug_container = st.container()
        
        try:
            # Get client using current session state
            client = get_client()
            
            # Prepare messages for API call
            messages = [{"role": "system", "content": st.session_state.system_prompt}]
            messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])
            
            # Start measuring response time
            start_time = time.time()
            
            # Generate response based on selected method
            if st.session_state.use_streaming:
                with st.spinner("Thinking..."):
                    full_response, debug_data = generate_response_streaming(
                        client, messages, message_placeholder
                    )
            else:
                full_response, debug_data = generate_response_non_streaming(
                    client, messages, message_placeholder
                )
            
            # Calculate and store response time
            response_time = time.time() - start_time
            debug_data["metrics"] = {
                "response_time_seconds": round(response_time, 3),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Display response metrics
            st.caption(f"Response time: {response_time:.2f} seconds")
            
            # Add assistant response to chat history with debug data
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "debug_data": debug_data
            })
            
            # Show debug data immediately if enabled
            if st.session_state.show_debug:
                with debug_container:
                    render_debug_data(debug_data)
            
        except Exception as e:
            error_message = f"Sorry, an error occurred: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

def render_debug_data(debug_data):
    """Render debug data in a clean, formatted way"""
    with st.expander("🔍 Debug Data"):
        # Display metrics in a clean banner at the top
        if "metrics" in debug_data:
            metrics = debug_data["metrics"]
            cols = st.columns([1, 1])
            with cols[0]:
                st.markdown(f"**⏱️ Response Time:** {metrics['response_time_seconds']} seconds")
            with cols[1]:
                st.markdown(f"**🕒 Timestamp:** {metrics['timestamp']}")
            
        
        # Check if this is a streaming response by looking for streaming_chunks key
        is_streaming_data = "streaming_chunks" in debug_data
        
        if is_streaming_data:
            # This is a streaming debug data
            tab1, tab2, tab3 = st.tabs(["📋 Request", "📊 Response", "🔄 Streaming Chunks"])
            
            with tab1:
                st.json(debug_data["request"])
                
            with tab2:
                st.json(debug_data["response"])
            
            with tab3:
                st.subheader("Sample Stream Chunks")
                chunks = debug_data["streaming_chunks"]
                if not chunks:
                    st.info("No streaming chunks captured")
                else:
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}**")
                        st.json(chunk)
                        if i < len(chunks) - 1:
                            st.divider()
        else:
            # This is a non-streaming debug data
            tab1, tab2 = st.tabs(["📋 Request", "📊 Response"])
            
            with tab1:
                st.json(debug_data["request"])
            
            with tab2:
                st.json(debug_data["response"])

# === MAIN APP ===

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Multi Model AI Chat Playground",
        page_icon="🤖",
        layout="wide",
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat interface
    render_chat_interface()
    
    # Get model info for the footer
    model_info = get_model_info(st.session_state.selected_model_id)
    model_name = model_info["name"]
    
    # Display API type for the footer
    api_type_display = "Azure AI Inference API" if st.session_state.api_type == "azure" else "Azure OpenAI API"
    
    # Footer
    st.markdown("---")
    st.caption(f"Using {api_type_display} (v{st.session_state.api_version}) - Current model: {model_name} - {st.session_state.endpoint}")

if __name__ == "__main__":
    main()
