# Azure OpenAI Chat Interface

Streamlit application for interacting with various Azure OpenAI models through a chat interface.

## Setup

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd oai-models-streamlit
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file in the root directory with your Azure OpenAI configurations:

   ```env
   # Standard model configuration
   AZURE_OPENAI_API_KEY_STANDARD=your-api-key-here
   AZURE_OPENAI_ENDPOINT_STANDARD=your-endpoint-here
   AZURE_OPENAI_API_VERSION_STANDARD=2024-xx-xx
   AZURE_OPENAI_DEPLOYMENT_NAME_STANDARD=your-deployment-name
   
   # You can add more models by following the same pattern with different suffixes
   AZURE_OPENAI_API_KEY_DATAZONE=your-api-key-here
   AZURE_OPENAI_ENDPOINT_DATAZONE=your-endpoint-here
   AZURE_OPENAI_API_VERSION_DATAZONE=2024-xx-xx
   AZURE_OPENAI_DEPLOYMENT_NAME_DATAZONE=your-deployment-name
   ```

   The application will automatically discover these models based on the environment variables.

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your default web browser.
