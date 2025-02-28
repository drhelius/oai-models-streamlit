"""
Simple configuration loader for LLM models.
Dynamically loads model configurations from environment variables.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _discover_models():
    """
    Discover models from environment variables by looking for variable groups
    with the pattern AZURE_OPENAI_*_SUFFIX.
    """
    models = {}
    
    # Find all variables that match AZURE_OPENAI_DEPLOYMENT_NAME_*
    deployment_vars = [v for v in os.environ if v.startswith('AZURE_OPENAI_DEPLOYMENT_NAME_')]
    
    for var in deployment_vars:
        # Extract the suffix (everything after "AZURE_OPENAI_DEPLOYMENT_NAME_")
        suffix = var.replace('AZURE_OPENAI_DEPLOYMENT_NAME_', '')
        model_id = suffix.lower()
        
        # Check if all required variables exist for this suffix
        required_vars_exist = all(
            f"AZURE_OPENAI_{key}_{suffix}" in os.environ
            for key in ["ENDPOINT", "API_KEY", "API_VERSION", "DEPLOYMENT_NAME"]
        )
        
        if required_vars_exist:
            # Get the deployment name to use as the display name
            deployment_name = os.environ[var]
            
            models[model_id] = {
                "name": deployment_name,
                "suffix": suffix
            }
    
    return models

# Build models dictionary once at import time
MODELS = _discover_models()

def get_model_names():
    """Return list of model names for the dropdown"""
    return [(key, model["name"]) for key, model in MODELS.items()]

def get_model_info(model_id):
    """Return model configuration information"""
    if model_id in MODELS:
        return MODELS[model_id]
    else:
        raise ValueError(f"Model ID '{model_id}' not found in configuration")

def get_env_variable_keys(model_id):
    """Return the environment variable keys for a model"""
    model = get_model_info(model_id)
    suffix = model["suffix"]
    
    return {
        "endpoint": f"AZURE_OPENAI_ENDPOINT_{suffix}",
        "api_key": f"AZURE_OPENAI_API_KEY_{suffix}",
        "api_version": f"AZURE_OPENAI_API_VERSION_{suffix}",
        "deployment_name": f"AZURE_OPENAI_DEPLOYMENT_NAME_{suffix}"
    }
