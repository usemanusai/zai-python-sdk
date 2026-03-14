# **Z.AI Python SDK**

A Python client library for interacting with the Z.AI API, providing easy access to advanced language models for chat completions, streaming responses, dynamic web search, image generation, and more.

## **Installation**

pip install requests

Clone the repository:

git clone \[https://github.com/iotbackdoor/zai-python-sdk.git\](https://github.com/iotbackdoor/zai-python-sdk.git)  
cd zai-python-sdk

## **Quick Start**

from zai.client import ZAIClient

\# Initialize client with automatic authentication  
client \= ZAIClient(auto\_auth=True)

\# Simple chat completion with Advanced Web Search and System Prompt  
response \= client.simple\_chat(  
    message="What is the capital of France?",  
    model="glm-4.5v",  
    system\_prompt="You are a helpful geography assistant.",  
    web\_search=True,           \# Enable dynamic web-grounding   
    image\_generation=False     \# Toggle image generation   
)  
print(response.content)

## **Features**

* Automatic guest token authentication  
* Support for multiple AI models  
* **NEW**: Full System Prompts routing  
* **NEW**: Toggle Web Search capabilities (web\_search=True)  
* **NEW**: Toggle Image Generation  
* Streaming and non-streaming responses  
* Customizable model parameters  
* Modular architecture for flexibility

## **API Reference**

### **Simple Chat Completion**

response \= client.simple\_chat(  
    message="Your message here",  
    model="glm-4.5v",         \# or "0727-360B-API"  
    system\_prompt="Act as an expert coder.", \# Contextual System Guidelines  
    enable\_thinking=True,     \# Enable thinking mode  
    web\_search=True,          \# Ground answers dynamically  
    image\_generation=False,   \# Generate visual content  
    temperature=0.7,          \# Control randomness (0.0-2.0)  
    top\_p=0.9,                \# Control diversity (0.0-1.0)  
    max\_tokens=500            \# Maximum response length  
)

## **Error Codes**

The SDK raises ZAIError exceptions for API-related errors.
