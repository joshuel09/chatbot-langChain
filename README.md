# chatbot-langChain

A chatbot for talking to your PDF data, built with LangChain, OpenAI, FAISS vector search, and conversational memory, with a Streamlit UI. A second app (`langChainAgent.py`) is a LangChain agent for image analysis: captioning, object detection, emotion detection, and age/gender estimation.

> **Note:** This is a 2023 demonstration project, kept as-is for reference. It uses the LangChain 0.0.x APIs (`langchain.chat_models`, `initialize_agent`, etc.), which have since been reorganized in LangChain 0.1+. Treat it as a historical example rather than production-ready code.

## Installation

1. Create and activate a virtual environment:

   ```sh
   # macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

   ```bat
   :: Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the dependencies (pinning LangChain to the API version this project was written against):

   ```sh
   pip install streamlit streamlit_chat "langchain<0.1" openai faiss-cpu tiktoken pypdf
   ```

3. Run the PDF chatbot:

   ```sh
   streamlit run langChainExp.py
   ```

   Or run the image-analysis agent (also needs `transformers`, `torch`, `deepface`, and `opencv-python`):

   ```sh
   streamlit run langChainAgent.py
   ```

Paste your OpenAI API key into the sidebar field and upload a file to start chatting. The key is read at runtime and never stored.
