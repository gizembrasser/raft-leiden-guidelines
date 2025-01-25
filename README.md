# RAG App with fine-tuned Llama-3.1 model

Chat with an LLM trained and fine-tuned on a RAFT dataset based on the Leiden Guidelines.

## Prerequisites

To set up the project environment and install the necessary dependencies, you'll need to have the following software installed on your computer:

1. Python: Ensure that you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
2. Ollama: Ensure that you have Ollama installed. You can download it from [ollama.com](https://ollama.com/download).
2. `pip`: This is the package installer for Python. It usually comes with Python. You can check if you have it by running `pip --version` in your command line.

## Setup

1. Download the project files. You can either clone the repository using Git or download the files directly as a ZIP archive.
2. Set up a virtual environment with Python 3.11 version:

```
python -m venv .venv
source .venv\Scripts\activate
```

3. Once the virtual environment is activated, install the project dependencies using the requirements.txt file:

```
pip install -r requirements.txt
```

4. Pull the llama3 model with Ollama:

```
ollama pull llama3
```

5. To start the Streamlit app, run the following command to access it in your web browser at the provided URL:

```
streamlit run app.py
```