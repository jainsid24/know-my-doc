# KnowMyDoc Utility

Introducing KnowMyDoc, a Python-based conversational AI utility that enables you to build a chatbot with your own data sources and web pages. With KnowMyDoc, you can easily create a chatbot that can answer complex questions by utilizing advanced machine learning techniques and natural language processing (NLP) algorithms.

KnowMyDoc leverages the LangChain library for LLM prompt engineering and conversation chaining. This means that you can easily customize the chatbot's prompts and personalize its responses based on the context and tone of the conversation. With KnowMyDoc's sophisticated LLM-based approach, the chatbot can maintain a consistent and coherent conversation even when dealing with large amounts of data.

KnowMyDoc also utilizes the Pinecone vector similarity search engine to enable fast and efficient lookup of relevant data. By creating embeddings of your documents and web pages, KnowMyDoc can quickly identify and retrieve the most relevant information for the user's queries.

Other features of KnowMyDoc include:

* Document loading from local data sources and web pages
* Text splitting to optimize indexing and similarity search
* NLTK support for text processing and tokenization
* Support for OpenAI embeddings and vector stores, including Chroma and Pinecone
* Logging support for troubleshooting and analysis

## Getting Started

### Prerequisites
This utility requires Python 3.9 or later. You will also need to install some dependencies before using the utility.

#### Dependencies

The following dependencies are required:

```
libmagic
poppler-utils
tesseract-ocr
libxml2-dev
libxslt1-dev
git
```

You can install these dependencies on macOS by running:

```
brew install libmagic poppler tesseract libxml2 libxslt git
```

### Installation

* Clone this repository:

```
git clone https://github.com/<username>/<repository_name>.git
```

* Install the required Python packages:

```
cd <repository_name>
pip install -r requirements.txt
```

* Download the required NLTK data:

```
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```

## Configuration

Before you can use the utility, you need to set up the configuration file. The configuration file is a YAML file that contains the following options:

* openai_api_key: Your OpenAI API key.
* pinecone_api_key: Your Pinecone API key.
* pinecone_api_env: The environment to use for Pinecone.
* data_directory: The directory where your local data sources are located.
* data_files_glob: A glob pattern that specifies which files in data_directory to use as data sources.
* webpages: A list of URLs of webpages to use as data sources.
* tone: The tone to use for the chatbot's responses (e.g., "formal", "informal", "friendly", etc.).
* persona: The persona to use for the chatbot.
* You can copy the config.example.yaml file to config.yaml and modify the options as needed.

## Usage

To start the chatbot, run:

```
python app.py
```

This will start the chatbot on port 5000.

To use the chatbot, send a POST request to http://localhost:5000/api/chat with a JSON payload containing the question to ask, like this:

```
curl -X POST \
  http://localhost:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the capital of France?"}'
```

This will return a JSON response containing the chatbot's answer to the question:

```
{"response": "The capital of France is Paris."}
```

## Contributing

If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
