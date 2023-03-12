import os
import logging
from flask import Flask, request, jsonify, render_template
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import WebBaseLoader
import yaml

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import nltk

nltk.download("punkt")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

template_dir = os.path.abspath("templates")
app = Flask(__name__, template_folder=template_dir, static_folder="static")

# Load the files
loader = DirectoryLoader(config["data_directory"], glob=config["data_files_glob"])
docs = loader.load()

webpages = config.get("webpages", [])
web_docs = []
for webpage in webpages:
    logger.info(f"Loading data from webpage {webpage}")
    loader = WebBaseLoader(webpage)
    web_docs += loader.load()

result = docs + web_docs

tone = config.get("tone", "default")
persona = config.get("persona", "default")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(result)
embeddings = OpenAIEmbeddings(openai_api_key=config["openai_api_key"])
docsearch = Chroma.from_documents(texts, embeddings)

# Initialize the QA chain
logger.info("Initializing QA chain...")
chain = load_qa_chain(
    OpenAIChat(),
    chain_type="stuff",
    memory=ConversationBufferMemory(memory_key="chat_history", input_key="human_input"),
    prompt=PromptTemplate(
        input_variables=["chat_history", "human_input", "context", "tone", "persona"],
        template="""You are a chatbot who acts like {persona}, having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer only in the {tone} tone. Use only the sources in the document to create a response."

{context}

{chat_history}
Human: {human_input}
Chatbot:""",
    ),
    verbose=False,
)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        # Get the question from the request
        question = request.json["question"]
        documents = docsearch.similarity_search(question, include_metadata=True)

        # Get the bot's response
        response = chain(
            {
                "input_documents": documents,
                "human_input": question,
                "tone": tone,
                "persona": persona,
            },
            return_only_outputs=True,
        )["output_text"]

        # Return the response as JSON
        return jsonify({"response": response})

    except Exception as e:
        # Log the error and return an error response
        logger.error(f"Error while processing request: {e}")
        return jsonify({"error": "Unable to process the request."}), 500


if __name__ == "__main__":
    app.run(debug=True)
