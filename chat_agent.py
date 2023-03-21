from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA, LLMMathChain, SerpAPIWrapper
from langchain.document_loaders import DirectoryLoader, WebBaseLoader
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from flask import request, jsonify, render_template
import os
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

llm = OpenAI(temperature=0)

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

# Create vectorstore
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
ruff_texts = text_splitter.split_documents(result)
# ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")
# ruff = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=ruff_db)


embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(ruff_texts, embeddings, collection_name="state-of-union")
state_of_union = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch)
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
tools = [
    Tool(
        name = "State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question."
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


def index():
    return render_template("index.html")

def chat():
    try:
        # Get the question from the request
        question = request.json["question"]

        response = agent.run(question)
        
        # Increment message counter
        session_counter = request.cookies.get('session_counter')
        if session_counter is None:
            session_counter = 0
        else:
            session_counter = int(session_counter) + 1

        # Check if it's time to flush memory
        # if session_counter % 10 == 0:
            # agent.memory.clear()

        # Set the session counter cookie
        resp = jsonify({"response": response})
        resp.set_cookie('session_counter', str(session_counter))

        # Return the response as JSON with the session counter cookie
        return resp

    except Exception as e:
        # Log the error and return an error response
        logger.error(f"Error while processing request: {e}")
        return jsonify({"error": "Unable to process the request."}), 500