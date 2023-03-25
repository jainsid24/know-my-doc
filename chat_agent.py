from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from flask import request, jsonify, render_template
from textblob import TextBlob
import webbrowser
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
os.environ["LANGCHAIN_HANDLER"] = "langchain"
# define a function to calculate the fibonacci number
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)

# define a function which sorts the input string alphabetically
def sort_string(string):
    return ''.join(sorted(string))

# define a function to turn a word in to an encrypted word
def encrypt_word(word):
    encrypted_word = ""
    for letter in word:
        encrypted_word += chr(ord(letter) + 1)
    return encrypted_word

# define a function to decrypt an encrypted word
def decrypt_word(word):
    decrypted_word = ""
    for letter in word:
        decrypted_word += chr(ord(letter) - 1)
    return decrypted_word

# Define a function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Define a function to open a google search for a given topic
def search_google(topic):
    webbrowser.open("https://www.google.com/search?q=" + topic)
    return "Opening Google Search for " + topic 

llm=ChatOpenAI(temperature=0)

# define a function to write your response to the output file
def write_to_file(response):
    with open("output.txt", "w") as f:
        f.write(response)
    return "Successfully wrote to file output.txt"

tools = [
    Tool(
        name = "Fibonacci",
        func = lambda n: str(fib(int(n))),
        description = "use when you want to calculate the nth fibanacci number"
    ),
    Tool(
        name = "Sort String",
        func = lambda string: sort_string(string),
        description = "use when you want to sort a string alphabetically"
    ),
    Tool(
        name = "Encrypt Word",
        func = lambda word: encrypt_word(word),
        description = "use when you want to encrypt a word"
    ),
    Tool(
        name = "Decrypt Word",
        func = lambda word: decrypt_word(word),
        description = "use when you want to decrypt a word"
    ),
    Tool(
        name = "Write to output file",
        func = lambda response: write_to_file(response),
        description = "use when you want to write your response to the output file"
    ),
    Tool(
        name = "Analyze Sentiment",
        func = lambda text: analyze_sentiment(text),
        description = "use when you want to analyze the sentiment of a text"
    ),
    Tool(
        name = "Search Google",
        func = lambda topic: search_google(topic),
        description = "use when you want to search google for a topic"
    )
]
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


def index():
    return render_template("index.html")

def chat():
    try:
        # Get the question from the request
        question = request.json["question"]

        response = agent.run(question)
        print("Response : {}",format(response))
        
        # Increment message counter
        session_counter = request.cookies.get('session_counter')
        if session_counter is None:
            session_counter = 0
        else:
            session_counter = int(session_counter) + 1

        # Check if it's time to flush memory
        # if session_counter % 10 == 0:
        #     agent.memory.clear()

        # Set the session counter cookie
        resp = jsonify({"response": response})
        resp.set_cookie('session_counter', str(session_counter))

        # Return the response as JSON with the session counter cookie
        return resp

    except Exception as e:
        # Log the error and return an error response
        logger.error(f"Error while processing request: {e}")
        return jsonify({"error": "Unable to process the request."}), 500