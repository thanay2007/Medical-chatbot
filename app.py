from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.prompt import system_prompt
from dotenv import load_dotenv
import os

app = Flask(__name__)

# 1. Load API Keys
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 2. Download Embeddings (The "Translator")
print("🧠 Loading Embeddings...")
embeddings = download_hugging_face_embeddings()

# 3. Connect to Existing Index (We don't create it again, just connect)
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 4. Initialize the Brain (Llama 3.1)
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0.5
)

# 5. Define the Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 6. Build the Chain (LCEL - The Modern, Stable Way)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- ROUTES ---

@app.route("/")
def index():
    # This serves your beautiful HTML file
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    # 1. Get the message from the HTML form
    msg = request.form["msg"]
    input_text = msg
    print(f"User Question: {input_text}")
    
    # 2. Ask the AI
    response = rag_chain.invoke(input_text)
    
    # 3. Print result to terminal (for debugging)
    print("Bot Response:", response)
    
    # 4. Send plain text back to the website
    return str(response)

if __name__ == '__main__':
    # '0.0.0.0' allows it to run on localhost correctly
    app.run(host="0.0.0.0", port=8080, debug=True)