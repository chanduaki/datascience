## Loading the Environment Variables
import os
from dotenv import load_dotenv
## LangChain
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


# Step 1 : Load the Environment Variables such as API Keys etc
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# Step 2 : Load the Documents such as PDFs ,JSON Fies or Python Files ... etc
loader = PyPDFLoader("ai.pdf")
documents = loader.load()
# print(documents)

# Step 3 : Split the documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
texts = text_splitter.split_documents(documents)
# print(text_splitter)

# Step 4 : Create text embeddings // Choose the text embeddings
embeddings = OpenAIEmbeddings()

# Step 5 : Create a vector store
db = Chroma.from_documents(texts, embeddings)
# print(db)

# Step 6 : Expose this index in a retriever interface
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# Step 7 : Create a RetrievalQA chain to answer questions:
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="map_reduce",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)

# Step 8 : QA Asking
print(qa("Tell me about AI in 2 lines? for a school student"))