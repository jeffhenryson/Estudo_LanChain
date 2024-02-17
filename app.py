from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(model_name="gpt-4", temperature=0.9)

loader = PyMuPDFLoader(r"C:\Users\jeffh\OneDrive\Desktop\Vs_code\Projetos\Estudo_LanChain\B450M Steel Legend.pdf")

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
)

# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=200)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always translate your answer to english.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
)

result = rag_chain.invoke("Para fazer um download com um pendriver bootavel. ele tem que ser MBR ou GPT??")

print(result)