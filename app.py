from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um escritor técnico de classe mundial"),
    ("user", "{input}")
])

chain = prompt | llm

input_question = {"input": "como o LangSmith pode ajudar com testes?"}
response = chain.invoke(input_question)
print(response)
