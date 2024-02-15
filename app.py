# Importa as bibliotecas necessárias para interação com o LangChain e a API da OpenAI, além do gerenciamento de variáveis de ambiente.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Carrega as variáveis de ambiente do arquivo .env.
load_dotenv()

# Obtém a chave da API da OpenAI do arquivo .env.
openai_api_key = os.getenv("OPENAI_API_KEY")

# Inicializa o modelo da OpenAI com a chave da API.
llm = ChatOpenAI(openai_api_key)

# Define um template de prompt que será usado para formatar as entradas.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

# Cria uma cadeia combinando o prompt com o modelo LLM, para processamento da entrada.
chain = prompt | llm

# Define a pergunta de entrada.
input_question = {"input": "how can langsmith help with testing?"}
# Invoca a cadeia com a pergunta de entrada e obtém a resposta.
response = chain.invoke(input_question)
# Imprime a resposta.
print(response)
