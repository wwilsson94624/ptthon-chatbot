from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import requests

API_URL = "http://localhost:11434/api/completion"

def query_ollama(prompt):
    payload = {"model": "llama2", "prompt": prompt}
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None
        return response.json()  # 嘗試解析 JSON
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError:
        print("Failed to parse JSON response")
    return None

response = query_ollama("Explain LangChain in simple terms.")
print(response)
from ollama import Ollama

# 初始化 Ollama 模型
ollama_client = Ollama()

# 指定使用的模型 (例如 GPT 模型)
model_name = "gpt3.5"

# 發送查詢
response = ollama_client.generate(
    prompt="請問今天的天氣如何？",
    model=model_name
)

print(response)
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# 使用 LangChain 提供的 LLM 接口
llm = OpenAI(model_name="gpt3.5", client=ollama_client)

# 定義 Prompt Template
template = PromptTemplate(
    input_variables=["question"],
    template="以下是使用者的問題：{question}。請提供詳細解答。"
)

# 建立 LLM Chain
llm_chain = LLMChain(llm=llm, prompt=template)

# 測試模型
question = "如何整合 Ollama 和 LangChain？"
result = llm_chain.run({"question": question})
print(result)

