from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import pyttsx3
import speech_recognition as sr
import re

# 初始化語音引擎
engine = pyttsx3.init()

def speak(text):
    """文字轉語音輸出"""
    engine.say(text)
    engine.runAndWait()

def load_document(file_path):
    """根據檔案類型載入文件"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path=file_path, extract_images=False)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path=file_path, encoding="utf-8")
    else:
        raise ValueError("目前僅支持 PDF 或 TXT 文件格式")
    return loader.load()

def initialize_knowledge_base(file_path):
    """初始化知識庫"""
    try:
        data = load_document(file_path)
        print("文件加載成功！")
    except Exception as e:
        print(f"文件加載失敗：{e}")
        return None

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    try:
        embeddings = OpenAIEmbeddings(openai_api_key="your_openai_api_key")
    except Exception as e:
        print(f"初始化 OpenAIEmbeddings 時發生錯誤：{e}")
        return None

    knowledge_base = FAISS.from_documents(docs, embeddings)
    return knowledge_base

def get_voice_input():
    """使用語音識別進行用戶輸入"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請說話...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=10)
                text = recognizer.recognize_google(audio, language="zh-TW")
                print(f"您說了：{text}")
                return text
            except sr.UnknownValueError:
                speak("抱歉，我無法辨識您的語音。請再試一次。")
            except sr.RequestError:
                speak("語音服務出現問題，請稍後再試。")
                return None

def calculate_bmi(weight, height):
    """計算 BMI 並返回分類"""
    bmi = weight / (height / 100) ** 2
    if bmi < 18.5:
        return bmi, "過輕"
    elif 18.5 <= bmi < 24.9:
        return bmi, "正常"
    elif 25 <= bmi < 29.9:
        return bmi, "過重"
    else:
        return bmi, "肥胖"

def get_fitness_plan(goal, bmi_category, retriever):
    """生成健身計畫"""
    llm = OllamaLLM(model="llama3.2")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    query = f"""
    基於以下資訊生成個性化的健身計畫：
    - 健身目標：{goal}
    - BMI 分類：{bmi_category}

    請根據背景知識給出專業建議，包括鍛煉計畫、推薦的動作，以及飲食建議。
    """
    response = qa_chain.run(query)
    return response

def main():
    """主程式"""
    speak("歡迎使用個性化健身計畫生成器。請提供您的體重和身高。")
    
    # 初始化知識庫
    file_path = 'fitness_knowledge.txt'  # 或 'rag1.pdf'
    knowledge_base = initialize_knowledge_base(file_path)
    if not knowledge_base:
        speak("知識庫初始化失敗，請確認文件是否存在。")
        return
    retriever = knowledge_base.as_retriever()

    # 獲取體重
    speak("請告訴我您的體重，單位是公斤。")
    weight_input = get_voice_input()
    try:
        weight = float(weight_input)
    except ValueError:
        speak("無法辨識體重，請輸入數字。")
        return

    # 獲取身高
    speak("請告訴我您的身高，單位是公分。")
    height_input = get_voice_input()
    try:
        height = float(height_input)
    except ValueError:
        speak("無法辨識身高，請輸入數字。")
        return

    # 計算 BMI 並提供分類
    bmi, bmi_category = calculate_bmi(weight, height)
    bmi_suggestion = f"您的 BMI 是 {bmi:.1f}，屬於{bmi_category}範圍。"
    print(bmi_suggestion)
    speak(bmi_suggestion)

    # 獲取健身目標
    speak("請告訴我您的健身目標，例如增肌、減脂或提升耐力。")
    goal = get_voice_input()

    # 生成健身計畫
    fitness_plan = get_fitness_plan(goal, bmi_category, retriever)
    speak("根據您的需求，我為您生成了以下健身計畫：")
    print(fitness_plan)
    speak(fitness_plan)

if __name__ == "__main__":
    main()
