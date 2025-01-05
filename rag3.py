import os
import re
import numpy as np
import speech_recognition as sr
import pyttsx3
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def speak(text):
    """文字轉語音"""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    """透過語音獲取使用者輸入，並轉換為文字"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請說話...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=10)
                text = recognizer.recognize_google(audio, language="zh-TW")
                print(f"你說的是: {text}")
                return text
            except sr.UnknownValueError:
                speak("抱歉，我沒聽清楚。請再試一次。")
            except sr.RequestError:
                speak("語音服務目前無法使用，請稍後再試。")
                return None

def get_chinese_input():
    """確保僅接受中文語音輸入"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請說中文...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=10)
                text = recognizer.recognize_google(audio, language="zh-TW")
                if re.match(r'^[\u4e00-\u9fa5]+$', text):
                    return text
                else:
                    speak("請確保您只說中文。")
            except sr.UnknownValueError:
                speak("抱歉，我無法辨識您的語音。請再試一次。")
            except sr.RequestError:
                speak("語音服務目前無法使用，請稍後再試。")
                return None

def calculate_bmi(weight, height):
    """計算 BMI 並提供建議"""
    bmi = weight / (height / 100) ** 2
    if bmi < 18.5:
        return f"您的 BMI 是 {bmi:.1f}，屬於過輕。建議增加肌肉量並維持均衡飲食。", bmi
    elif 18.5 <= bmi < 24.9:
        return f"您的 BMI 是 {bmi:.1f}，屬於正常範圍。可設定進一步的健身目標！", bmi
    elif 25 <= bmi < 29.9:
        return f"您的 BMI 是 {bmi:.1f}，屬於過重範圍。建議制定減重計劃並控制飲食。", bmi
    else:
        return f"您的 BMI 是 {bmi:.1f}，屬於肥胖範圍。請諮詢專業人士制定健康策略。", bmi

def get_fitness_plan_with_rag(goal, weight, height, bmi, db, chat_model):
    """使用 RAG 和 LLM 生成個人化健身計劃"""
    query = f"針對體重 {weight} 公斤，身高 {height} 公分，BMI {bmi:.1f}，目標為 {goal} 的人生成健身計劃。"
    retrieved_info = retrieve_best_match(query, db, chat_model)
    fitness_plan = f"檢索到的相關資訊：{retrieved_info}\n\n根據目標的計劃："

    llm = OllamaLLM(model="llama3.2")
    template = """
    根據以下檢索到的資訊與用戶數據，生成個人化健身計劃：
    {retrieved_info}
    
    - 體重: {weight} 公斤
    - 身高: {height} 公分
    - BMI: {bmi:.1f}
    - 目標: {goal}

    提供專業建議，包括運動計劃、推薦的鍛鍊方式及飲食建議。
    """
    prompt = PromptTemplate(input_variables=["retrieved_info", "goal", "weight", "height", "bmi"], template=template)
    response = llm.invoke(prompt.format(retrieved_info=retrieved_info, goal=goal, weight=weight, height=height, bmi=bmi))
    return fitness_plan + response

def load_and_process_documents(file_path):
    """載入、切分並處理文件以嵌入"""
    loader = TextLoader(file_path=file_path, encoding="utf8")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=10,
        separators=["\n\n", "\n", " ", "", "。", "，"]
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

def create_or_load_vector_db(chunks):
    """建立或載入向量資料庫"""
    embeddings_model = OllamaEmbeddings(model="llama3.2")  # 初始化嵌入模型
    # 使用 FAISS 建立向量資料庫
    db = FAISS.from_documents(chunks, embeddings_model)
    return db

def retrieve_best_match(query, db, chat_model):
    """檢索與查詢最相關的文件"""
    response = db.similarity_search(query, k=1)
    return response[0].page_content if response else "無相關資訊"

def main():
    """主程式邏輯"""
    speak("歡迎使用個人化健身計劃生成器。")
    #speak("現在，讓我們透過文件檢索增強您的體驗。")

    file_path = 'fitness_knowledge.txt'  # 替換為您的文件路徑
    chunks = load_and_process_documents(file_path)
    db = create_or_load_vector_db(chunks)

    chat_model = ChatOllama(model="llama3.2")

    speak("請提供您的體重（公斤）。")
    weight_input = get_voice_input()
    try:
        weight = float(weight_input)
    except ValueError:
        speak("體重輸入無效。請輸入數值。")
        return

    speak("現在，請提供您的身高（公分）。")
    height_input = get_voice_input()
    try:
        height = float(height_input)
    except ValueError:
        speak("身高輸入無效。請輸入數值。")
        return

    bmi_suggestion, bmi = calculate_bmi(weight, height)
    speak(bmi_suggestion)

    speak("請告訴我您的健身目標，例如增肌、減重或耐力提升。")
    goal = get_chinese_input()

    fitness_plan = get_fitness_plan_with_rag(goal, weight, height, bmi, db, chat_model)
    speak("以下是您的個人化健身計劃：")
    print(fitness_plan)

if __name__ == "__main__":
    main()
