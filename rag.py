import speech_recognition as sr
import pyttsx3
from langchain_ollama import OllamaLLM  # 使用 OllamaLLM
from langchain.prompts import PromptTemplate


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import re

# 初始化文字轉語音引擎
engine = pyttsx3.init()

def speak(text):
    """將文字轉換為語音"""
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    """使用語音轉文字進行用戶輸入，無限重試直到成功"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請說話...")
        while True:  # 持續重試直到成功
            try:
                audio = recognizer.listen(source, timeout=10)
                text = recognizer.recognize_google(audio, language="zh-TW")
                print(f"您說了：{text}")
                return text
            except sr.UnknownValueError:
                speak("抱歉，我無法辨識您的語音。請再試一次。")
            except sr.RequestError:
                speak("語音服務出現問題。請稍後再試。")
                return None

def get_chinese_input():
    """使用語音轉文字進行用戶輸入，無限重試直到成功，且只接受中文輸入"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請說話...")
        while True:  # 持續重試直到成功
            try:
                audio = recognizer.listen(source, timeout=10)
                text = recognizer.recognize_google(audio, language="zh-TW")  # 預設為中文
                print(f"您說了：{text}")

                # 檢查是否只有中文
                if re.match(r'^[\u4e00-\u9fa5]+$', text):  # 正規表達式檢查是否只有中文字符
                    return text
                else:
                    speak("請確保您只說中文。")
                    print("請確保您只說中文。")
                    continue  # 重新要求語音輸入
            except sr.UnknownValueError:
                speak("抱歉，我無法辨識您的語音。請再試一次。")
            except sr.RequestError:
                speak("語音服務出現問題。請稍後再試。")
                return None

def calculate_bmi(weight, height):
    """計算 BMI 並給出建議"""
    bmi = weight / (height / 100) ** 2
    if bmi < 18.5:
        return bmi, "過輕"
    elif 18.5 <= bmi < 24.9:
        return bmi, "正常"
    elif 25 <= bmi < 29.9:
        return bmi, "過重"
    else:
        return bmi, "肥胖"

def initialize_knowledge_base(file_path):
    """初始化知識庫"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
    except FileNotFoundError:
        print("檔案不存在，請檢查路徑是否正確。")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(data)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(docs, embeddings)
    return knowledge_base

def get_fitness_plan(goal, bmi_category, retriever):
    """使用 RAG + OllamaLLM 生成個性化健身計畫"""
    llm = OllamaLLM(model="llama3.2")  # 使用 OllamaLLM
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
    """主程式邏輯"""
    speak("歡迎使用個性化健身計畫生成器。請提供您的體重和身高。")

    # 初始化知識庫
    knowledge_base = initialize_knowledge_base('fitness_knowledge.txt')
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

    # 計算 BMI 並給建議
    bmi, bmi_category = calculate_bmi(weight, height)
    bmi_suggestion = f"您的 BMI 是 {bmi:.1f}，屬於{bmi_category}範圍。"
    print(bmi_suggestion)
    speak(bmi_suggestion)

    # 獲取健身目標
    speak("請告訴我您的健身目標，例如增肌、減脂或提升耐力。")
    goal = get_chinese_input()

    # 生成健身計畫
    fitness_plan = get_fitness_plan(goal, bmi_category, retriever)
    speak("根據您的需求，我為您生成了以下健身計畫：")
    print(fitness_plan)
    speak(fitness_plan)

if __name__ == "__main__":
    main()
