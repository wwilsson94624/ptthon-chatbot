import speech_recognition as sr
import pyttsx3
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

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

def initialize_knowledge_base(data):
    """初始化知識庫"""
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(data)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(docs, embeddings)
    return knowledge_base

def get_fitness_plan(goal, bmi_category, retriever):
    """使用 RAG + HuggingFaceHub 生成個性化健身計畫"""
    # 使用 Hugging Face Hub 模型
    llm = HuggingFaceHub(
        repo_id="Qwen/Qwen-7b",  # 替換為 Hugging Face 的模型
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
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
    file_content = """
    健身相關的基本知識：
1. 增肌：
   - 飲食：每天攝取比維持體重更多的熱量，並保證每公斤體重攝取1.6-2.2克蛋白質。同時保持碳水化合物攝取以支持訓練。
   - 訓練：以重量訓練為核心，專注於多關節動作如深蹲、硬拉、臥推、引體向上，並逐步增加訓練重量。
   - 頻率：每週訓練3-5天，分部位進行訓練，如胸、背、腿、肩等。

2. 減脂：
   - 飲食：創造每日500-1000卡的熱量赤字，重點在於減少高糖和高脂肪食物，並增加蔬菜、水果和高纖維食品。
   - 訓練：結合有氧運動（如跑步、游泳）和抗阻力訓練。高強度間歇訓練（HIIT）也有助於燃燒脂肪。
   - 頻率：每週5-6天運動，包含3-4天有氧運動和2-3天力量訓練。

3. 提升耐力：
   - 訓練方式：長時間的低強度有氧運動（如慢跑、長距離自行車騎行）或階段性耐力訓練（如高山徒步）。
   - 增加訓練時間和距離，但需逐步提升以避免過度疲勞。
   - 飲食：高碳水化合物飲食以補充能量，運動後補充電解質。

4. 健康飲食原則：
   - 多樣化：均衡攝取蛋白質、碳水化合物和脂肪。
   - 控制份量：避免暴飲暴食，每餐細嚼慢嚥。
   - 補水：每天至少喝2-3升水，運動期間適量補充。
   - 健康脂肪來源：包括魚類、堅果、亞麻籽油。

5. 休息與恢復：
   - 睡眠：每晚至少7-9小時的高質量睡眠，有助於肌肉生長和恢復。
   - 主動恢復：低強度活動如散步或瑜伽可以促進血液循環。
   - 按摩與放鬆：通過按摩槍、泡沫軸或熱水浴放鬆肌肉。

進階健身技巧：
1. 超負荷原則：逐漸增加訓練強度、重量或次數。
2. 紀錄與追蹤：記錄每日飲食和訓練內容，定期檢視進展。
3. 結合不同運動形式：例如重量訓練與有氧運動相結合，以達到最佳效果。

常見健身動作與技巧：
1. 下半身：
   - 深蹲：鍛煉大腿、臀部和核心。
   - 硬拉：加強整體力量，重點在下背部。
   - 弓步：改善平衡與腿部力量。
2. 上半身：
   - 引體向上：鍛煉背部和肱二頭肌。
   - 啞鈴肩推：增強肩部力量與穩定性。
   - 臥推：針對胸肌和肱三頭肌。
3. 核心：
   - 平板支撐：增強核心穩定性。
   - 側平板支撐：專注於側腹肌。
   - 捲腹：提高腹部肌肉的耐力與力量。

其他：
1. 運動風險管理：避免超負荷訓練，保持正確動作姿勢。
2. 設定目標：短期（如減脂1公斤）與長期目標（如完成馬拉松）。
3. 適應性訓練：根據體能和時間調整訓練內容。
1. 體重分類 (BMI)
過輕 (BMI < 18.5)
這類型的人體重低於正常範圍，可能面臨營養不足的風險。過輕可能影響免疫系統，導致疲勞，甚至增加骨折風險。
建議：

增加熱量攝入，特別是來自營養豐富的食物（如健康脂肪、蛋白質和複合碳水化合物）。
增加力量訓練來增強肌肉量。
每天攝取足夠的蛋白質，以支持肌肉合成和健康的免疫系統。
定期監測體重變化，確保逐步增重而非過快增重。
正常範圍 (BMI 18.5 - 24.9)
體重在健康範圍內，這是理想的狀態。這些人通常有較低的心血管疾病風險，且新陳代謝較為正常。
建議：

維持當前體重，專注於平衡飲食和運動。
進行力量訓練來增加肌肉質量，提高新陳代謝。
增加有氧運動來保持心肺健康。
保持穩定的生活方式，避免過度放縱飲食。
過重 (BMI 25 - 29.9)
體脂肪比正常範圍高，可能會增加患上高血壓、糖尿病和心臟病的風險。過重的人可能會發現運動時較為吃力。
建議：

控制飲食，減少高熱量、高脂肪的食物攝入。
增加有氧運動的時間和強度（如跑步、游泳、快走等），促進脂肪燃燒。
增加力量訓練，有助於增加基礎代謝率，減少體脂肪。
善用健康飲食和飲水習慣，減少不必要的熱量攝入。
肥胖 (BMI ≥ 30)
肥胖會顯著增加慢性疾病風險，包括糖尿病、心血管疾病、睡眠呼吸暫停症等。肥胖的人常會有活動障礙和精神壓力。
建議：

需要採取綜合的減重方案，飲食控制是最關鍵的一環，必須創造熱量赤字。
建議進行有氧運動，如游泳、騎自行車等，並提高運動強度來增加卡路里消耗。
增加力量訓練，避免肌肉流失並提高基礎代謝率。
若必要，可以尋求專業醫生或營養師的指導，進行健康的減重計劃。
2. 健身目標分類
減脂 (Weight Loss / Fat Loss)
減脂是許多體重過重或肥胖人士的首要目標。減脂的關鍵是創造卡路里赤字，這通常通過飲食控制和增加運動來實現。
建議：

每週至少進行150分鐘的中等強度有氧運動或75分鐘的高強度有氧運動。
增加蛋白質攝取，以維持肌肉質量並減少脂肪。
減少精緻糖和高脂食物，選擇低卡高纖維的食物。
配合定期的力量訓練來提高基礎代謝率。
增肌 (Muscle Gain / Hypertrophy)
增肌適合於過輕或正常體重的人，他們希望增強肌肉質量、力量和耐力。
建議：

進行每週至少三次的力量訓練，重點是大肌群的訓練（如深蹲、硬拉、推舉等）。
增加蛋白質攝取（每日每公斤體重1.6–2.2克），以支持肌肉修復和生長。
進行適量的碳水化合物攝取來提供能量，支援高強度訓練。
充足的休息和睡眠以促進肌肉恢復和生長。
提高耐力 (Endurance Improvement)
耐力訓練專注於提升心肺功能和持久力。適合有心血管健康目標的人群。
建議：

進行長時間的有氧運動（如跑步、游泳、騎自行車等），並逐步增加運動的時長和強度。
每週至少150分鐘的中等強度有氧運動或75分鐘的高強度有氧運動。
增加運動的變化，防止單一運動導致過度疲勞或受傷。
保持良好的飲食習慣，補充足夠的碳水化合物和水分。
維持體重 (Weight Maintenance)
這是對健康體重人群的一個目標，維持理想體重，避免增重或減重過多。
建議：

持續平衡飲食，避免過多攝取過度加工或高熱量食物。
保持定期的運動習慣，包括有氧運動和力量訓練。
定期監測體重，調整飲食和運動計劃以應對生活中的變化。

    """
    # 初始化知識庫
    knowledge_base = initialize_knowledge_base("fitness_knowledge.txt")
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
