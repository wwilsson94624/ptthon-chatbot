import os
import re
import numpy as np
import speech_recognition as sr
import pyttsx3
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def speak(text):
    """Convert text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    """Get user input via voice and return as text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=10)
                text = recognizer.recognize_google(audio, language="zh-TW")
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                speak("Sorry, I didn't catch that. Please try again.")
            except sr.RequestError:
                speak("Voice service is currently unavailable. Please try later.")
                return None

def get_chinese_input():
    """Ensure only Chinese input via voice recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak Chinese...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=10)
                text = recognizer.recognize_google(audio, language="zh-TW")
                if re.match(r'^[\u4e00-\u9fa5]+$', text):
                    return text
                else:
                    speak("Please ensure you are speaking only Chinese.")
            except sr.UnknownValueError:
                speak("Sorry, I couldn't recognize your voice. Please try again.")
            except sr.RequestError:
                speak("Voice service is currently unavailable. Please try later.")
                return None

def calculate_bmi(weight, height):
    """Calculate BMI and provide suggestions."""
    bmi = weight / (height / 100) ** 2
    if bmi < 18.5:
        suggestion = f"Your BMI is {bmi:.1f}, which is underweight. Consider a muscle gain and balanced diet plan."
    elif 18.5 <= bmi < 24.9:
        suggestion = f"Your BMI is {bmi:.1f}, which is normal. Set further fitness goals!"
    elif 25 <= bmi < 29.9:
        suggestion = f"Your BMI is {bmi:.1f}, which is overweight. Consider a weight loss plan and dietary control."
    else:
        suggestion = f"Your BMI is {bmi:.1f}, which is obese. Please consult with a professional for a health strategy."
    return suggestion, bmi

def get_fitness_plan(goal, weight, height, bmi, retrieved_info=""):
    """Generate a personalized fitness plan using LLM."""
    llm = OllamaLLM(model="llama3.2")
    template = """
    Generate a personalized fitness plan based on the following information:
    {retrieved_info}
    
    - Weight: {weight} kg
    - Height: {height} cm
    - BMI: {bmi:.1f}
    - Goal: {goal}

    Provide professional advice, including workout plans, recommended exercises, and dietary suggestions.
    """
    prompt = PromptTemplate(input_variables=["retrieved_info", "weight", "height", "bmi", "goal"], template=template)
    response = llm.invoke(prompt.format(
        retrieved_info=retrieved_info, weight=weight, height=height, bmi=bmi, goal=goal
    ))
    return response

def load_and_process_documents(file_path):
    """Load, split, and process documents for embedding."""
    loader = TextLoader(file_path=file_path, encoding="utf8")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=10,
        separators=["\n\n", "\n", " ", "", "。", "，"]
    )
    chunks = text_splitter.split_documents(pages)
    return chunks
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

def create_or_load_vector_db(chunks, persist_directory='db'):
    """Create or load vector database."""
    # 初始化嵌入模型
    embeddings_model = OllamaEmbeddings(model="llama3.2")
    
    try:
        # 嘗試加載現有數據庫
        if os.path.exists(persist_directory):
            db = FAISS.load_local(persist_directory, embeddings_model)
            print("Loaded existing vector database.")
        else:
            # 創建新的數據庫
            db = FAISS.from_documents(chunks, embeddings_model)
            db.save_local(persist_directory)
            print("Created and saved new vector database.")
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        return None

    return db
def retrieve_best_match(query, db, chat_model):
    """Retrieve the most relevant document for a query."""
    try:
        response = db.query(llm=chat_model, question=query)
        if not response or "result" not in response:
            return "No relevant information found."
        return response["result"]
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return "Error retrieving information."


def main():
    """Main program logic."""
    speak("Welcome to the Personalized Fitness Plan Generator.")
    speak("Please provide your weight in kilograms.")
    speak("Now, let's enhance your experience with document retrieval.")

    file_path = 'fitness_knowledge.txt'  # Replace with the path to your document
    chunks = load_and_process_documents(file_path)
    db = create_or_load_vector_db(chunks)

    chat_model = ChatOllama(model="llama3.2")

    speak("Please provide your weight in kilograms.")
    weight_input = get_voice_input()
    try:
        weight = float(weight_input)
    except ValueError:
        speak("Invalid weight input. Please enter a numeric value.")
        return

    speak("Now, please provide your height in centimeters.")
    height_input = get_voice_input()
    try:
        height = float(height_input)
    except ValueError:
        speak("Invalid height input. Please enter a numeric value.")
        return

    bmi_suggestion, bmi = calculate_bmi(weight, height)
    print(f"BMI Suggestion: {bmi_suggestion}, BMI: {bmi}")
    speak(bmi_suggestion)

    speak("Please tell me your fitness goal, such as muscle gain, weight loss, or endurance improvement.")
    goal = get_chinese_input()

    fitness_plan = get_fitness_plan(goal, weight, height, bmi)

    speak("Here is your personalized fitness plan:")
    print(fitness_plan)

    speak("Now, let's enhance your experience with document retrieval.")

    file_path = 'fitness_knowledge.txt'  # Replace with the path to your document
    chunks = load_and_process_documents(file_path)
    db = create_or_load_vector_db(chunks)

    chat_model = ChatOllama(model="llama3.2")
    query = "Describe Shohei Ohtani's performance in the 2023 WBC in Traditional Chinese."
    response = retrieve_best_match(query, db, chat_model)
    speak("Here is the result of your query:")
    print(response)

if __name__ == "__main__":
    main()