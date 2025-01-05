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
from langchain_chroma.vectorstores import Chroma
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
        return f"Your BMI is {bmi:.1f}, which is underweight. Consider a muscle gain and balanced diet plan."
    elif 18.5 <= bmi < 24.9:
        return f"Your BMI is {bmi:.1f}, which is normal. Set further fitness goals!"
    elif 25 <= bmi < 29.9:
        return f"Your BMI is {bmi:.1f}, which is overweight. Consider a weight loss plan and dietary control."
    else:
        return f"Your BMI is {bmi:.1f}, which is obese. Please consult with a professional for a health strategy."

def get_fitness_plan(goal, weight, height):
    """Generate a personalized fitness plan using LLM."""
    llm = OllamaLLM(model="llama3.2")
    template = """
    Generate a personalized fitness plan based on the following information:
    - Weight: {weight} kg
    - Height: {height} cm
    - Goal: {goal}

    Provide professional advice, including workout plans, recommended exercises, and dietary suggestions.
    """
    prompt = PromptTemplate(input_variables=["goal", "weight", "height"], template=template)
    response = llm.invoke(prompt.format(goal=goal, weight=weight, height=height))
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

def create_or_load_vector_db(chunks, persist_directory='db'):
    """Create or load vector database."""
    embeddings_model = OllamaEmbeddings(model="llama3.2")
    if os.path.exists(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
    else:
        db = Chroma.from_documents(
            collection_name="fitness_plans",
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_directory
        )
    return db

def retrieve_best_match(query, db, chat_model):
    """Retrieve the most relevant document for a query."""
    response = db.query(llm=chat_model, question=query)
    return response

def main():
    """Main program logic."""
    speak("Welcome to the Personalized Fitness Plan Generator.")
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

    bmi_suggestion = calculate_bmi(weight, height)
    speak(bmi_suggestion)

    speak("Please tell me your fitness goal, such as muscle gain, weight loss, or endurance improvement.")
    goal = get_chinese_input()

    fitness_plan = get_fitness_plan(goal, weight, height)
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
