import os
import re
import dotenv
from openai import OpenAI

from text_prob import match_confidence

dotenv.load_dotenv()

# --- 1. Setup: Load Data and Configure Client ---

try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
except Exception as e:
    print("Error initializing OpenAI client.")
    exit()

# --- 2. Design Questions ---

questions = {
    "Lookups": [
        "How many transactions were paid for with Cash? List the transaction IDs.",
        "What is the category of the product with the highest unit price in the dataset?"
    ],
    "Fabricated": [
        "Was transaction 3150cd73-1605-45c1-b3e1-4622f83762cb returned by the customer?",
    ],
    "Generate": [
        "Generate another row for the dataset. Make sure all fields are real and valid."
    ]
}

# --- 3. Core Analysis Function ---

def analyze_question(context: str, question: str):
    system_prompt = f"""
You are a data analysis assistant. You will be given a block of text containing CSV data.
Your task is to answer questions based on the information present in the provided data.

Provide a concise, direct answer to the question.
"""

    user_prompt = f"Here is the data:\n\n---\n{context}\n---\n\nQuestion: {question}"

    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_ID", "openai/gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            logprobs=True,  # Request log probabilities
        )

        # --- 4. Parse the Response and Analyze Logprobs ---
        print(f"❓ Question:\n{question}\n")
        if response.choices[0].message.reasoning:
            print(f"🤖 LLM Reasoning:\n{response.choices[0].message.reasoning}\n")
        
        response_text = response.choices[0].message.content
        print(f"🤖 LLM Response:\n{response_text}\n")

        uuid_pattern = r'\b[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\b'
        uuid_matches = list(re.finditer(uuid_pattern, response_text, re.IGNORECASE))

        uuid_confidence = match_confidence(uuid_matches, response.choices[0].logprobs)
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.99))

        for text, confidence in uuid_confidence:
            print(f"🔍 UUID: {text}, Confidence: {confidence}, Grounded: {confidence >= threshold}\n")
        
        print("=" * 100 + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- 0. Run the Experiment ---

if __name__ == "__main__":
    with open("data.csv", "r") as f:
        csv_data = f.read()
        for _, q_list in questions.items():
            for q in q_list:
                analyze_question(context=csv_data, question=q)
