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
    "Easy (Direct Lookup - Grounded)": [
        "What was the total amount for transaction 918e4a7a-29d0-4565-99d9-b6abf8e54be5?",
    ],
    "Medium (Simple - Grounded)": [
        "How many transactions were paid for with Cash?",
    ],
    "Hard (Multi-step - Grounded)": [
        "What is the category of the product with the highest unit price in the dataset?",
    ],
    "Impossible (Fabricated)": [
        "Was transaction fff306b2-c7d8-4b79-8abe-e77e93172e08 returned by the customer?",
    ],
    "Generate": [
        "Generate another sample row for the dataset. Make sure all the fields are valid."
    ]
}

# --- 3. Core Analysis Function ---

def analyze_question(context: str, question: str):
    """
    Sends a question to the LLM, gets the response with logprobs, and analyzes the
    confidence of the grounding tag.
    """
    system_prompt = f"""
You are a data analysis assistant. You will be given a block of text containing CSV data.
Your task is to answer questions based *only* on the information present in the provided data.
Do not make assumptions or use external knowledge.

Provide a concise, direct answer to the question.
"""

    user_prompt = f"Here is the data:\n\n---\n{context}\n---\n\nQuestion: {question}"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            logprobs=True,      # Request log probabilities
            top_logprobs=5
        )

        # --- 4. Parse the Response and Analyze Logprobs ---
        print(f"❓ Question:\n{question}\n")
        full_response_text = response.choices[0].message.content
        print(f"🤖 LLM Response:\n{full_response_text}\n")

        uuid_pattern = r'\b[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\b'
        uuid_matches = list(re.finditer(uuid_pattern, full_response_text, re.IGNORECASE))

        uuid_confidence = match_confidence(uuid_matches, response.choices[0].logprobs)
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))

        for text, confidence in uuid_confidence:
            print(f"🔍 UUID: {text}, Confidence: {confidence}, Grounded: {confidence >= threshold}\n")
        
        print("=" * 100 + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- 5. Run the Experiment ---

if __name__ == "__main__":
    with open("data.csv", "r") as f:
        csv_data = f.read()
        for _, q_list in questions.items():
            for q in q_list:
                analyze_question(context=csv_data, question=q)
