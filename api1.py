import requests
import json
import time
import schedule

# Hugging Face API details
API_TOKEN = ""
MODEL = "deepset/roberta-base-squad2"  # Question-answering model
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

# Headers for authentication
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Function to query Hugging Face API
def query_hugging_face(query, context="France is a country in Europe. Its capital is Paris."):
    payload = {
        "inputs": {
            "question": query,
            "context": context
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise error for bad status
        result = response.json()
        
        # Extract the answer
        answer = result.get("answer", "No answer found").strip()
        # End response at first newline (\n) if present
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()
        
        return {"query": query, "response": answer}
    except Exception as e:
        return {"query": query, "error": str(e)}

# Function to format and print the result
def process_query(query):
    result = query_hugging_face(query)
    formatted_output = {
        "query": result["query"],
        "response": result.get("response", f"Error: {result.get('error', 'Unknown')}")
    }
    print(json.dumps(formatted_output, indent=2))
    return formatted_output

# Background job for periodic execution
def background_job():
    query = "What is the capital of France?"
    process_query(query)

# Example: Run once
if __name__ == "__main__":
    # Single execution
    process_query("What is the capital of India?")
    
    # Uncomment below to run in background every 5 minutes
    # schedule.every(5).minutes.do(background_job)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)