import os
import groq

# Set up the Groq API key (replace with your actual API key)
GROQ_API_KEY = "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"
client = groq.Client(api_key=GROQ_API_KEY)

# Function to get medical advice from Llama 3
def get_recommendation(diagnosis):
    prompt = f"Provide treatment recommendations for the skin disease: {diagnosis}. Include home remedies and when to see a doctor."

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "You are a medical assistant."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
