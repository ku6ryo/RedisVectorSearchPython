import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

embedding_model = "text-embedding-ada-002"
gpt_model = "gpt-3.5-turbo"

def get_embedding(text: str):
    res = openai.Embedding.create(input=text, model=embedding_model)
    embedding = res['data'][0]['embedding']
    return embedding


def chat_completion(messages: list):
    res = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        temperature=0,
    )
    return res["choices"][0]["message"]["content"]