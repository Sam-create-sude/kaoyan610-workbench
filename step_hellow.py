import os 
from dotenv import load_dotenv 
from openai import OpenAI

load_dotenv()

client=OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

response=client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role":"system","content":"你是一个专业的产品经理助手，回答简洁。"},
        {"role":"user","content":"用三句话解释以下什么是RAG"},
    ],
)
print(response.choices[0].message.content)