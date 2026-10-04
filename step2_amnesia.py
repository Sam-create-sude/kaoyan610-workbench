# import os 
# from dotenv import load_dotenv 
# from openai import OpenAI

# load_dotenv()

# client=OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )

# responce1=client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role":"user","content":"我叫小明，请记住"},
#     ],
# )

# print("第一次调用",responce1.choices[0].message.content)

# responce2=client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role":"user","content":"我叫什么名字？"},
#     ],
# )
# print("第二次调用:",responce2.choices[0].message.content)

import os 
from dotenv import load_dotenv 
from openai import OpenAI

load_dotenv()

client=OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

messages=[
    {"role":"system","content":"你是一个美女，回答简洁。"}
]

while True:
   user_input=input("\n你：")

   if user_input.strip() in ("exit","quit","退出"):
      print("对话结束。")
      break
   messages.append({"role":"user","content":user_input}) 

   response=client.chat.completions.create(
      model="deepseek-chat",
      messages=messages,
   )

   reply=response.choices[0].message.content
   messages.append({"role":"assistant","content":reply})

   print("助手：", reply)
   print(f"  [历史 {len(messages)} 条 | prompt_tokens={response.usage.prompt_tokens} | 缓存命中={response.usage.prompt_cache_hit_tokens}]")