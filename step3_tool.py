import os
import json
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
import urllib.parse
import urllib.request

load_dotenv()
client=OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# def get_weather(city):
#     return f"{city}今天天气晴朗，气温22到28摄氏度。"

def get_weather(city):
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        cur = data["current_condition"][0]
        desc = cur["weatherDesc"][0]["value"]
        return f"{city}当前{desc}，气温 {cur['temp_C']} 摄氏度，体感 {cur['FeelsLikeC']} 摄氏度，湿度 {cur['humidity']}%。"

    except Exception as e:
        return f"天气查询失败（{type(e).__name__}），请稍后再试。"


tools=[
    {
        "type":"function",
        "function":{
            "name":"get_current_time",
            "description":"获取当前的日期和时间。当用户询问现在几点、今天几号、当前日期时使用。",
            "parameters":{
                "type":"object",
                "properties":{},
            },
        },
    },
    
    {
        "type":"function",
        "function":{
            "name":"get_weather",
            "description":"查询指定城市的天气。当用户询问某个城市天气、气温、下雨时使用。",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"要查询天气的城市名称，例如：北京、上海、深圳",
                    }
                },
                "required":["city"]
            },
        },
    }
]

messages=[
    {"role":"system","content":"你是一个专业的产品经理助手，回答简洁。"},
    {"role":"user","content":"杭州今天天气怎么样？"},
]

response=client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
)

msg=response.choices[0].message

print("模型返回的内容：",msg.content)
print("模型申请的调用：",msg.tool_calls)

if msg.tool_calls:
    tool_call=msg.tool_calls[0]
    print("模型要调用的函数名：",tool_call.function.name)
    print("模型的参数：",tool_call.function.arguments)
    if tool_call.function.name=="get_weather":
        args=json.loads(tool_call.function.arguments)
        result=get_weather(args["city"])
    elif tool_call.function.name=="get_current_time":
        result=get_current_time()
    else:
        result="没有这个工具"
    
    print("你的程序真执行，结果是：",result)

    messages.append(msg)

    messages.append(
        {
            "role":"tool",
            "tool_call_id":tool_call.id,
            "content":result,
        }
    )

    final=client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
    )
    print("最终回答：",final.choices[0].message.content)