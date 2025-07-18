import json
from transformers import AutoModelForCausalLM, AutoTokenizer

t = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")


messages = [
    # 系统提示
    {"role": "system", "content": "你是一个能调用外部 API 的助手。"},
    # 用户发起查询
    {"role": "user", "content": "帮我查一下 7 月底纽约的天气，并预约那两天的餐厅。"},
    # 模型自动调用 get_weather_forecast 函数
    {
        "role": "assistant",
        "content": None,
        "function_call": {
            "name": "get_weather_forecast",
            "arguments": json.dumps({"location": "New York", "date_start": "2025-07-28", "date_end": "2025-07-29"}),
        },
    },
    # 假设函数执行后的返回内容
    {
        "role": "function",
        "name": "get_weather_forecast",
        "content": json.dumps(
            {
                "forecasts": [
                    {"date": "2025-07-28", "temp": "29°C", "condition": "Sunny"},
                    {"date": "2025-07-29", "temp": "27°C", "condition": "Partly Cloudy"},
                ]
            }
        ),
    },
    # 模型基于天气结果继续对话
    {
        "role": "assistant",
        "content": "7月28日纽约晴天，气温29°C；7月29日多云，27°C。你想在哪一天订餐厅？提供餐厅名、时间和人数即可。",
    },
    # 用户指定预订
    {"role": "user", "content": "那帮我预订7月28日晚上7点，四人的意大利餐厅 'Luigi\\'s'."},
    # 模型自动调用 book_restaurant 函数
    {
        "role": "assistant",
        "content": None,
        "function_call": {
            "name": "book_restaurant",
            "arguments": json.dumps(
                {
                    "restaurant_name": "Luigi's",
                    "location": "New York",
                    "date": "2025-07-28",
                    "time": "19:00",
                    "party_size": 4,
                }
            ),
        },
    },
    # 函数执行后的响应
    {
        "role": "function",
        "name": "book_restaurant",
        "content": json.dumps({"success": True, "booking_id": "XYZ123", "details": "Luigi's, 2025-07-28 19:00 for 4"}),
    },
    # 模型确认最终结果
    {
        "role": "assistant",
        "content": "✅ 已为您在 2025‑07‑28 19:00 于 Luigi's 预订 4 人。预订号：XYZ123。如果需要更改或查看别的日子，随时告诉我。",
    },
]

msg = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(msg)
