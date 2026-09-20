import json
from dotenv import load_dotenv
from groq import Groq
import os 

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def calculator(expression: str) -> str:
    """Evaluates a basic math expression safely."""
    try:
        # only allow safe characters — never eval() raw model output blindly in production
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """Returns fake weather data for a city (mocked for testing tool selection)."""
    fake_data = {
        "bangalore": "28°C, partly cloudy",
        "mumbai": "31°C, humid",
        "delhi": "36°C, clear skies",
    }
    key = city.strip().lower()
    if key in fake_data:
        return f"Weather in {city}: {fake_data[key]}"
    return f"No weather data available for {city}"
    
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a basic math expression like '12 * 7' or '(4+5)/3'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city, e.g. 'Bangalore'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# map tool names to actual Python functions
available_functions = {
    "calculator": calculator,
    "get_weather": get_weather
}
messages = [
    {"role": "user", "content": "Get the weather in Delhi, then take the numeric temperature from that and multiply it by 3."}
]

while True:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools  # same tools list you already defined
    )

    msg = response.choices[0].message
    messages.append(msg)

    if msg.tool_calls:
        for call in msg.tool_calls:
            func_name = call.function.name
            func_args = json.loads(call.function.arguments)  # Groq returns a JSON string, not a dict

            print(f"[Model requested: {func_name}({func_args})]")

            result = available_functions[func_name](**func_args)
            print(f"[Tool result: {result}]")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,   # Groq requires this to link result to the specific call
                "content": str(result)
            })
        continue
    else:
        print("FINAL ANSWER:", msg.content)
        break