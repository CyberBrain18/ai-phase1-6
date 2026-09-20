import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"

# --- Worker 1: a narrow, math-only specialist ---
def calculator_agent(problem: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a math specialist. Solve the problem precisely, showing only the calculation and final numeric answer. No prose, no extra commentary."},
            {"role": "user", "content": problem}
        ]
    )
    return response.choices[0].message.content

# --- Worker 2: a narrow, writing-only specialist ---
def writer_agent(instruction: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a writing specialist. Given instructions and any supplied facts, write clear, well-structured prose. Do not do any math yourself — only use numbers you are explicitly given."},
            {"role": "user", "content": instruction}
        ]
    )
    return response.choices[0].message.content

orchestrator_tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator_agent",
            "description": "Delegates a math problem to a specialist that solves it precisely.",
            "parameters": {
                "type": "object",
                "properties": {"problem": {"type": "string"}},
                "required": ["problem"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "writer_agent",
            "description": "Delegates a writing task to a specialist. Pass any facts/numbers it needs explicitly in the instruction.",
            "parameters": {
                "type": "object",
                "properties": {"instruction": {"type": "string"}},
                "required": ["instruction"]
            }
        }
    }
]

available_agents = {
    "calculator_agent": calculator_agent,
    "writer_agent": writer_agent
}

def run_orchestrator(user_task):
    messages = [
        {"role": "system", "content": "You are an orchestrator. Break the task into subtasks and delegate each to the right specialist agent. Do not solve math yourself — always delegate it. Do not write final prose yourself — always delegate it to the writer."},
        {"role": "user", "content": user_task}
    ]

    while True:
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=orchestrator_tools)
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for call in msg.tool_calls:
                func_name = call.function.name
                func_args = json.loads(call.function.arguments)
                print(f"[Orchestrator delegates to: {func_name}({func_args})]")

                result = available_agents[func_name](**func_args)
                print(f"[{func_name} returned: {result}]\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })
            continue
        else:
            print("FINAL ANSWER:\n", msg.content)
            break


if __name__ == "__main__":
    run_orchestrator("Calculate 45 times 12, then write a short, friendly one-paragraph message to a client explaining the result.")