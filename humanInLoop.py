import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def calculator(expression: str) -> str:
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def send_email(to: str, subject: str, body: str) -> str:
    # in a real system this would actually call an email API — mocked here for safety
    return f"[MOCK] Email sent to {to} — subject: '{subject}'"

# each tool is tagged with whether it needs a human checkpoint before running
tool_config = {
    "calculator": {"function": calculator, "requires_approval": False},
    "send_email": {"function": send_email, "requires_approval": True},
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a basic math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Sends an email to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]

def run_agent(user_task):
    messages = [{"role": "user", "content": user_task}]

    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools
        )
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for call in msg.tool_calls:
                func_name = call.function.name
                func_args = json.loads(call.function.arguments)
                config = tool_config[func_name]

                if config["requires_approval"]:
                    print(f"\n[APPROVAL NEEDED] The agent wants to call: {func_name}({func_args})")
                    approval = input("Approve? (yes/no): ").strip().lower()

                    if approval != "yes":
                        result = "Action rejected by user. Do not attempt this action again; ask the user what they'd like instead."
                    else:
                        result = config["function"](**func_args)
                else:
                    # no approval needed — runs immediately, same as Phase 2
                    result = config["function"](**func_args)

                print(f"[Tool result: {result}]")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result)
                })
            continue
        else:
            print("\nFINAL ANSWER:", msg.content)
            break


if __name__ == "__main__":
    run_agent("Calculate 45*12, then send an email to test@example.com summarizing the result.")