import os
import json
import traceback
from dotenv import load_dotenv
from google import genai
from google.genai import types
from browser import BrowserController

load_dotenv()

def load_skills(path="skills/youtube_download_skill.md"):
    with open(path, "r") as f:
        return f.read()

# Define tools in Gemini's format
tools = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="navigate",
        description="Go to a URL",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "url": types.Schema(type="STRING", description="The URL to navigate to")
            },
            required=["url"]
        )
    ),
    types.FunctionDeclaration(
        name="fill_field",
        description="Fill a form field using a CSS selector",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "selector": types.Schema(type="STRING", description="CSS selector of the field"),
                "value": types.Schema(type="STRING", description="Value to fill in")
            },
            required=["selector", "value"]
        )
    ),
    types.FunctionDeclaration(
        name="click",
        description="Click an element using a CSS selector",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "selector": types.Schema(type="STRING", description="CSS selector to click")
            },
            required=["selector"]
        )
    ),
    types.FunctionDeclaration(
        name="get_page_text",
        description="Read the current page content to understand what's on screen",
        parameters=types.Schema(
            type="OBJECT",
            properties={}
        )
    ),
    types.FunctionDeclaration(
        name="ask_user",
        description="Ask the user a question when you need missing info or confirmation before submitting",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "question": types.Schema(type="STRING", description="Question to ask the user")
            },
            required=["question"]
        )
    )
])

import logging

# Logging setup
LOG_PATH = os.path.join(os.path.dirname(__file__), "agent.log")
logger = logging.getLogger("web_agent")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)

def run_tool(browser, tool_name, tool_input):
    if tool_name == "navigate":
        return browser.navigate(tool_input["url"])
    elif tool_name == "fill_field":
        return browser.fill_field(tool_input["selector"], tool_input["value"])
    elif tool_name == "click":
        return browser.click(tool_input["selector"])
    elif tool_name == "get_page_text":
        return browser.get_page_text()
    elif tool_name == "ask_user":
        print(f"\n🤖 Agent: {tool_input['question']}")
        user_response = input("You: ")
        return f"User answered: {user_response}"

def run_agent(user_task):
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception as e:
        print(f"Error creating genai client: {e}")
        traceback.print_exc()
        return

    browser = BrowserController(headless=False)
    print("✅ Browser launched")
    skills = load_skills()

    system_prompt = f"""You are an automated browser agent.
You have detailed knowledge about how to interact with specific websites via the skills below.
Always follow the rules defined in the skills file.
If any required data is missing, use ask_user tool to request it before proceeding.
Always ask for confirmation before final form submission.

--- SKILLS ---
{skills}
"""

    logger.info("Starting agent for task: %s", user_task)

    # Build conversation history
    history = []
    
    # Add the initial user task
    history.append(types.Content(
        parts=[types.Part(text=user_task)]
    ))

    print(f"\n🚀 Starting agent for task: {user_task}\n")

    while True:
        try:
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[tools],
                    temperature=0.1             # lower = more deterministic actions
                )
            )
        except Exception as e:
            logger.error("Error calling model.generate_content: %s", e)
            print(f"Error calling model.generate_content: {e}")
            traceback.print_exc()
            break

        candidate = response.candidates[0]
        content = candidate.content

        # Add assistant response to history
        history.append(content)

        # Check if there are any function calls
        function_calls = [p for p in content.parts if p.function_call]

        # If no function calls, agent is done
        if not function_calls:
            for part in content.parts:
                if part.text:
                    print(f"\n✅ Agent: {part.text}")
            break

        # Process all function calls and collect results
        tool_result_parts = []
        for part in function_calls:
            fc = part.function_call
            # fc.args may be a JSON string or an object; normalize to dict
            raw_args = fc.args
            try:
                if isinstance(raw_args, str):
                    args = json.loads(raw_args)
                else:
                    # try to convert to dict if it has a to_dict() style
                    try:
                        args = dict(raw_args)
                    except Exception:
                        try:
                            args = raw_args.to_dict()
                        except Exception:
                            args = raw_args

                logger.info("Calling tool: %s → %s", fc.name, args)
                print(f"🔧 Calling tool: {fc.name} → {args}")
                result = run_tool(browser, fc.name, args)
                logger.info("Tool result: %s", result)
                print(f"   Result: {result}")
            except Exception as e:
                logger.exception("Unhandled exception in run loop: %s", e)
                print(f"   Error running tool {fc.name}: {e}")
                traceback.print_exc()
                result = f"error: {e}"

            tool_result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result}
                    )
                )
            )

        # Add tool results back into history as a user turn
        history.append(types.Content(
            role="user",
            parts=tool_result_parts
        ))

    try:
        browser.close()
        logger.info("Browser closed")
    except Exception as e:
        logger.exception("Error closing browser")

if __name__ == "__main__":
    task = input("What should the agent do?\nYou: ")
    run_agent(task)