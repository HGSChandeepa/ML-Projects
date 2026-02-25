import asyncio
import json
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent

load_dotenv()

COURSES_URL = "https://10x.adomicarts.com/courses"

api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set in .env")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_GEMINI_MODEL"),
    api_key=SecretStr(api_key),
    temperature=0,
)

task = f"""
Go to {COURSES_URL}. 
If there is any SSL or certificate warning, click proceed/accept to continue.

Collect details about every course listed on the page.

For each course:
1. Click on the course to open its detail page
2. Collect: course name, description, price, duration, instructor, and any other visible info
3. Go back to {COURSES_URL}
4. Repeat for every course

When done, return ONLY a raw JSON array, no markdown, no code fences:
[
  {{
    "name": "...",
    "description": "...",
    "price": "...",
    "duration": "...",
    "instructor": "...",
    "extras": "..."
  }}
]
"""

async def main():
    print("=" * 50)
    print("  browser-use course scraper  (v0.1.46)")
    print("=" * 50)
    print(f"\n[INFO] Target: {COURSES_URL}\n")

    agent = Agent(task=task, llm=llm)
    result = await agent.run()

     # ── Token usage ──────────────────────────────────────
    try:
        usage = result.total_input_tokens(), result.total_output_tokens()
        print("\n[TOKENS]")
        print(f"  Input tokens:  {usage[0]:,}")
        print(f"  Output tokens: {usage[1]:,}")
        print(f"  Total tokens:  {usage[0] + usage[1]:,}")
    except Exception:
        # Fallback: sum from individual action results
        try:
            input_tokens = sum(
                step.result.usage_metadata.get("input_tokens", 0)
                for step in result.action_results()
                if hasattr(step, "result") and hasattr(step.result, "usage_metadata")
            )
            output_tokens = sum(
                step.result.usage_metadata.get("output_tokens", 0)
                for step in result.action_results()
                if hasattr(step, "result") and hasattr(step.result, "usage_metadata")
            )
            print("\n[TOKENS]")
            print(f"  Input tokens:  {input_tokens:,}")
            print(f"  Output tokens: {output_tokens:,}")
            print(f"  Total tokens:  {input_tokens + output_tokens:,}")
        except Exception as e:
            print(f"\n[TOKENS] Could not retrieve token count: {e}")
            print(f"  Available result attrs: {[a for a in dir(result) if not a.startswith('_')]}")
    # ─────────────────────────────────────────────────────


    raw = result.final_result() or ""

    print("\n[RAW RESULT]")
    print(raw)


    

    # Strip accidental markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        courses = json.loads(cleaned)
        with open("courses.json", "w", encoding="utf-8") as f:
            json.dump(courses, f, indent=2)
        print(f"\n[DONE] ✓ Saved {len(courses)} course(s) to courses.json")
    except Exception as e:
        print(f"\n[WARN] Could not parse JSON: {e}")
        with open("courses_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        print("[WARN] Raw output saved to courses_raw.txt")

asyncio.run(main())