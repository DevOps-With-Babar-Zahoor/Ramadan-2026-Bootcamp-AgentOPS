# Lab 2: Simulating a ReAct Loop in Python

**Goal:** Build a "Mental Model" of an Agent by coding a simplified control loop.
**Audience:** DevOps Engineers (Python knowledge: Basic)
**Prerequisites:** Python installed (no external libraries needed)

---

## 🎯 Lab Objective

We are going to demystify the "Magic" of AI Agents.
We will build a single Python script that simulates an Agent.

It will:
1.  Receive a user question.
2.  "Think" about what to do (using a mock function).
3.  "Act" by calling a tool.
4.  "Observe" the result.
5.  Repeat until the goal is met.

This is the **ReAct Pattern** (Reason + Act).

---

## 💻 Step 1: The Tools

First, we define what our agent *can do*.
In a real system, these would be API calls. Here, they are simple functions.

```python
# tools.py

def get_weather(city):
    """Simulates looking up weather data."""
    print(f"[TOOL] Querying weather API for {city}...")
    # Mock data
    if city.lower() == "london":
        return "Rainy, 15°C"
    elif city.lower() == "dubai":
        return "Sunny, 40°C"
    else:
        return "Unknown location"

def calculator(expression):
    """Simulates a calculation tool."""
    print(f"[TOOL] Calculating: {expression}")
    try:
        # unsafe eval used for simplicity only!
        return str(eval(expression)) 
    except:
        return "Error in calculation"
```

---

## 🧠 Step 2: The "Brain" (Mock LLM)

Real agents use GPT-4. We will use a `mock_llm` function.
Instead of real intelligence, we hardcode the "reasoning steps" to demonstrate the structure.

*   **Input:** The conversation history (context).
*   **Output:** A structured "Thought" and "Action".

```python
# mock_llm.py

def mock_llm(history):
    """
    Simulates an LLM's decision making process based on the current context.
    It returns a JSON-like dictionary.
    """
    history_str = str(history).lower()

    # SCENARIO: User asks "What is the weather in Dubai?"
    if "weather in dubai" in history_str and "rainy" not in history_str and "sunny" not in history_str:
        return {
            "thought": "The user wants Dubai weather. I should use the weather tool.",
            "tool": "get_weather",
            "tool_input": "Dubai",
            "final_answer": None
        }
    
    # SCENARIO: We have tool output (observation), so we can answer.
    elif "sunny, 40°c" in history_str:
         return {
            "thought": "I have the weather data. I can answer the user.",
            "tool": None,
            "tool_input": None,
            "final_answer": "The weather in Dubai is Sunny and 40°C."
        }

    # FALLBACK
    return {
        "thought": "I don't know what to do.",
        "tool": None,
        "tool_input": None,
        "final_answer": "I am confused."
    }
```

---

## 🔄 Step 3: The Control Loop (The Engine)

This is the **AgentOps** part.
This `while` loop is the heart of every autonomous agent.

```python
# agent.py
from mock_llm import mock_llm
from tools import get_weather, calculator
def run_agent(user_query):
    # 1. Initialize State
    history = [f"User: {user_query}"]
    step_count = 0
    max_steps = 5  # SAFETY LIMIT (Circuit Breaker)

    print(f"🏁 GOAL: {user_query}\n")

    while step_count < max_steps:
        step_count += 1
        print(f"--- Step {step_count} ---")

        # 2. REASON: Call the 'LLM' to allow it to think
        response = mock_llm(history)
        
        print(f"💭 THOUGHT: {response['thought']}")

        # 3. CHECK: Are we done?
        if response['final_answer']:
            print(f"✅ FINAL ANSWER: {response['final_answer']}")
            return

        # 4. ACT: Execute the selected tool
        tool_name = response['tool']
        tool_arg = response['tool_input']
        
        observation = "Error: Tool not found"
        
        print(f"🛠️ ACTION: Calling {tool_name}('{tool_arg}')")

        if tool_name == "get_weather":
            observation = get_weather(tool_arg)
        elif tool_name == "calculator":
            observation = calculator(tool_arg)

        print(f"👀 OBSERVATION: {observation}")

        # 5. UPDATE STATE: Append result to history so LLM knows what happened
        history.append(f"Observation: {observation}")
        print("") # formatting

    print("❌ Failed to reach goal within step limit.")

# --- Run It ---
if __name__ == "__main__":
    run_agent("What is the weather in Dubai?")
```


---

## 🔬 Lab Analysis
```python
# agent_lab.py
# main entry point
#!/usr/bin/env python3
"""
Lab 2: ReAct Loop - Entry Point
Run this file to execute the agent simulation.
"""

from agent import run_agent

def main():
    # 🎯 Define the user's goal
    goal = "What is the weather in Dubai?"
    
    # 🚀 Run the agent
    print(f"\n🏁 GOAL: {goal}\n")
    run_agent(goal)

if __name__ == "__main__":
    main()
```


Save this code as `agent_lab.py` and run it:
`python agent_lab.py`

**Expected Output:**
```text
🏁 GOAL: What is the weather in Dubai?

--- Step 1 ---
💭 THOUGHT: The user wants Dubai weather. I should use the weather tool.
🛠️ ACTION: Calling get_weather('Dubai')
[TOOL] Querying weather API for Dubai...
👀 OBSERVATION: Sunny, 40°C

--- Step 2 ---
💭 THOUGHT: I have the weather data. I can answer the user.
✅ FINAL ANSWER: The weather in Dubai is Sunny and 40°C.
```

## ⚠️ Reflection for DevOps Engineers

Look at the `while` loop again. Imagine this running in production.

1.  **The Infinite Loop:** What if `mock_llm` kept saying "Use Weather Tool" forever? We added `max_steps`. That is a **Guardrail**.
2.  **State Explosion:** The `history` list grows with every step. In a real LLM, this consumes **Context Window (Money)**.
3.  **Tool Failure:** What if `get_weather` timed out? The agent needs error handling logic in the loop.
4.  **Observability:** Without the `print` statements, this system is a Black Box. In production, every `THOUGHT`, `ACTION`, and `OBSERVATION` must be sent to OpenTelemetry.

**This script is a toy.**
**But the ARCHITECTURE is real.**

This is exactly how LangChain, AutoGen, and CrewAI work under the hood.
And managing this loop at scale... is **AgentOps**.
