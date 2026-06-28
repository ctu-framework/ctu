import requests
import json
from json_repair import repair_json
import re


class Agent:

    def __init__(self, api_url, model, temperature, num_ctx):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx






    # ----------------------------------------
    # Build prompt (FULL TOOL INFO)
    # ----------------------------------------
    def build_prompt(self, query, tools, history=None, feedback=None, guidance=None):
        prompt = f"""
You are an AI agent that calls only ONE tool at a time to solve the query.
The output of each tool called is provided.
Check previous outputs before deciding what tool to call.
Check feedback, if provided, before deciding what tool to call.
Check guidance, if provided, before deciding what tool to call.
Do not generate tool output.
Select only one tool to call.
When you are done, call the tool "Finish".
Output valid JSON. Follow this output format:

{{
  "thought": "write your thought here",
  "action": "write your action here",
  "tool": "tool_name",
  "arguments": {{
    "param": "value",
    "param": "value"
  }}
}}

Orignal Query:
{query}

"""

        # Adds tokens to the prompt
        if history:
            prompt += "\nPrevious outputs:\n"
            for step in history:
                prompt += f"- {step['tool']}\n"

                if "output" in step:
                    prompt += "  output:\n"
                    prompt += f"    {step['output']}\n"

        # Adds history to the prompt
        if history:
            prompt += "\nPrevious successful steps:\n"
            for step in history:
                prompt += f"- {step['tool']}\n"


        # Adds feedback to the prompt
        if feedback:
            prompt += f"""

# Your previous action was rejected due to a constraint violation.

# Feedback:
# {feedback}

# Please choose a different tool or correct your arguments to satisfy the constraints.
# Do not repeat the same mistake.

"""

        # Adds guidance to the prompt
        if guidance:
            prompt += f"""

# Guidance:
# {guidance}

# Use this information to help decide your next action.
# These are suggested tools based on constraints, not mandatory choices.
        """

        prompt += f"""

All available tools:
{json.dumps(tools, indent=2)}
"""

        return prompt






    # ----------------------------------------
    # Extract JSON safely
    # ----------------------------------------
    def extract_json(self, text):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        return text

    # ----------------------------------------
    # Call LLM
    # ----------------------------------------
    def call_llm(self, prompt):

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": self.num_ctx
                    }
                }
            )


            raw = response.json()["response"]

            json_text = self.extract_json(raw)
            data = json.loads(json_text)


            return data

        except Exception as e:
            # print("[WARNING] Agent parsing failed:", e)
            return None








    # ----------------------------------------
    # Action
    # ----------------------------------------
    def act(self, query, tools, history, feedback=None, guidance=None):
        prompt = self.build_prompt(query, tools, history, feedback, guidance)
        output = self.call_llm(prompt)

        if not output:
            return None

        if isinstance(output, list):
            output = output[-1]

        return {
            "tool": output.get("tool"),
            "arguments": output.get("arguments", {})
        }