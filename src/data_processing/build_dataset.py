from json_repair import repair_json
import json
import requests
import yaml
import re
import ast
from tqdm import tqdm

# Parallelism
from concurrent.futures import ThreadPoolExecutor



# ----------------------------------------
# Load config
# ----------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mode = config["mode"]

# ----------------------------------------
# Paths
# ----------------------------------------
input_dataset_path = config["preprocess"][mode]["input_dataset_path"]
output_dataset_path = config["preprocess"][mode]["output_dataset_path"]

# ----------------------------------------
# Model / API
# ----------------------------------------
api_url = config["preprocess"]["api_url"]
model = config["preprocess"]["model"]
temperature = config["preprocess"]["temperature"]
num_ctx = config["preprocess"]["num_ctx"]

# ----------------------------------------
# Parameters
# ----------------------------------------
max_examples = config["preprocess"][mode].get("max_examples")
max_workers = config["preprocess"]["max_workers"]



# ========================================
# -------- Extraction Functions ----------
# ========================================

def extract_system_message(conversations):
    systems = [c for c in conversations if c["from"] == "system"]
    return systems[0]["value"] if len(systems) == 1 else None


def extract_user_query(conversations):
    users = [c for c in conversations if c["from"] == "user"]
    return users[0]["value"].strip() if len(users) == 1 else None


def extract_tool_block(system_text):
    pattern = r"Specifically, you have access to the following APIs:\s*(\[[\s\S]*\])"
    match = re.search(pattern, system_text)
    if not match:
        raise ValueError("API block not found")
    return match.group(1)

def extract_assistant_function_pairs(conversations):
    pairs = []

    for a, f in zip(conversations, conversations[1:]):
        if a.get("from") == "assistant" and f.get("from") in ["function", "tool"]:
            pairs.append((a, f))

    return pairs



def extract_tool_output_map(pairs):
    """
    Returns dict: tool_name → output

    If ANY issue occurs (parsing failure, mismatch, etc.),
    returns None and prints a warning.
    """

    pattern = r"Action:\s*(.*?)\s*Action Input"
    output_map = {}

    if not pairs:
        return None

    for (a, f) in pairs:

        # Extract tool name
        text = a.get("value", "")
        match = re.search(pattern, text)

        if not match:
            return None

        tool_name = match.group(1).strip()

        # Skip Finish safely (not an error)
        if tool_name == "Finish":
            continue

        # Extract raw value
        raw_value = f.get("value")

        if not raw_value:
            return None

        # Repair + parse JSON (level 1)
        try:
            parsed = json.loads(raw_value)
        except Exception as e:
            return None

        # Extract response
        if isinstance(parsed, dict):
            response = parsed.get("response", parsed)
        else:
            response = parsed


        output_map[tool_name] = response

    return output_map


def clean_description(description):
    if "The description of this function is:" in description:
        description = description.split("The description of this function is:")[-1]
    return description.strip().strip('"')



def parse_tools(api_block, output_map):
    api_list = ast.literal_eval(api_block)

    tools = []

    for api in api_list:
        name = api.get("name")

        tool = {
            "name": name,
            "description": clean_description(api.get("description", "")),
            "parameters": []
        }

        properties = api.get("parameters", {}).get("properties", {})
        for pname, pinfo in properties.items():
            tool["parameters"].append({
                "name": pname,
                "type": pinfo.get("type", ""),
                "description": pinfo.get("description", "")
            })

        if name in output_map:
            tool["output"] = output_map[name]

        tools.append(tool)

    return tools


# ========================================
# -------- Prompt Builders ---------------
# ========================================

def build_access_prompt(tool_names):
    return f"""
You are generating FORBIDDEN TOOLS for a tool-using system.
You are given a list of existing tool names.

Your task:
- Create one, two, or three NEW tools similar to existing tools
- They must:
  - follow the naming pattern
  - belong to the same domain
  - look realistic and plausible

Each new tool MUST include:
- name
- short description
- parameters (name + type)

Return ONLY valid JSON with output. No explanations.

Existing tool names:
{json.dumps(tool_names, indent=2)}

This is an example output:
{{
  "forbidden_tools": [
    {{
      "name": "get_user_profile_extended",
      "description": "Retrieve extended profile",
      "parameters": [
        {{"name": "user_id", "type": "string"}},
        {{"name": "include_history", "type": "boolean"}}
      ]
    }}
  ]
}}
"""





def build_temporal_prompt(tool_names):
    return f"""
You are generating TEMPORAL constraints for a tool-using system.
You are given a list of existing tool names.

Task:
- Detect if one tool must be used before another
- Use parameter dependencies or strong logic only
- Create one, two, or three temporal constraints
- Keep constraints separate in JSON, even if they share the same before or after
- Temporal constraints must be realistic
- If no clear dependency, return empty

Existing tool names:
{json.dumps(tool_names, indent=2)}

Return ONLY valid JSON with output. No explanations.

This is an example output:
{{
  "constraints": [
    {{
      "type": "temporal",
      "before": "tool_name_A",
      "after": "tool_name_B"
    }},
    {{
      "type": "temporal",
      "before": "tool_name_A",
      "after": "tool_name_C"
    }}
  ]
}}
"""


def build_contextual_prompt(query, tools):
    return f"""
You are generating CONTEXTUAL constraints for a tool-using system.
You are given a user query and a list of existing tools.

Task:
- Identify values from query that must be used as parameters
- Create one, two, or three contextual constraints
- Examples: numbers, names, identifiers
- Do not invent values
- Contextual constraints must be realistic
- If no clear constraint, do nothing

User query:
{query}

Existing tools:
{json.dumps(tools, indent=2)}

Return ONLY valid JSON with output. No explanations.

This is an example output:
{{
  "constraints": [
    {{
      "type": "contextual",
      "tool": "tool_name",
      "requires": {{
        "param1": "value",
        "param2": "value"
      }}
    }},
    {{
      "type": "contextual",
      "tool": "tool_name",
      "requires": {{
        "param1": "value",
        "param2": "value"
      }}
    }},
    {{
      "type": "contextual",
      "tool": "tool_name",
      "requires": {{}}
    }}
  ]
}}
"""


# ========================================
# -------- LLM Call ----------------------
# ========================================

def call_api(prompt, api_url, model, temperature):
    try:
        response = requests.post(
            api_url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx
                }
            }
        )

        text = response.json()["response"]
        data = json.loads(text)
        return data

    except Exception as e:
        return []


# ========================================
# -------- Constraint Cleanup -------------
# ========================================

def clean_constraints(constraints):
    cleaned = []

    for c in constraints:
        if not isinstance(c, dict):
            continue
        if "type" not in c:
            continue

        # Optional: basic sanity checks
        if c["type"] == "temporal":
            if "before" not in c or "after" not in c:
                continue

        cleaned.append(c)

    return cleaned






# ========================================
# -------- Parallelism ------------------
# ========================================

def run_example(i, example):


    conversations = example.get("conversations", [])

    system_text = extract_system_message(conversations)
    query = extract_user_query(conversations)

    if not system_text:
        return None

    if not query:
        return None

    try:
        pairs = extract_assistant_function_pairs(conversations)
        output_map = extract_tool_output_map(pairs)

        if output_map is None:
            return None

        api_block = extract_tool_block(system_text)
        tools = parse_tools(api_block, output_map)
    except Exception as e:
        return None

    if not tools:
        return None

    # ====================================
    # LLM Calls (3 separate)
    # ====================================


    tool_names = [t["name"] for t in tools]

    access_output = call_api(build_access_prompt(tool_names), api_url, model, temperature)
    forbidden_tools = access_output.get("forbidden_tools", [])

    temporal_output = call_api(build_temporal_prompt(tool_names), api_url, model, temperature)
    temporal_constraints = temporal_output.get("constraints", [])

    contextual_output = call_api(build_contextual_prompt(query, tools), api_url, model, temperature)
    contextual_constraints = contextual_output.get("constraints", [])


    if forbidden_tools:
        tools.extend(forbidden_tools)

    access_constraint = {
        "type": "access",
        "forbidden": [t["name"] for t in forbidden_tools]
    }


    constraints = (
        [access_constraint]
        + temporal_constraints
        + contextual_constraints
    )

    constraints = clean_constraints(constraints)


    return {
        "query": query,
        "tools": tools,
        "constraints": constraints
    }


# ========================================
# -------- Dataset Builder --------------
# ========================================

def build_dataset(data):

    dataset = []
    skipped = 0

    data_to_process = data[:max_examples]


    # Parallelism
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        raw_results = list(tqdm(
            pool.map(lambda x: run_example(*x),enumerate(data_to_process)),
            total=len(data_to_process),
            desc="Processing Examples"
    ))

    # count skips
    skipped = raw_results.count(None)

    # filter out None's (skipped)
    valid_results = [r for r in raw_results if r is not None]

    # ids
    for i, r in enumerate(valid_results, start=1):
        r["id"] = i

    dataset = valid_results

        

    print(f"Done. Total: {len(dataset)}, Skipped: {skipped}")
    return dataset


# ========================================
# -------- Main --------------------------
# ========================================

def main():
    with open(input_dataset_path) as f:
        raw_data = json.load(f)

    dataset = build_dataset(raw_data)

    with open(output_dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved to {output_dataset_path}")


if __name__ == "__main__":
    main()