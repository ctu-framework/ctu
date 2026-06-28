class Controller:

    def __init__(self, agent, constraint_engine, max_steps=5):
        self.agent = agent
        self.engine = constraint_engine
        self.max_steps = max_steps

    # ----------------------------------------
    # Message (for logging)
    # ----------------------------------------
    def generate_message(self, reason, tool):

        rtype = reason.get("type")

        if rtype == "access_violation":
            return f"Access violation: tool '{tool}'"

        elif rtype == "temporal_violation":
            return f"Temporal violation: '{tool}' requires '{reason.get('required_before')}'"

        elif rtype == "contextual_violation":
            subtype = reason.get("subtype")

            if subtype == "missing_parameter":
                return f"Missing parameter '{reason.get('parameter')}' for '{tool}'"

            elif subtype == "value_mismatch":
                return (
                    f"Mismatch in '{tool}': {reason.get('parameter')} "
                    f"(expected '{reason.get('expected')}', received '{reason.get('received')}')"
                )

            return f"Contextual violation in '{tool}'"

        return "Unknown violation"

    # ----------------------------------------
    # Feedback (to agent)
    # ----------------------------------------
    def generate_feedback(self, reason, tool):

        rtype = reason.get("type")

        if rtype == "access_violation":
            return (
                f"Your previous action used tool '{tool}', but this tool is not allowed. "
                f"Choose a different tool."
            )

        elif rtype == "temporal_violation":
            return (
                f"Your previous action used tool '{tool}', but it was called too early. "
                f"You must call '{reason.get('required_before')}' before using this tool."
            )

        elif rtype == "contextual_violation":

            subtype = reason.get("subtype")

            if subtype == "missing_parameter":
                return (
                    f"Your previous action used tool '{tool}', but parameter "
                    f"'{reason.get('parameter')}' is missing. Please include it."
                )

            elif subtype == "value_mismatch":
                return (
                    f"Your previous action used tool '{tool}', but parameter "
                    f"'{reason.get('parameter')}' is incorrect. Use a value consistent with the query."
                )

            return (
                f"Your previous action used tool '{tool}', but parameters were invalid."
            )

        return f"Your previous action with '{tool}' was invalid."

    # ----------------------------------------
    # Guidance formatting
    # ----------------------------------------
    def generate_guidance(self, guidance):

        if not guidance:
            return None

        formatted = []

        for item in guidance:
            if isinstance(item, dict):
                tool = item.get("tool")
                note = item.get("note")

                if note:
                    formatted.append(f"{tool} ({note})")
                else:
                    formatted.append(tool)
            else:
                formatted.append(item)

        return "Preferred tools based on current state: " + ", ".join(formatted)

    # ----------------------------------------
    # Tool utilities
    # ----------------------------------------
    def format_tools_for_prompt(self, tools):

        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
            for t in tools
        ]

    def get_available_tool_names(self, tools):
        return [t["name"] for t in tools]

    def get_tool_output(self, tools, tool_name):
        tool_def = next((t for t in tools if t["name"] == tool_name), None)
        return tool_def.get("output") if tool_def else None

    # ----------------------------------------
    # Main execution loop
    # ----------------------------------------
    def run(self, query, tools):

        history = []
        trace = []
        feedback = None

        tools_prompt = self.format_tools_for_prompt(tools)
        available_tools = self.get_available_tool_names(tools)

        for step in range(self.max_steps):

            # -------------------------------
            # Compute guidance (if supported)
            # -------------------------------
            guidance = None

            if self.engine.guidance:
                raw = self.engine.compute_valid_tools(history)
                guidance = self.generate_guidance(raw)

            # -------------------------------
            # Agent decision
            # -------------------------------
            action = self.agent.act(
                query,
                tools_prompt,
                history,
                feedback,
                guidance
            )
            

            # -------------------------------
            # Handle NONE
            # -------------------------------
            if action is None:
                # print(f"[ERROR] Step {step}: No action")

                trace.append({
                    "type": "no_action",
                    "step": step
                })

                feedback = None
                continue

            tool = action.get("tool")

            # -------------------------------
            # Handle invalid tool
            # -------------------------------
            if tool not in available_tools:
                # print(f"[ERROR] Step {step}: Invalid tool: {tool}")

                trace.append({
                    "type": "invalid_tool",
                    "tool": tool,
                    "step": step
                })

                feedback = None
                continue

            # -------------------------------
            # Validate
            # -------------------------------
            valid, reason = self.engine.validate(action, history)

            trace.append({
                "type": "action",
                "tool": tool,
                "arguments": action.get("arguments", {}),
                "status": "accepted" if valid else "rejected",
                "reason": reason
            })

            if not valid and self.engine.enforce:
                msg = self.generate_message(reason, tool)
                
                if self.engine.feedback:
                    feedback = self.generate_feedback(reason, tool)

                # print(f"[REJECTED] Step {step}: {msg}")
                continue

            # -------------------------------
            # Accept action
            # -------------------------------
            # print(f"[ACCEPTED] Step {step}: {tool}")

            output = self.get_tool_output(tools, tool)
            feedback = None

            if output is not None:
                history.append({"tool": tool, "output": output})
            else:
                history.append({"tool": tool})

            # -------------------------------
            # Stop conditions
            # -------------------------------
            if tool == "Finish":
                if step == 0:
                    return {
                        "status": "premature_finish",
                        "history": history,
                        "trace": trace
                    }

                return {
                    "status": "finished",
                    "history": history,
                    "trace": trace
                }

        # -------------------------------
        # Max steps
        # -------------------------------
        # print("[WARNING] Max steps reached")

        return {
            "status": "max_steps_exceeded",
            "history": history,
            "trace": trace
        }