class NoConstraintEngine:
    """
    Baseline constraint engine.

    This engine does NOT enforce any constraints.
    It allows all actions, provides no feedback,
    and does not compute any guidance.
    But it detects and counts violations,
    for evaluation purposes

    Used to simulate a pure ReAct-style agent.
    """

    # Class defaults
    enforces_constraints = False
    provides_feedback = False
    provides_guidance = False
    

    def __init__(self, constraints):
        # Constraints are ignored in this engine
        self.constraints = constraints

        # Runtime capabilities (can be modified by config)
        self.enforce = NoConstraintEngine.enforces_constraints
        self.feedback = NoConstraintEngine.provides_feedback
        self.guidance = NoConstraintEngine.provides_guidance

    # ----------------------------------------
    # Validation
    # ----------------------------------------
    def validate(self, action, history):

        tool = action["tool"]
        args = action.get("arguments", {})

        valid, reason = self.check_access(tool)
        if not valid:
            return True, reason  # always accept

        valid, reason = self.check_temporal(tool, history)
        if not valid:
            return True, reason  # always accept

        valid, reason = self.check_contextual(tool, args)
        if not valid:
            return True, reason  # always accept

        return True, None

    # ----------------------------------------
    # Access constraints
    # ----------------------------------------
    def check_access(self, tool):
        for c in self.constraints:
            if c["type"] != "access":
                continue

            forbidden = c.get("forbidden", [])
            if tool in forbidden:
                return False, {
                    "type": "access_violation",
                    "tool": tool
                }

        return True, None

    # ----------------------------------------
    # Temporal constraints
    # ----------------------------------------
    def check_temporal(self, tool, history):
        for c in self.constraints:
            if c["type"] != "temporal":
                continue

            before = c.get("before")
            after = c.get("after")

            tool_list = [entry["tool"] for entry in history]

            if tool == after:
                if before not in tool_list:
                    return False, {
                        "type": "temporal_violation",
                        "required_before": before,
                        "missing": before,
                        "current": tool
                    }

        return True, None

    # ----------------------------------------
    # Contextual constraints
    # ----------------------------------------
    def normalize(self, val):
        if isinstance(val, list):
            return [self.normalize(item) for item in val]
        if isinstance(val, (int, float)):
            return str(val).strip("'\" ")
        if isinstance(val, str):
            return val.strip("'\" ")
        return str(val).strip("'\" ")

    def check_contextual(self, tool, args):
        for c in self.constraints:
            if c["type"] != "contextual":
                continue

            if c.get("tool") != tool:
                continue

            required = c.get("requires", {})

            for key, value in required.items():
                if key not in args:
                    return False, {
                        "type": "contextual_violation",
                        "subtype": "missing_parameter",
                        "tool": tool,
                        "parameter": key
                    }

                if self.normalize(args[key]) != self.normalize(value):
                    return False, {
                        "type": "contextual_violation",
                        "subtype": "value_mismatch",
                        "tool": tool,
                        "parameter": key,
                        "expected": value,
                        "received": args[key]
                    }

        return True, None