import networkx as nx
import json


class KGConstraintEngine:
    """
    Graph-based constraint engine (active guidance).

    Uses a graph to represent tool relationships and constraints.
    The key capability is computing the set of valid next actions
    based on execution state (history), enabling constraint-guided
    decision-making (active neuro-symbolic behavior).
    """

    enforces_constraints = True
    provides_feedback = True
    provides_guidance = True


    def __init__(self, tools, constraints):

        # Runtime capabilities (can be modified by config)
        self.enforce = KGConstraintEngine.enforces_constraints
        self.feedback = KGConstraintEngine.provides_feedback
        self.guidance = KGConstraintEngine.provides_guidance



        self.G = nx.DiGraph()

        # Ensure all tools exist as nodes
        for t in tools:
            self.G.add_node(t["name"])


        for c in constraints:

            if c["type"] == "temporal":
                if not c["before"]: continue
                if not c["after"]: continue
                self.G.add_edge(
                    c["before"],
                    c["after"],
                    type="temporal"
                )

            elif c["type"] == "access":
                for tool in c["forbidden"]:
                    if not c["forbidden"]: continue
                    if tool in self.G.nodes:
                        self.G.nodes[tool]["forbidden"] = True

            elif c["type"] == "contextual":
                if not c["tool"]: continue
                if c["tool"] not in self.G: continue
                reqs = c.get("requires", {})
                for param, value in reqs.items():
                    self.G.nodes[c["tool"]]["required_arg"] = {
                        "param": param,
                        "value": value
                    }

    # ----------------------------------------
    # Validation (passive behavior)
    # ----------------------------------------
    def validate(self, action, history):
        tool = action["tool"]
        args = action.get("arguments", {})

        valid, reason = self.check_access(tool)
        if not valid:
            return False, reason

        valid, reason = self.check_temporal(tool, history)
        if not valid:
            return False, reason

        valid, reason = self.check_contextual(tool, args)
        if not valid:
            return False, reason

        return True, None


    # -------------------------------
    # ACCESS (forbidden)
    # -------------------------------
    def check_access(self, tool):

        node = self.G.nodes.get(tool, {})

        if node.get("forbidden", False):
            return False, {
                "type": "access_violation",
                "tool": tool
            }

        return True, None


    # -------------------------------
    # TEMPORAL
    # -------------------------------
    def check_temporal(self, tool, history):

        used = set(h["tool"] for h in history)

        for before in self.G.predecessors(tool):
            if before not in used:
                return False, {
                    "type": "temporal_violation",
                    "required_before": before,
                    "missing": before,
                    "current": tool
                }

        return True, None


    # -------------------------------
    # CONTEXTUAL
    # -------------------------------
    def normalize(self, val):
        if isinstance(val, list):
            return [self.normalize(item) for item in val]
        if isinstance(val, (int, float)):
            return str(val).strip("'\" ")
        if isinstance(val, str):
            return val.strip("'\" ")
        return str(val).strip("'\" ")

    def check_contextual(self, tool, args):

        node = self.G.nodes.get(tool, {})
        req = node.get("required_arg")

        if req:
            key = req["param"]
            value = req["value"]

            # Missing parameter
            if key not in args:
                return False, {
                    "type": "contextual_violation",
                    "subtype": "missing_parameter",
                    "tool": tool,
                    "parameter": key
                }

            # Value mismatch
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



    # ----------------------------------------
    # ACTIVE GUIDANCE (core contribution)
    # ----------------------------------------
    def compute_valid_tools(self, history):
        """
        Compute tools that are executable given the current state.
        """

        used = set(h["tool"] for h in history)
        valid = []

        for tool in self.G.nodes:

            if tool in used:
                continue

            node = self.G.nodes[tool]

            # Forbidden
            if node.get("forbidden", False):
                continue

            # Temporal constraints
            temporal_ok = True
            for pred in self.G.predecessors(tool):
                if pred not in used:
                    temporal_ok = False
                    break
            if not temporal_ok:
                continue

            # Contextual constraints
            req = node.get("required_arg")

            if req:
                valid.append({
                    "tool": tool,
                    "note": f"requires parameter '{req['param']}' with a specific value"
                })
                continue

            valid.append({
                "tool": tool
            })

        return valid if valid else None