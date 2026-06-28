class Evaluator:

    def __init__(self):

        # ---------------------------
        # Core counts
        # ---------------------------
        self.total = 0  # total number of scenarios processed
        self.finished = 0  # scenarios that reach normal completion (Finish after at least one step)
        self.premature_finish = 0  # scenarios where Finish is called at first step
        self.max_steps_exceeded = 0  # scenarios that hit max step limit without finishing

        # ---------------------------
        # Violations
        # ---------------------------
        self.total_violations = 0  # total number of rejected actions
        self.access_violations = 0  # violations of forbidden tool usage
        self.temporal_violations = 0  # violations of ordering constraints
        self.contextual_violations = 0  # violations of parameter/value constraints

        # ---------------------------
        # Steps / trace
        # ---------------------------
        self.total_events = 0  # total number of trace events (actions + noise)
        self.total_actions = 0  # number of action steps (excluding noise)
        self.total_accepted = 0  # number of accepted actions
        self.total_rejected = 0  # number of rejected actions
        self.no_action_steps = 0  # steps where agent returned None
        self.invalid_tool_steps = 0  # steps where agent selected an invalid tool

        # ---------------------------
        # Agent quality
        # ---------------------------
        self.first_step_success = 0  # tasks solved in minimal steps (first valid action + Finish)

    # ----------------------------------------
    # Update with one result
    # ----------------------------------------
    def update(self, result):

        self.total += 1

        status = result["status"]
        history = result["history"]
        trace = result["trace"]

        # ---------------------------
        # Status handling
        # ---------------------------
        if status == "finished":
            self.finished += 1

            # valid completion = did something before Finish
            if len(history) > 1:
                pass  # optionally track separately

        elif status == "premature_finish":
            self.premature_finish += 1

        elif status == "max_steps_exceeded":
            self.max_steps_exceeded += 1

        # ---------------------------
        # Trace analysis
        # ---------------------------
        self.total_events += len(trace)

        for step in trace:

            step_type = step.get("type")

            # ---------------------------
            # Noise steps
            # ---------------------------
            if step_type == "no_action":
                self.no_action_steps += 1
                continue

            if step_type == "invalid_tool":
                self.invalid_tool_steps += 1
                continue

            # ---------------------------
            # Action steps
            # ---------------------------
            if step_type == "action":
                self.total_actions += 1

                if step["status"] == "accepted":
                    self.total_accepted += 1
                elif step["status"] == "rejected":
                    self.total_rejected += 1

                reason = step.get("reason")
                if reason:
                    self.total_violations += 1

                    rtype = reason.get("type")

                    if rtype == "access_violation":
                        self.access_violations += 1
                    elif rtype == "temporal_violation":
                        self.temporal_violations += 1
                    elif rtype == "contextual_violation":
                        self.contextual_violations += 1

        # ---------------------------
        # First-step success
        # ---------------------------
        if len(trace) >= 2:
            first = trace[0]
            second = trace[1]

            if (
                first.get("type") == "action" and
                first.get("status") == "accepted" and
                second.get("type") == "action" and
                second.get("tool") == "Finish"
            ):
                self.first_step_success += 1

    # ----------------------------------------
    # Summary
    # ----------------------------------------
    def summarize(self):

        # print("\n====== EVALUATION ======")

        if self.total == 0:
            # print("No data.")
            return

        # ---------------------------
        # Status
        # ---------------------------
        completion_rate = self.finished / self.total
        
        # print(f"Total examples: {self.total}")
        # print(f"Finished: {self.finished}")
        # print(f"Premature finish: {self.premature_finish}")
        # print(f"Max steps exceeded: {self.max_steps_exceeded}")
        # print(f"Completion rate: {completion_rate:.2f}")

        # ---------------------------
        # Violations
        # ---------------------------
        # print(f"\nViolations:")
        # print(f"  Total: {self.total_violations}")
        # print(f"  Access: {self.access_violations}")
        # print(f"  Temporal: {self.temporal_violations}")
        # print(f"  Contextual: {self.contextual_violations}")

        # ---------------------------
        # Trace / Efficiency
        # ---------------------------
        avg_events = self.total_events / self.total if self.total > 0 else 0

        acceptance_ratio = (
            self.total_accepted / self.total_actions
            if self.total_actions > 0 else 0
        )

        # print(f"\nExecution:")
        # print(f"  Avg events per run: {avg_events:.2f}")
        # print(f"  Total actions: {self.total_actions}")
        # print(f"  Accepted: {self.total_accepted}")
        # print(f"  Rejected: {self.total_rejected}")
        # print(f"  Acceptance ratio: {acceptance_ratio:.2f}")

        # print(f"\nNoise:")
        # print(f"  No action: {self.no_action_steps}")
        # print(f"  Invalid tool: {self.invalid_tool_steps}")

        # ---------------------------
        # Agent quality
        # ---------------------------
        first_step_success_rate = (
            self.first_step_success / self.total
            if self.total > 0 else 0
        )

        # print(f"\nAgent quality:")
        # print(f"  First-step success: {self.first_step_success / self.total:.2f}")





        return {
            # ---------------------------
            # Core
            # ---------------------------
            "total": self.total,
            "finished": self.finished,
            "premature_finish": self.premature_finish,
            "max_steps_exceeded": self.max_steps_exceeded,
            "completion_rate": completion_rate,

            # ---------------------------
            # Violations
            # ---------------------------
            "violations": {
                "total": self.total_violations,
                "access": self.access_violations,
                "temporal": self.temporal_violations,
                "contextual": self.contextual_violations
            },

            # ---------------------------
            # Execution
            # ---------------------------
            "execution": {
                "avg_events": avg_events,
                "total_actions": self.total_actions,
                "accepted": self.total_accepted,
                "rejected": self.total_rejected,
                "acceptance_ratio": acceptance_ratio
            },

            # ---------------------------
            # Noise
            # ---------------------------
            "noise": {
                "no_action": self.no_action_steps,
                "invalid_tool": self.invalid_tool_steps
            },

            # ---------------------------
            # Agent quality
            # ---------------------------
            "agent_quality": {
                "first_step_success_rate": first_step_success_rate
            }
        }

