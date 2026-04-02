import dspy
import re
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
import litellm


class PlanSignature(dspy.Signature):
    history = dspy.InputField()
    plan = dspy.OutputField(desc="Short internal plan, not shown to user")


class ActSignature(dspy.Signature):
    history = dspy.InputField()
    plan = dspy.InputField()
    action = dspy.OutputField(desc="Must EXACTLY follow SWE-agent format: DISCUSSION, one sentence, then a single command in a code block")


class PlanActModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.Predict(PlanSignature)
        self.act = dspy.Predict(ActSignature)


    def safe_extract(self, pred, field):
        if isinstance(pred, str):
            return pred

        if hasattr(pred, field):
            return getattr(pred, field)

        return str(pred)


    def forward(self, history):
        plan_pred = self.plan(history=history)
        
        plan = self.safe_extract(plan_pred, "plan")

        act_pred = self.act(history=history, plan=plan)
        action = self.safe_extract(act_pred, "action")

        return dspy.Prediction(plan=plan, action=action)
    

class LiteLLMAdapter(dspy.LM):
    def __init__(self):
        self.litellm = litellm
        self.model = "openai/gpt-4"
        self.kwargs = {}
     
        
    def clean(self, text):
        return re.sub(r"\[\[.*?\]\]", "", text).strip()
    
    
    def __call__(self, messages=None, **kwargs):
        if messages is None:
            return ""

        system_prompt = (
            "You are a SWE-agent coder.\n"
            "You MUST output EXACTLY in this format:\n\n"
            "DISCUSSION\n"
            "One short sentence.\n\n"
            "```\n"
            "single command here\n"
            "```\n\n"
            "Rules:\n"
            "- Exactly one sentence in DISCUSSION\n"
            "- Exactly one command\n"
            "- No extra text\n"
            "- No JSON\n"
        )

        response = self.litellm.completion(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0
        )

        content = self.clean(response["choices"][0]["message"]["content"])
        print(f"🚀 CLEANED RESPONSE: {[content]}")
        
        return [content]


class DSPyPolicyWrapper:
    def __init__(self, model):
        print("🚀 DSPY WRAPPER INITIALIZING")

        lm = LiteLLMAdapter()

        dspy.settings.configure(lm=lm)
        dspy.settings.experimental = True
        
        examples = [
            dspy.Example(
                history="Test failure mentions raw_building.py line 470",
                plan="inspect relevant lines",
                action=(
                    "DISCUSSION\n"
                    "View the relevant lines around the failure location.\n\n"
                    "```\n"
                    "str_replace_editor view /testbed/astroid/raw_building.py --view_range 460 480\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="Deprecation warning from numpy attribute access",
                plan="find problematic getattr usage",
                action=(
                    "DISCUSSION\n"
                    "Search for getattr usage that may trigger numpy warnings.\n\n"
                    "```\n"
                    "grep -n getattr -R /testbed\n"
                    "```"
                )
            ).with_inputs("history"),
            
            dspy.Example(
                history="Found getattr(sys.modules[modname], name) causing warning",
                plan="patch unsafe getattr",
                action=(
                    "DISCUSSION\n"
                    "Modify getattr usage to safely handle deprecated numpy attributes.\n\n"
                    "```\n"
                    "str_replace_editor str_replace /testbed/astroid/raw_building.py --old_str 'getattr(sys.modules[modname], name)' --new_str 'getattr(sys.modules.get(modname, None), name, None)'\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="Edited code but unsure if issue fixed",
                plan="run tests",
                action=(
                    "DISCUSSION\n"
                    "Run the test suite to verify the fix.\n\n"
                    "```\n"
                    "pytest -q\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="Tests still failing after patch",
                plan="reinspect failure",
                action=(
                    "DISCUSSION\n"
                    "Search for remaining occurrences of problematic pattern.\n\n"
                    "```\n"
                    "grep -n getattr -R /testbed\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="All tests passing successfully",
                plan="submit solution",
                action=(
                    "DISCUSSION\n"
                    "All tests are passing so submit the solution.\n\n"
                    "```\n"
                    "submit\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="Already inspected relevant file and identified issue",
                plan="apply fix",
                action=(
                    "DISCUSSION\n"
                    "Apply the identified fix instead of further searching.\n\n"
                    "```\n"
                    "str_replace_editor str_replace /testbed/file.py --old_str 'buggy_code' --new_str 'fixed_code'\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="Repeated grep commands with no new info",
                plan="take action",
                action=(
                    "DISCUSSION\n"
                    "Stop searching and modify the code based on current findings.\n\n"
                    "```\n"
                    "str_replace_editor str_replace /testbed/file.py --old_str 'x' --new_str 'y'\n"
                    "```"
                )
            ).with_inputs("history"),
        ]

        VALID_TOOLS = [
            "bash",
            "python",
            "grep",
            "search",
            "open_file",
            "create_file",
            "write_file",
            "append_file",
            "str_replace_editor",
            "delete_file",
            "submit"
        ]
        
        def metric(example, pred, trace=None):
            action = str(pred.action).lower()
            history = example.history.lower()

            score = 0

            if any(tool in action for tool in VALID_TOOLS):
                score += 1
            else:
                return -5  # invalid actions are fatal


            def action_type(a):
                if "submit" in a:
                    return "submit"
                if "str_replace_editor" in a:
                    return "edit"
                if any(x in a for x in ["pytest", "test"]):
                    return "validate"
                if any(x in a for x in ["view", "open_file"]):
                    return "inspect"
                if "grep" in a or "search" in a:
                    return "search"
                return "other"

            a_type = action_type(action)

            weights = {
                "search": 1,
                "inspect": 2,
                "edit": 5,
                "validate": 4,
                "submit": 6
            }

            score += weights.get(a_type, 0)

            def infer_stage(hist):
                if "pass" in hist or "all tests passing" in hist:
                    return "ready_to_submit"
                if any(k in hist for k in ["edited", "patch", "modified"]):
                    return "post_edit"
                if any(k in hist for k in ["line", "file", "def ", "class "]):
                    return "localized"
                if any(k in hist for k in ["error", "fail", "warning", "traceback"]):
                    return "diagnosing"
                return "start"

            stage = infer_stage(history)

            ideal = {
                "start": "search",
                "diagnosing": "search",
                "localized": "edit",
                "post_edit": "validate",
                "ready_to_submit": "submit"
            }

            if a_type == ideal.get(stage):
                score += 4
            else:
                score -= 2

            if trace and hasattr(trace, "steps") and len(trace.steps) >= 2:
                prev = str(trace.steps[-2].pred.action).lower()
                prev_type = action_type(prev)

                # reward forward transitions
                transitions = {
                    ("search", "inspect"): +2,
                    ("inspect", "edit"): +4,
                    ("edit", "validate"): +3,
                    ("validate", "submit"): +5
                }

                score += transitions.get((prev_type, a_type), 0)

                # penalize backward moves
                if prev_type == "edit" and a_type == "search":
                    score -= 4
                if prev_type == "validate" and a_type == "search":
                    score -= 3

            # penalize repititions
            if trace and hasattr(trace, "steps"):
                past = [str(s.pred.action).strip().lower() for s in trace.steps[:-1]]

                if action.strip() in past:
                    score -= 5

                # repeated grep penalty
                if a_type == "search":
                    recent = past[-3:]
                    if sum("grep" in x for x in recent) >= 2:
                        score -= 3

            # reward narrowing scope
            if a_type == "inspect" and "/" in action:
                score += 1  # file-level inspection

            if a_type == "edit" and "--old_str" in action:
                score += 2  # targeted edit

            
            if a_type == "submit":
                if stage == "ready_to_submit":
                    score += 6
                else:
                    score -= 8

            # stagnation
            if trace and hasattr(trace, "steps") and len(trace.steps) >= 4:
                last_types = [action_type(str(s.pred.action)) for s in trace.steps[-4:]]

                # no edit after many steps → bad
                if "edit" not in last_types:
                    score -= 3

            return score

        optimizer = BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=4,
            num_threads=1 
        )

        try:
            self.module = optimizer.compile(
                PlanActModule(),
                trainset=examples
            )
            print("✅ DSPY COMPILE COMPLETE")

        except Exception as e:
            print("⚠️ DSPY COMPILE FAILED — FALLBACK MODE")
            self.module = PlanActModule()


    def enforce_format(self, text):
        # Specific format based off YAMLs/SWE-agent
        text = str(text).strip()

        if not text.startswith("DISCUSSION"):
            text = "DISCUSSION\n" + text

        if "```" not in text:
            parts = text.split("\n", 2)
            discussion = parts[1] if len(parts) > 1 else "Fix the issue."
            command = parts[-1] if len(parts) > 2 else "grep -n TODO -R ."

            text = (
                "DISCUSSION\n"
                f"{discussion.strip()}\n\n"
                "```\n"
                f"{command.strip()}\n"
                "```"
            )

        return text

    def run(self, history):
        print("🔥 DSPY RUN")

        pred = self.module(history=history)

        plan = pred.plan
        action = pred.action

        print("🧠 PLAN:", plan)
        print("⚡ RAW ACTION:", action)

        formatted = self.enforce_format(action)

        return {"message": formatted}