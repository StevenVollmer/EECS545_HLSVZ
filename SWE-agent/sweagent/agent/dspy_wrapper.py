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
        
	# Fix these with better examples
        examples = [
            dspy.Example(
                history="ImportError: cannot import name 'parse'",
                plan="search for parse definition",
                action=(
                    "DISCUSSION\n"
                    "Search for where parse is defined to fix the import.\n\n"
                    "```\n"
                    "grep -n parse -R .\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="ModuleNotFoundError: utils.helpers",
                plan="find helpers module",
                action=(
                    "DISCUSSION\n"
                    "Locate the helpers module to fix the import path.\n\n"
                    "```\n"
                    "grep -n helpers -R .\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="AttributeError: visit_unknown missing",
                plan="locate method usage",
                action=(
                    "DISCUSSION\n"
                    "Search for visit_unknown to see where it should be implemented.\n\n"
                    "```\n"
                    "grep -n visit_unknown -R .\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="AssertionError: expected 5 but got 4",
                plan="find failing assertion",
                action=(
                    "DISCUSSION\n"
                    "Search for the failing assertion to inspect the logic.\n\n"
                    "```\n"
                    "grep -n AssertionError -R .\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="Test failed in test_parser.py line 42",
                plan="open test file",
                action=(
                    "DISCUSSION\n"
                    "Open the failing test file to inspect the issue.\n\n"
                    "```\n"
                    "open_file test_parser.py\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="Bug in logic",
                plan="edit code",
                action=(
                    "DISCUSSION\n"
                    "Edit the incorrect logic in the code.\n\n"
                    "```\n"
                    "str_replace_editor\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="All tests passing",
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
                history="Ran tests, failure in parser.py line 42",
                plan="inspect failing file",
                action=(
                    "DISCUSSION\n"
                    "Open the file causing the failure to inspect the issue.\n\n"
                    "```\n"
                    "open_file parser.py\n"
                    "```"
                )
            ).with_inputs("history"),

            dspy.Example(
                history="Edited code, tests still failing",
                plan="iterate fix",
                action=(
                    "DISCUSSION\n"
                    "Search for related logic to refine the fix.\n\n"
                    "```\n"
                    "grep -n parser -R .\n"
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
            # Rewards progression
            
            action = str(pred.action).lower()
            score = 0

            if any(tool in action for tool in VALID_TOOLS):
                score += 1

            if "error" in example.history.lower() and "grep" in action:
                score += 2

            if "test" in example.history.lower() and "open_file" in action:
                score += 2

            if "submit" in action:
                if "pass" in example.history.lower():
                    score += 5
                else:
                    score -= 5

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
