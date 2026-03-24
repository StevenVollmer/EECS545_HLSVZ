import dspy
import re
from dspy.teleprompt import BootstrapFewShotWithRandomSearch


# =========================
# SIGNATURES (PLAIN TEXT ONLY)
# =========================

class PlanSignature(dspy.Signature):
    """Create a debugging plan."""
    history = dspy.InputField()
    plan = dspy.OutputField(
        desc="Return plain text only. No JSON. No brackets."
    )


class ActSignature(dspy.Signature):
    """Generate next action."""
    history = dspy.InputField()
    plan = dspy.InputField()
    action = dspy.OutputField(
        desc="Return ONE valid SWE-agent command. Plain text only."
    )


# =========================
# MODULE (CRASH-PROOF)
# =========================

class PlanActModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.Predict(PlanSignature)
        self.act = dspy.Predict(ActSignature)

    def safe_extract(self, pred, field):
        # completely bypass DSPy structured parsing issues
        if isinstance(pred, str):
            return pred

        if hasattr(pred, field):
            return getattr(pred, field)

        return str(pred)

    def clean(self, text):
        text = str(text)
        text = re.sub(r"\[\[.*?\]\]", "", text)  # remove weird tags
        text = text.replace("{", "").replace("}", "")
        return text.strip()

    def forward(self, history):
        plan_pred = self.plan(history=history)
        plan = self.clean(self.safe_extract(plan_pred, "plan"))

        act_pred = self.act(history=history, plan=plan)
        action = self.clean(self.safe_extract(act_pred, "action"))

        return dspy.Prediction(plan=plan, action=action)


# =========================
# LITELLM ADAPTER (STRING SAFE)
# =========================

class LiteLLMAdapter(dspy.LM):
    def __init__(self, model):
        self.client = model
        self.model = "openai/gpt-4"
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        messages = kwargs.get("messages", None)

        if messages is None and len(args) > 0:
            messages = args[0]

        if messages is None:
            return ""

        result = self.client.query(messages)

        # normalize EVERYTHING to string
        if isinstance(result, str):
            text = result
        elif hasattr(result, "choices"):
            text = result.choices[0].message.content
        elif isinstance(result, dict) and "choices" in result:
            text = result["choices"][0]["message"]["content"]
        else:
            text = str(result)

        return text.strip()


# =========================
# WRAPPER
# =========================

class DSPyPolicyWrapper:
    def __init__(self, model):
        print("🚀 DSPY WRAPPER INITIALIZING")

        lm = LiteLLMAdapter(model)

        # 🔥 FORCE NON-JSON MODE
        dspy.settings.configure(
            lm=lm,
            adapter=dspy.adapters.ChatAdapter()
        )
        dspy.settings.experimental = True

        # =========================
        # TRAINING DATA
        # =========================
        examples = [

            dspy.Example(
                history="ImportError: cannot import name 'parse'",
                plan="search parse, inspect import, fix path",
                action="grep -n parse -R ."
            ).with_inputs("history"),

            dspy.Example(
                history="ModuleNotFoundError: utils.helpers",
                plan="locate helpers module and fix import",
                action="grep -n helpers -R ."
            ).with_inputs("history"),

            dspy.Example(
                history="AttributeError: visit_unknown missing",
                plan="search method and implement",
                action="grep -n visit_unknown -R ."
            ).with_inputs("history"),

            dspy.Example(
                history="AssertionError: expected 5 but got 4",
                plan="locate failing logic",
                action="grep -n AssertionError -R ."
            ).with_inputs("history"),

            dspy.Example(
                history="Test failed in test_parser.py line 42",
                plan="open test file",
                action="open_file test_parser.py"
            ).with_inputs("history"),

            dspy.Example(
                history="Bug in logic",
                plan="edit incorrect code",
                action="str_replace_editor"
            ).with_inputs("history"),

            dspy.Example(
                history="All tests passing",
                plan="submit solution",
                action="submit"
            ).with_inputs("history"),
        ]

        # =========================
        # METRIC (LIGHTWEIGHT)
        # =========================
        VALID_TOOLS = ["grep", "open_file", "str_replace_editor", "submit"]

        def metric(example, pred, trace=None):
            action = str(pred.action).lower()

            score = 0

            if any(tool in action for tool in VALID_TOOLS):
                score += 2

            for word in example.history.lower().split():
                if word in action:
                    score += 1

            if action.strip() == example.action.lower():
                score += 5

            if "submit" in action and "passing" not in example.history.lower():
                score -= 3

            return score

        # =========================
        # OPTIMIZER (STABLE)
        # =========================
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=4,
            num_threads=1  # 🔥 prevents parallel crash
        )

        # =========================
        # SAFE COMPILE
        # =========================
        try:
            self.module = optimizer.compile(
                PlanActModule(),
                trainset=examples
            )
            print("✅ DSPY COMPILE COMPLETE")

        except Exception as e:
            print("⚠️ DSPY COMPILE FAILED — FALLBACK MODE")
            print(e)
            self.module = PlanActModule()

    # =========================
    # RUN
    # =========================

    def run(self, history):
        print("🔥 DSPY RUN")

        pred = self.module(history=str(history))

        plan = str(pred.plan)
        action = str(pred.action)

        print("🧠 PLAN:", plan)
        print("⚡ ACTION:", action)

        return {"message": action}
