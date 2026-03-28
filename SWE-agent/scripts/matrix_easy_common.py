#!/usr/bin/env python3
"""Shared helpers for matrix_easy config generation and execution."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VariantSpec:
    name: str
    enable_planner: bool
    enable_reviewer: bool
    planner_size: str | None
    coder_size: str
    reviewer_size: str | None
    template_variant: str


def _build_variant_specs() -> list[VariantSpec]:
    specs = [
        VariantSpec(
            name="small_coder_only",
            enable_planner=False,
            enable_reviewer=False,
            planner_size=None,
            coder_size="small",
            reviewer_size=None,
            template_variant="small_coder_only",
        ),
        VariantSpec(
            name="big_coder_only",
            enable_planner=False,
            enable_reviewer=False,
            planner_size=None,
            coder_size="big",
            reviewer_size=None,
            template_variant="small_coder_only",
        ),
    ]

    for planner_size in ("small", "big"):
        for coder_size in ("small", "big"):
            specs.append(
                VariantSpec(
                    name=f"{planner_size}_planner_{coder_size}_coder",
                    enable_planner=True,
                    enable_reviewer=False,
                    planner_size=planner_size,
                    coder_size=coder_size,
                    reviewer_size=None,
                    template_variant="big_planner_small_coder",
                )
            )

    for planner_size in ("small", "big"):
        for coder_size in ("small", "big"):
            for reviewer_size in ("small", "big"):
                specs.append(
                    VariantSpec(
                        name=f"{planner_size}_planner_{coder_size}_coder_{reviewer_size}_reviewer",
                        enable_planner=True,
                        enable_reviewer=True,
                        planner_size=planner_size,
                        coder_size=coder_size,
                        reviewer_size=reviewer_size,
                        template_variant="big_planner_small_coder_big_reviewer",
                    )
                )
    return specs


def _build_default_variants() -> list[str]:
    # Keep the default sweep focused on the most decision-relevant comparisons:
    # coder-only baseline, planner/coder with the planner on the big slot,
    # and reviewer-enabled runs with the reviewer on the big slot.
    return [
        "small_coder_only",
        "big_coder_only",
        "big_planner_small_coder",
        "big_planner_big_coder",
        "big_planner_small_coder_big_reviewer",
        "big_planner_big_coder_big_reviewer",
    ]


VARIANT_SPECS = _build_variant_specs()
ALL_VARIANTS = [spec.name for spec in VARIANT_SPECS]
DEFAULT_VARIANTS = _build_default_variants()
VARIANTS = DEFAULT_VARIANTS
VARIANT_SPEC_BY_NAME = {spec.name: spec for spec in VARIANT_SPECS}

INSTANCE_SETS: dict[str, dict[str, Any]] = {
    "lite_default": {
        "description": "Default SWE-bench Lite dev slice from the variant YAMLs.",
    },
    "sweagent_easy_first": {
        "description": "SWE-agent's in-repo easiest known SWE-bench dev fixture.",
        "filter": "pydicom__pydicom-1458",
        "shuffle": False,
    },
    "4omini_smoke": {
        "description": "Single historically reachable issue for weak/small-model smoke tests.",
        "filter": "pylint-dev__astroid-1866",
        "shuffle": False,
    },
    "4omini_easy_pair": {
        "description": "Two historically most reachable Lite-dev issues in this repo's prior runs.",
        "filter": "pylint-dev__astroid-1866|pvlib__pvlib-python-1072",
        "shuffle": False,
    },
    "astroid_only": {
        "description": "Restrict to the strongest historically reachable astroid issue.",
        "filter": "pylint-dev__astroid-1866",
        "shuffle": False,
    },
}


@dataclass
class SlotModelSpec:
    role_model_name: str
    client_model_name: str
    api_base: str
    api_key: str | None = None
    max_input_tokens: int | None = 300000
    max_output_tokens: int | None = None
    litellm_model_registry: str | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return repo_root().parent


def config_dir() -> Path:
    return repo_root() / "config" / "custom_configs" / "matrix_easy"


def preset_config_path() -> Path:
    return config_dir() / "model_presets.yaml"


def config_path(variant: str) -> Path:
    return config_dir() / f"{variant}.yaml"


def default_sweagent_bin() -> Path:
    candidates = [
        project_root() / "env" / "bin" / "sweagent",
        project_root() / ".venv" / "bin" / "sweagent",
        repo_root() / "env" / "bin" / "sweagent",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_results_root() -> Path:
    return repo_root() / "matrix_easy_runs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_presets() -> dict[str, Any]:
    return load_yaml(preset_config_path())


def preset_names() -> list[str]:
    return sorted(load_presets().get("profiles", {}).keys())


def sweep_names() -> list[str]:
    return sorted(load_presets().get("sweeps", {}).keys())


def instance_set_names() -> list[str]:
    return sorted(INSTANCE_SETS.keys())


def resolve_instance_set(instance_set_name: str) -> dict[str, Any]:
    if instance_set_name not in INSTANCE_SETS:
        raise KeyError(f"Unknown instance set '{instance_set_name}'")
    return copy.deepcopy(INSTANCE_SETS[instance_set_name])


def resolve_profile(profile_name: str) -> dict[str, Any]:
    presets = load_presets().get("profiles", {})
    if profile_name not in presets:
        raise KeyError(f"Unknown model profile '{profile_name}'")
    return copy.deepcopy(presets[profile_name])


def resolve_sweep(sweep_name: str) -> list[str]:
    sweeps = load_presets().get("sweeps", {})
    if sweep_name not in sweeps:
        raise KeyError(f"Unknown matrix sweep '{sweep_name}'")
    return list(sweeps[sweep_name])


def _resolve_nonsecret_env(value: Any) -> Any:
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    env_var_name = value[1:]
    resolved = os.getenv(env_var_name)
    if not resolved:
        raise KeyError(f"Environment variable '{env_var_name}' is not set")
    return resolved


def build_slot_spec(raw: dict[str, Any]) -> SlotModelSpec:
    role_model_name = str(_resolve_nonsecret_env(raw["role_model_name"]))
    client_model_name = str(_resolve_nonsecret_env(raw.get("client_model_name", role_model_name)))
    api_base = str(_resolve_nonsecret_env(raw.get("api_base", "https://api.openai.com/v1")))
    return SlotModelSpec(
        role_model_name=role_model_name,
        client_model_name=client_model_name,
        api_base=api_base,
        api_key=raw.get("api_key"),
        max_input_tokens=raw.get("max_input_tokens", 300000),
        max_output_tokens=raw.get("max_output_tokens"),
        litellm_model_registry=raw.get("litellm_model_registry"),
    )


def build_profile(profile_name: str, overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, SlotModelSpec]:
    profile = resolve_profile(profile_name)
    if overrides:
        for slot_name, slot_overrides in overrides.items():
            profile.setdefault(slot_name, {})
            profile[slot_name].update({k: v for k, v in slot_overrides.items() if v is not None})
    return {slot_name: build_slot_spec(slot_cfg) for slot_name, slot_cfg in profile.items() if slot_name in {"small", "big"}}


def slot_override_from_args(prefix: str, args: Any) -> dict[str, Any]:
    return {
        "role_model_name": getattr(args, f"{prefix}_role_model", None),
        "client_model_name": getattr(args, f"{prefix}_client_model", None),
        "api_base": getattr(args, f"{prefix}_api_base", None),
        "api_key": getattr(args, f"{prefix}_api_key", None),
        "max_input_tokens": getattr(args, f"{prefix}_max_input_tokens", None),
    }


def role_model_config(spec: SlotModelSpec) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": spec.client_model_name,
        "api_base": spec.api_base,
    }
    if spec.api_key is not None:
        config["api_key"] = spec.api_key
    if spec.max_input_tokens is not None:
        config["max_input_tokens"] = spec.max_input_tokens
    if spec.max_output_tokens is not None:
        config["max_output_tokens"] = spec.max_output_tokens
    if spec.litellm_model_registry is not None:
        config["litellm_model_registry"] = spec.litellm_model_registry
    return config


def _replace_bundle_paths(base: dict[str, Any]) -> None:
    bundles = base.get("agent", {}).get("tools", {}).get("bundles", [])
    if not isinstance(bundles, list):
        return
    new_bundles: list[dict[str, Any]] = []
    replaced_edit_bundle = False
    for bundle in bundles:
        if not isinstance(bundle, dict):
            new_bundles.append(bundle)
            continue
        path = str(bundle.get("path", ""))
        if path.endswith("tools/edit_anthropic"):
            replaced_edit_bundle = True
            for replacement_path in ("tools/windowed", "tools/windowed_edit_linting"):
                replacement = dict(bundle)
                replacement["path"] = replacement_path
                new_bundles.append(replacement)
            continue
        new_bundles.append(bundle)
    if replaced_edit_bundle:
        base["agent"]["tools"]["bundles"] = new_bundles
        env_variables = base["agent"]["tools"].setdefault("env_variables", {})
        env_variables.setdefault("WINDOW", "100")
        env_variables.setdefault("OVERLAP", "2")
        env_variables.setdefault("FIRST_LINE", "0")
        env_variables.setdefault("CURRENT_FILE", "")


def _modernize_matrix_templates(base: dict[str, Any]) -> None:
    roles = base.get("agent", {}).get("roles", {})
    if not isinstance(roles, dict):
        return
    planner = roles.get("planner")
    coder = roles.get("coder")
    reviewer = roles.get("reviewer")

    if isinstance(planner, dict):
        planner["system_template"] = """{% if enable_planner %}
You are the planner.
Your job is to localize the bug and hand the coder a tight execution contract.
Do not edit files. Do not write code. Do not write plan.txt.
Use read-only commands only.

Every response must follow this format exactly:
DISCUSSION
One short sentence.

```
single command here
```

Rules:
- Output exactly one DISCUSSION header.
- Output exactly one fenced code block.
- Put exactly one command inside the code block.
- If ready to hand off, use `handoff '<json payload>'`.
- The handoff must be compact JSON on one line.
- Recommended keys: `problem_summary`, `root_cause_hypothesis`, `files_likely_affected`, `target_symbols`, `required_validations`, `forbidden_edits`.
- Focus on the smallest likely edit surface.
- Prefer `grep -n`, `open`, and `goto` over broad repo scans.
- Do not propose unrelated refactors.
{% else %}
Planner is disabled.
{% endif %}"""
        planner["instance_template"] = """{% if enable_planner %}
Repository root: {{working_dir}}

Problem statement:
<problem_statement>
{{problem_statement}}
</problem_statement>

Requirements:
- Identify the likely target file or files first.
- Name the symbols or lines the coder should inspect.
- Tell the coder what validation to run after editing.
- Hand off as soon as you have a concrete execution plan.
{% else %}
Planner is disabled.
{% endif %}"""
        planner["next_step_template"] = """OBSERVATION:
{{observation}}
Reminder: respond with exactly one DISCUSSION block and one command.
{% if enable_planner %}
If you know the target file, likely root cause, and validation, hand off now.
{% endif %}
(Open: {{open_file}})
(Dir: {{working_dir}})
bash-$"""
        planner["next_step_no_output_template"] = planner["next_step_template"]

    if isinstance(coder, dict):
        coder["system_template"] = """You are the coder.
{% if enable_planner %}
Read and follow the planner contract in `handoff.txt`.
Treat it as the default execution plan unless direct code evidence disproves it.
{% else %}
No planner is available for this run. You must localize the bug yourself from the issue text.
{% endif %}
{% if enable_reviewer %}
A reviewer is enabled for this run. When the patch is ready, hand off to review instead of submitting directly.
{% else %}
No reviewer is enabled for this run. You are responsible for deciding when the patch is ready.
{% endif %}

You should work in a short loop:
1. localize
2. inspect exact lines
3. edit narrowly
4. validate
5. inspect diff
6. only then submit or hand off

Every response must follow this format exactly:
DISCUSSION
One short sentence.

```
single command here
```

Rules:
- Output exactly one DISCUSSION header.
- Output exactly one fenced code block.
- Put exactly one command inside the code block.
- Never use terminal editors such as `vim`, `vi`, `nano`, or `emacs`.
- Use the windowed editing tools for code changes: `open`, `goto`, `edit`.
- Use `open <path> [line]` to inspect a file.
- Use `goto <line>` to move the window.
- Use `edit <start_line>:<end_line>` followed by replacement text and `end_of_edit` to replace a line range.
- Prefer editing existing relevant files over creating new files.
- Do not create a new file unless the issue clearly requires it.
- Do not submit without a meaningful diff.
- Before submit or handoff, run `git diff --stat` or `git diff --name-only` and inspect the changed files.
- Before submit or handoff, run at least one targeted validation after the last edit.
- If an edit fails or lands in the wrong place, re-open the file and use a smaller line range instead of retrying blindly.
- Prefer `grep -n`, `open`, and `goto` over broad scans once you know the likely file.
- Stay on the smallest relevant set of files.
{% if enable_reviewer %}
- When implementation is ready, use `handoff '<json payload>'` for the reviewer.
- Include: `change_summary`, `files_changed`, `tests_run`, `test_results`, `open_risks`.
{% else %}
- Only use `submit` when the patch is non-empty, relevant, and validated.
{% endif %}"""
        coder["instance_template"] = """{% if enable_planner %}
The planner prepared `{{working_dir}}/handoff.txt` for you.
- Your first step must be to read it.
- Do not re-read it unless it changes.
{% endif %}

Problem statement:
<problem_statement>
{{problem_statement}}
</problem_statement>

Execution rules:
- First localize the bug to the most likely file.
- Open the exact region before editing.
- Make small edits.
- Validate after editing.
- Check the diff before submit or review handoff.
- Never submit an empty diff or a patch that only creates unrelated files.

(Open: {{open_file}})
(Dir: {{working_dir}})
bash-$"""
        coder["next_step_template"] = """OBSERVATION:
{{observation}}
Reminder: respond with exactly one DISCUSSION block and one command.
{% if enable_planner %}
Do not re-read `handoff.txt` unless it changed.
{% endif %}
If the last edit failed, narrow the edit range before trying again.
Before submit or handoff, confirm the diff is relevant and validation ran after the final edit.
{% if enable_reviewer %}
If ready, use `handoff '<json payload>'`.
{% else %}
If ready, use `submit`.
{% endif %}
(Open: {{open_file}})
(Dir: {{working_dir}})
bash-$"""
        coder["next_step_no_output_template"] = coder["next_step_template"]

    if isinstance(reviewer, dict):
        reviewer["system_template"] = """{% if enable_reviewer %}
You are the reviewer.
Your job is to decide whether the coder's patch is ready.

Every response must follow this format exactly:
DISCUSSION
One short sentence.

```
single command here
```

Rules:
- Output exactly one DISCUSSION header.
- Output exactly one fenced code block.
- Put exactly one command inside the code block.
- Read `handoff.txt` before reviewing.
- Inspect the diff and run targeted validation.
- If the patch is acceptable, the command must be exactly `submit`.
- If the patch is not acceptable, use `handoff '<json payload>'` and return control.
- Prefer `git diff`, targeted file inspection, and targeted tests over broad exploration.
- Reject empty diffs, irrelevant diffs, or unvalidated patches.
{% else %}
Reviewer is disabled.
{% endif %}"""
        reviewer["instance_template"] = """{% if enable_reviewer %}
Reviewer instructions:
- Read `{{working_dir}}/handoff.txt`.
- Inspect current changes with `git diff`.
- Validate the changed behavior.
- If the patch is good, submit.
- Otherwise hand off clear feedback.

(Open: {{open_file}})
(Dir: {{working_dir}})
bash-$
{% else %}
Reviewer is disabled.
{% endif %}"""
        reviewer["next_step_template"] = """OBSERVATION:
{{observation}}
Reminder: respond with exactly one DISCUSSION block and one command.
{% if enable_reviewer %}
If the patch is good, use `submit`.
If not, use `handoff '<json payload>'`.
{% endif %}
(Open: {{open_file}})
(Dir: {{working_dir}})
bash-$"""
        reviewer["next_step_no_output_template"] = reviewer["next_step_template"]


def build_variant_config(
    variant: str,
    profile_name: str,
    output_root: Path,
    slot_overrides: dict[str, dict[str, Any]] | None = None,
    instance_set_name: str | None = None,
    instance_filter: str | None = None,
    instance_slice: str | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    spec = VARIANT_SPEC_BY_NAME[variant]
    base = load_yaml(config_path(spec.template_variant))
    _replace_bundle_paths(base)
    _modernize_matrix_templates(base)
    profile = build_profile(profile_name, slot_overrides)

    agent = base["agent"]
    agent["enable_planner"] = spec.enable_planner
    agent["enable_reviewer"] = spec.enable_reviewer

    role_sizes = {
        "planner": spec.planner_size or "big",
        "coder": spec.coder_size,
        "reviewer": spec.reviewer_size or "big",
    }
    for role_name, slot_name in role_sizes.items():
        slot_spec = profile[slot_name]
        agent[role_name] = slot_spec.role_model_name
        agent[f"{role_name}_model_config"] = role_model_config(slot_spec)

    shared_spec = profile["big"]
    agent["model"]["name"] = shared_spec.client_model_name
    agent["model"]["api_base"] = shared_spec.api_base
    agent["model"]["per_instance_call_limit"] = 50
    if shared_spec.api_key is not None:
        agent["model"]["api_key"] = shared_spec.api_key
    if shared_spec.max_input_tokens is not None:
        agent["model"]["max_input_tokens"] = shared_spec.max_input_tokens
    if shared_spec.max_output_tokens is not None:
        agent["model"]["max_output_tokens"] = shared_spec.max_output_tokens
    if shared_spec.litellm_model_registry is not None:
        agent["model"]["litellm_model_registry"] = shared_spec.litellm_model_registry

    if instance_set_name is not None:
        instance_set = resolve_instance_set(instance_set_name)
        instances_cfg = base["instances"]
        if "filter" in instance_set:
            instances_cfg["filter"] = instance_set["filter"]
            instances_cfg.pop("slice", None)
        if "shuffle" in instance_set:
            instances_cfg["shuffle"] = instance_set["shuffle"]

    if instance_filter is not None:
        base["instances"]["filter"] = instance_filter
        base["instances"].pop("slice", None)

    if instance_slice is not None:
        base["instances"]["slice"] = instance_slice

    if num_workers is not None:
        base["num_workers"] = num_workers

    base["output_dir"] = str(output_root / variant)
    return base
