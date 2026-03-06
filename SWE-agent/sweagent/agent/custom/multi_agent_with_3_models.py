"""
Three-role multi-agent implementation built on top of DefaultAgent.

This follows the same shape as the existing custom multi-agent agents:
- role-local histories
- normal SWE-agent tool loop for each role
- explicit role transitions driven by tool actions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template
from typing_extensions import Self

from sweagent.agent.agents import DefaultAgent, MultiAgentConfigThreeModels, TemplateConfig
from sweagent.agent.models import get_model
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.tools.tools import ToolHandler
from sweagent.types import StepOutput
from sweagent.utils.log import get_logger


@dataclass
class RoleTemplatesConfig:
    roles: dict[str, TemplateConfig] = field(default_factory=dict)


class Role:
    def __init__(self, name: str, template: TemplateConfig):
        self.name = name
        self.system_template = template.system_template
        self.instance_template = template.instance_template
        self.next_step_template = template.next_step_template
        self.next_step_no_output_template = template.next_step_no_output_template
        self.next_step_truncated_observation_template = template.next_step_truncated_observation_template
        self.max_observation_length = template.max_observation_length
        self.demonstrations = template.demonstrations
        self.demonstration_template = template.demonstration_template
        self.put_demos_in_history = template.put_demos_in_history
        self.history = []

    def reset(self) -> None:
        self.history.clear()


class MultiAgent(DefaultAgent):
    role_templates: RoleTemplatesConfig | None = None
    roles: dict[str, Role]

    def __init__(
        self,
        *args,
        role_templates: RoleTemplatesConfig | None = None,
        models: dict | None = None,
        enable_planner: bool = True,
        enable_reviewer: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.role_templates = role_templates
        self.models = models or {}
        self.enable_planner = enable_planner
        self.enable_reviewer = enable_reviewer
        self.role_order: list[str] = []
        self.current_role: Role | None = None

    @property
    def current_role_name(self) -> str:
        if self.current_role is None:
            return ""
        return self.current_role.name

    @classmethod
    def from_config(cls, config: MultiAgentConfigThreeModels) -> Self:
        config = config.model_copy(deep=True)
        models = {
            "planner": get_model(config.model.model_copy(update={"name": config.planner}), config.tools),
            "coder": get_model(config.model.model_copy(update={"name": config.coder}), config.tools),
            "reviewer": get_model(config.model.model_copy(update={"name": config.reviewer}), config.tools),
        }
        role_templates = RoleTemplatesConfig(roles=config.roles)

        return cls(
            templates=config.templates,
            tools=ToolHandler(config.tools),
            history_processors=config.history_processors,
            model=models["planner"],
            models=models,
            enable_planner=config.enable_planner,
            enable_reviewer=config.enable_reviewer,
            max_requeries=config.max_requeries,
            action_sampler_config=config.action_sampler,
            role_templates=role_templates,
        )

    def _init_roles(self) -> None:
        assert self.role_templates is not None
        self.roles = {}
        self.role_histories = {}
        self.role_loggers = {}
        emojis = {"planner": "🗓️", "coder": "💻", "reviewer": "👓"}

        enabled_roles = {"coder"}
        if self.enable_planner:
            enabled_roles.add("planner")
        if self.enable_reviewer:
            enabled_roles.add("reviewer")

        for role_name, template in self.role_templates.roles.items():
            if role_name not in enabled_roles:
                continue
            role = Role(role_name, template)
            role.reset()
            self.roles[role_name] = role
            self.role_histories[role_name] = role.history
            self.role_loggers[role_name] = get_logger(role_name, emoji=emojis.get(role_name, "🧠"))

        self.role_order = [name for name in ["planner", "coder", "reviewer"] if name in self.roles]
        self.current_role = self.roles[self.role_order[0]]

    def _get_format_dict(self, **kwargs) -> dict[str, Any]:
        return super()._get_format_dict(
            enable_planner=self.enable_planner,
            enable_reviewer=self.enable_reviewer,
            active_roles=",".join(getattr(self, "role_order", [])),
            **kwargs,
        )

    def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        super().setup(env, problem_statement, output_dir)
        self.history.clear()
        self._init_roles()
        for role in self.roles.values():
            self._init_role_history(role)

    def _init_role_history(self, role: Role) -> None:
        self.add_system_message_to_role_history(role)
        self.add_demonstrations_to_role_history(role)
        self.add_instance_template_to_role_history(state=self.tools.get_state(self._env), role=role)  # type: ignore[arg-type]

    def _append_role_history(self, item: dict[str, Any], role: Role) -> None:
        self._chook.on_query_message_added(**item)
        item.setdefault("role_name", role.name)
        item.setdefault("multi_agent", True)
        role.history.append(item)  # type: ignore[arg-type]

    def add_system_message_to_role_history(self, role: Role) -> None:
        assert self._problem_statement is not None
        system_msg = Template(role.system_template).render(**self._get_format_dict())
        self.logger.info("SYSTEM (%s)\n%s", role.name, system_msg)
        self._append_role_history(
            {"role": "system", "content": system_msg, "agent": self.name, "message_type": "system_prompt"},
            role,
        )

    def add_demonstrations_to_role_history(self, role: Role) -> None:
        for demonstration_path in role.demonstrations:
            self._add_demonstration_to_role_history(role, demonstration_path)

    def _add_demonstration_to_role_history(self, role: Role, demonstration_path: Path) -> None:
        if role.demonstration_template is None and not role.put_demos_in_history:
            raise ValueError("Cannot use demonstrations without a demonstration template or put_demos_in_history=True")

        self.logger.info("DEMONSTRATION: %s", demonstration_path)
        demo_text = Path(demonstration_path).read_text()
        if demonstration_path.suffix == ".yaml":
            demo_history = yaml.safe_load(demo_text)["history"]
        else:
            demo_history = json.loads(demo_text)["history"]

        if self.templates.put_demos_in_history:
            for entry in demo_history:
                if entry["role"] != "system":
                    entry["is_demo"] = True
                    self._append_role_history(entry, role)
        else:
            demo_history = [entry for entry in demo_history if entry["role"] != "system"]
            demo_message = "\n".join(entry["content"] for entry in demo_history)
            assert role.demonstration_template is not None
            demonstration = Template(role.demonstration_template).render(demonstration=demo_message)
            self._append_role_history(
                {
                    "agent": self.name,
                    "content": demonstration,
                    "is_demo": True,
                    "role": "user",
                    "message_type": "demonstration",
                },
                role,
            )

    def add_instance_template_to_role_history(self, state: dict[str, str], role: Role) -> None:
        assert role.history[-1]["role"] == "system" or role.history[-1].get("is_demo", False)
        self._add_role_templated_messages_to_history([role.instance_template], role, **state)  # type: ignore[arg-type]

    def _add_role_templated_messages_to_history(
        self,
        templates: list[str],
        role: Role,
        tool_call_ids: list[str] | None = None,
        **kwargs: str | int | None,
    ) -> None:
        messages = []
        format_dict = self._get_format_dict(**kwargs)
        for template in templates:
            messages.append(Template(template).render(**format_dict))

        message = "\n".join(messages)
        self.logger.info("🤖 %s MODEL INPUT\n%s", role.name.upper(), message, extra={"highlighter": None})
        history_item: dict[str, Any] = {
            "role": "user",
            "content": message,
            "agent": self.name,
            "message_type": "observation",
        }
        if tool_call_ids:
            assert len(tool_call_ids) == 1
            history_item["role"] = "tool"
            history_item["tool_call_ids"] = tool_call_ids
        self._append_role_history(history_item, role)

    def add_step_to_role_history(self, step: StepOutput, role: Role) -> None:
        self._append_role_history(
            {
                "role": "assistant",
                "content": step.output,
                "thought": step.thought,
                "action": step.action,
                "agent": self.name,
                "tool_calls": step.tool_calls,
                "message_type": "action",
                "thinking_blocks": step.thinking_blocks,
            },
            role,
        )

        if step.observation.strip() == "":
            templates = [role.next_step_no_output_template]
            elided_chars = 0
        elif len(step.observation) > role.max_observation_length:
            templates = [role.next_step_truncated_observation_template]
            elided_chars = len(step.observation) - role.max_observation_length
        else:
            templates = [role.next_step_template]
            elided_chars = 0

        self._add_role_templated_messages_to_history(
            templates,
            role,
            observation=step.observation,
            elided_chars=elided_chars,
            max_observation_length=role.max_observation_length,
            tool_call_ids=step.tool_call_ids,
            **(step.state or {}),
        )

    def get_active_history(self) -> list[dict[str, Any]]:
        assert self.current_role is not None
        return self.current_role.history

    def _switch_to_role(self, next_role_name: str) -> None:
        assert self.current_role is not None
        previous = self.current_role.name
        self.current_role = self.roles[next_role_name]
        self.logger.info("ROLE SWITCH %s -> %s", previous, next_role_name.upper())

    def _repo_file_path(self, filename: str) -> str:
        assert self._env is not None
        if self._env.repo is None:
            return f"/{filename}"
        return f"/{self._env.repo.repo_name}/{filename}"

    def _read_handoff_payload(self) -> dict[str, Any] | None:
        assert self._env is not None
        try:
            payload_text = self._env.read_file(self._repo_file_path("handoff.txt"), encoding="utf-8", errors="backslashreplace")
        except FileNotFoundError:
            return None
        except Exception as exc:
            self.logger.warning("Failed to read handoff.txt: %s", exc)
            return None

        payload_text = payload_text.strip()
        if not payload_text:
            return None
        try:
            data = json.loads(payload_text)
        except json.JSONDecodeError:
            self.logger.warning("handoff.txt is not valid JSON")
            return None
        if not isinstance(data, dict):
            self.logger.warning("handoff.txt JSON payload must be an object")
            return None
        return data

    def _handle_disabled_reviewer_handoff(self, step_output: StepOutput) -> StepOutput:
        if self.enable_reviewer:
            return step_output
        assert self.current_role is not None
        if self.current_role.name != "coder":
            return step_output
        if "handoff" not in (step_output.action or "").lower():
            return step_output
        self.logger.info("Reviewer disabled: converting coder handoff into submission")
        return self.attempt_autosubmission_after_error(step_output)

    def _handle_reviewer_handoff_policy(self, step_output: StepOutput) -> StepOutput:
        assert self.current_role is not None
        if self.current_role.name != "reviewer":
            return step_output
        if "handoff" not in (step_output.action or "").lower():
            return step_output

        payload = self._read_handoff_payload() or {}
        next_role = str(payload.get("next_role", "")).lower().strip()
        valid_targets = {"coder"}
        if self.enable_planner:
            valid_targets.add("planner")

        if next_role in valid_targets:
            return step_output

        self.logger.info("Reviewer handoff missing valid next_role; stopping run instead of continuing")
        step_output = step_output.model_copy(deep=True)
        step_output.done = True
        if not step_output.exit_status:
            step_output.exit_status = "review_stopped"
        message = (
            "Reviewer did not request another agent turn. "
            "Stopping run. Include next_role=coder or next_role=planner in handoff JSON to continue."
        )
        if step_output.observation:
            step_output.observation = f"{step_output.observation}\n{message}"
        else:
            step_output.observation = message
        return step_output

    def advance_current_role(self, step_output: StepOutput) -> None:
        assert self.current_role is not None
        action = (step_output.action or "").lower()
        observation = step_output.observation or ""

        if "handoff" not in action:
            return
        if "error:" in observation.lower():
            self.logger.info("Handoff failed for role %s", self.current_role.name)
            return

        if self.current_role.name == "planner":
            self._switch_to_role("coder")
        elif self.current_role.name == "coder":
            if self.enable_reviewer:
                self._switch_to_role("reviewer")
        elif self.current_role.name == "reviewer":
            payload = self._read_handoff_payload() or {}
            next_role = str(payload.get("next_role", "coder")).lower()
            if next_role == "planner" and self.enable_planner:
                self._switch_to_role("planner")
            else:
                self._switch_to_role("coder")

    def run_role_forward(self, role: Role) -> StepOutput:
        self.history = role.history
        self.model = self.models[role.name]
        return self.forward_with_handling(self.messages)

    def concat_histories(self) -> None:
        roles_list = list(self.roles.values())
        if not roles_list:
            self.history = []
            return
        self.history = roles_list[0].history
        for role in roles_list[1:]:
            self.history += role.history

    def step(self) -> StepOutput:
        assert self._env is not None
        assert self.current_role is not None
        self._chook.on_step_start()

        n_step = len(self.trajectory) + 1
        self.logger.info("%s STEP %s %s", "=" * 25, n_step, "=" * 25)

        step_output = self.run_role_forward(self.current_role)
        step_output = self._handle_disabled_reviewer_handoff(step_output)
        step_output = self._handle_reviewer_handoff_policy(step_output)
        self.add_step_to_role_history(step_output, self.current_role)

        self.info["submission"] = step_output.submission
        self.info["exit_status"] = step_output.exit_status  # type: ignore[assignment]
        self.info.update(self._get_edited_files_with_context(patch=step_output.submission or ""))  # type: ignore[arg-type]
        self.info["model_stats"] = self.model.stats.model_dump()
        self.add_step_to_trajectory(step_output)

        if len(self.role_order) > 1 and not step_output.done:
            self.advance_current_role(step_output)

        if step_output.done:
            self.concat_histories()

        self._chook.on_step_done(step=step_output, info=self.info)
        return step_output
