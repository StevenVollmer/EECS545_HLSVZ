"""
Docstring for Project.SWE-agent.sweagent.agent.custom.multi_agent

#CUSTOM class created for experimental agent architecture. We will implement an agent
    that includes a planner, coder, and reviewer. Each will have its own history and may require
    its own parser to read its output and convert to an appropriate file format. This was adapted
    from ShellAgent in extra/shell_agent.py

Example useage:
sweagent run \
    --config SWE-agent/config/custom_configs/multi_agent.yaml \
    --env.repo.github_url=https://github.com/SWE-agent/test-repo \
    --problem_statement.text="Add a simple hello world function to a new file named hello.py."

sweagent run \
    --config SWE-agent/config/custom_configs/multi_agent_planner_4.yaml \
    --env.repo.github_url=https://github.com/SWE-agent/test-repo \
    --problem_statement.text="Add a simple hello world function to a new file named hello.py."
"""

import pprint
from jinja2 import Template
from dataclasses import dataclass
from pathlib import Path
from sweagent.environment.swe_env import SWEEnv
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.utils.log import get_logger
from typing_extensions import Self
from sweagent.tools.tools import ToolHandler
from typing import Any
from sweagent.types import (StepOutput)
from sweagent.agent.agents import DefaultAgent, TemplateConfig, MultiAgentConfig
from sweagent.agent.models import get_model
from dataclasses import dataclass, field


@dataclass
class RoleTemplatesConfig:
    # 'roles' is a dictionary mapping string keys to TemplateConfig objects
    roles: dict[str, TemplateConfig] = field(default_factory=dict)


class Role:
    def __init__(self, name, template: TemplateConfig):
        self.name = name
        self.system_template = template.system_template
        self.instance_template = template.instance_template
        self.next_step_template = template.next_step_template
        self.next_step_no_output_template = template.next_step_no_output_template
        self.next_step_truncated_observation_template = (template.next_step_truncated_observation_template)
        self.max_observation_length = template.max_observation_length
        self.history = []

    def reset(self):
        self.history.clear()


class MultiAgent(DefaultAgent):
    role_templates: RoleTemplatesConfig | None = None
    roles: dict[str, Role]

    def __init__(
        self, *args, role_templates: RoleTemplatesConfig | None = None, **kwargs
    ):
        """The agent handles the behaviour of the model and how it interacts with the environment.
        To run the agent, either call `self.run` or `self.setup` and then `self.step` in a loop.
        """
        super().__init__(*args, **kwargs)
        self.role_templates = role_templates
        # TODO may need to change parsing functions , for example to extract list from planner's output
        # if isinstance(self.model, HumanThoughtModel):
        #     self.tools.config.parse_function = ThoughtActionParser()
        # elif isinstance(self.model, HumanModel):
        #     self.tools.config.parse_function = ActionOnlyParser()

    @property
    def current_role_name(self) -> str:
        return self.current_role.name

    @classmethod
    def from_config(cls, config: MultiAgentConfig) -> Self:
        config = config.model_copy(deep=True)
        model = get_model(config.model, config.tools)
        role_templates = RoleTemplatesConfig(roles=config.roles)

        return cls(
            templates=next(
                iter(role_templates.roles.values())
            ),  # not used, but avoids creating error with parent class DefaultAgent
            tools=ToolHandler(config.tools),
            history_processors=config.history_processors,
            model=model,
            max_requeries=config.max_requeries,
            action_sampler_config=config.action_sampler,
            role_templates=role_templates,
        )

    def _init_roles(self):
        assert self.role_templates is not None
        self.roles = {}
        self.role_histories = {}
        self.role_loggers = {}
        emojis = {"planner": "🗓️", "coder": "💻", "reviewer": "👓"}

        # role_templates.roles is a dictionary with roles defined in the config.yaml
        for (role_name, template) in (self.role_templates.roles.items()):
            role = Role(role_name, template)
            role.reset()
            self.roles[role_name] = role
            self.role_histories[role_name] = role.history
            self.role_loggers[role_name] = get_logger(
                role_name, emoji=emojis.get(role_name, "🧠"))

        self.role_order = list(self.roles.keys())
        self.current_role = self.roles[self.role_order[0]]

    def setup(self, env: SWEEnv, problem_statement: ProblemStatement | ProblemStatementConfig, output_dir: Path = Path(".")) -> None:
        """Setup the agent for a new instance. This includes formatting the system message and adding demonstrations to the history.
        NOTE: This mirrors DefaultAgent.add_instance_message_to_history
        but writes to role-local history instead of self.history
        This method is called by `self.run`.
        """
        # setup everything as if DefaulatAgent
        super().setup(env, problem_statement, output_dir)
        # clear history after DefaultAgent setup
        self.history.clear()
        # Initialize roles, role histories are kept separate
        self._init_roles()
        # Populate role histories
        for role in self.roles.values():
            self._init_role_history(role)

    def _init_role_history(self, role: Role):
        self.add_system_message_to_role_history(role)
        self.add_role_instance_template_to_history(
            state=self.tools.get_state(self._env), role=role)

    def _append_role_history(self, item: dict[str, Any], role: Role) -> None:
        """Adds an item to the role specific history.
        NOTE: This mirrors DefaultAgent.add_instance_message_to_history
        but writes to role-local history instead of self.history"""
        self._chook.on_query_message_added(**item)
        item.setdefault("role_name", role.name)
        item.setdefault("multi_agent", True)
        role.history.append(item)  # type: ignore

    def add_system_message_to_role_history(self, role: Role) -> None:
        """Add system message to role specific history
        NOTE: This mirrors DefaultAgent.add_system_message_to_history
        but writes to role-local history instead of self.history"""

        assert self._problem_statement is not None
        system_msg = Template(role.system_template).render(
            **self._get_format_dict())
        self.logger.info(f"SYSTEM ({role.name})\n{system_msg}")
        self._append_role_history(
            {
                "role": "system",
                "content": system_msg,
                "agent": self.name,
                "message_type": "system_prompt"
            },
            role)

    def add_role_instance_template_to_history(self, state: dict[str, str], role: Role) -> None:
        """Add observation to role specific history, as well as the instance template or demonstrations if we're
        at the start of a new attempt.
        NOTE: This mirrors DefaultAgent.add_instance_message_to_history
        but writes to role-local history instead of self.history
        """
        templates: list[str] = []
        # Determine observation template based on what prior observation was
        assert role.history[-1]["role"] == "system" or role.history[-1].get(
            "is_demo", False)
        # Show instance template if prev. obs. was initial system message
        templates = [role.instance_template]

        self._add_role_templated_messages_to_history(
            templates, role, **state)  # type: ignore

    def _add_role_templated_messages_to_history(
        self,
        templates: list[str],
        role: Role,
        tool_call_ids: list[str] | None = None,
        **kwargs: str | int | None,
    ) -> None:
        """Populate selected template(s) with information (e.g., issue, arguments, state) and add to history.
        NOTE: This mirrors DefaultAgent but writes to role-local history instead of self.history

        Args:
            templates: templates to populate and add to history
            role: specific multi-agent role (i.e. planner, coder, reviewer)
            tool_call_ids: tool call ids to be added to the history
            **kwargs: keyword arguments to be passed to the templates (in addition to the
                ones in `self._get_format_dict`)
        """
        messages = []

        format_dict = self._get_format_dict(**kwargs)
        for template in templates:
            try:
                messages.append(Template(template).render(**format_dict))
            except KeyError:
                self.logger.debug(
                    "The following keys are available: %s", format_dict.keys()
                )
                raise

        message = "\n".join(messages)

        # We disable syntax highlighting here, because some inputs can lead to a complete cross-thread
        # freeze in the agent. See https://github.com/SWE-agent/SWE-agent/issues/901 .
        self.logger.info(f"🤖 {role.name.upper()} MODEL INPUT\n{message}",
                         extra={"highlighter": None})
        history_item: dict[str, Any] = {
            "role": "user",
            "content": message,
            "agent": self.name,
            "message_type": "observation",
        }
        if tool_call_ids:
            assert (len(tool_call_ids) == 1), "This should be ensured by the FunctionCalling parse method"
            history_item["role"] = "tool"
            history_item["tool_call_ids"] = tool_call_ids
        self._append_role_history(history_item, role)

    def add_step_to_role_history(self, step: StepOutput, role: Role) -> None:
        """Adds a step (command that was run and output) to the model history"""
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
            role
        )

        elided_chars = 0

        if step.observation.strip() == "":
            # Show no output template if observation content was empty
            templates = [role.next_step_no_output_template]
        # NOTE does this still work with how we defined self and roles?
        elif len(step.observation) > role.max_observation_length:
            # NOTE does this still work with how we defined self and roles?
            templates = [role.next_step_truncated_observation_template]
            # NOTE does this still work with how we defined self and roles?
            elided_chars = len(step.observation) - role.max_observation_length
        else:
            # Show standard output template if there is observation content
            templates = [role.next_step_template]
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
        return self.current_role.history

    def advance_current_role(self) -> None:
        # 1. Find the current index based on the name
        current_idx = self.role_order.index(self.current_role.name)
        # 2. Use modulo to wrap around automatically
        next_idx = (current_idx + 1) % len(self.role_order)
        # 3. Update to the next Role object
        self.current_role = self.roles[self.role_order[next_idx]]

    def run_role_forward(self, role: Role) -> StepOutput:
        # Use communicate instead of execute. 
        # Also check if self._env is not None to satisfy the Pylance warning.
        if self._env is not None:
            res = self._env.communicate(input="cat hello.py")
            print(f"DEBUG: File Content of hello.py:\n{res}")
        else:
            print("DEBUG: Environment is None!")
        
        self.history = role.history
        try:

            self.messages

            # pp = pprint.PrettyPrinter(indent=2)
            # print("--- FULL SELF OBJECT STATE ---")
            # pp.pprint(vars(self))
            # print("------------------------------")
            
            # print(f"\n{'='*20} AGENT STATE {'='*20}")
            # for attr, value in self.__dict__.items():
            #     # We skip history to keep the output focused, as you already know it
            #     if attr == "history":
            #         print(f"{attr}: <list of {len(value)} items>")
            #         continue
            #     print(f"{attr}: {repr(value)[:10]}...") # truncate very long strings
            # print(f"{'='*53}\n")
            
            print(f"\n{'='*20} AGENT STATE {'='*20}")
            print(f"DEBUG: Formatted messages for {role.name}:")
            # pprint.pprint(self.messages)
            print(f"{'='*53}\n")

            step_output = self.forward_with_handling(self.messages)

            # For planner: use raw model output as thought
            if role.name == "planner" and not step_output.thought:
                step_output.thought = step_output.output

            return step_output
        finally:
            pass

    def step(self) -> StepOutput:
        """Run a step of the agent. This is a wrapper around `self.forward_with_handling`
        with additional bookkeeping:

        1. Update message history with performed action and observation
        2. Update trajectory with the final executed result
        3. Update the info dictionary

        Returns:
            step_output: step output (same as the output of `self.forward_with_handling`)
        """

        assert self._env is not None
        self._chook.on_step_start()

        n_step = len(self.trajectory) + 1
        self.logger.info("=" * 25 + f" STEP {n_step} " + "=" * 25)

        # CUSTOM modified these 2 functions for managing current role
        step_output = self.run_role_forward(self.current_role)
        # step_output.info["role_name"] = self.current_role.name
        self.add_step_to_role_history(step_output, self.current_role)

        self.info["submission"] = step_output.submission
        self.info["exit_status"] = step_output.exit_status  # type: ignore
        self.info.update(self._get_edited_files_with_context(patch=step_output.submission or ""))  # type: ignore
        self.info["model_stats"] = self.model.stats.model_dump()
        self.add_step_to_trajectory(step_output)

        # change current_role to next in order for the next step call (only if you have more than one)
        if len(self.role_order) > 1:
            self.advance_current_role()

        self._chook.on_step_done(step=step_output, info=self.info)
        return step_output

        """run is not modified as it handles a significant amount of bookeeping and interaction with SWE-Bench.
        Modifying the step logic is enough"""
