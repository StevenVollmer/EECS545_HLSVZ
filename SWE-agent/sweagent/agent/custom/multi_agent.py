from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel
from typing_extensions import Self

from sweagent import __version__
from sweagent.agent.action_sampler import AbstractActionSampler, ActionSamplerConfig
from sweagent.agent.history_processors import HistoryProcessor
from sweagent.agent.hooks.abstract import CombinedAgentHook
from sweagent.agent.models import (
    AbstractModel,
    HumanModel,
    HumanThoughtModel,
    ModelConfig,
    get_model,
)
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.tools.parsing import (
    ActionOnlyParser,
    ThoughtActionParser,
)
from sweagent.tools.tools import ToolHandler
from sweagent.types import AgentInfo, AgentRunResult, StepOutput, Trajectory
from sweagent.utils.log import get_logger

from sweagent.agent.agents import AbstractAgent, TemplateConfig, MultiAgentConfig

class MultiAgent(AbstractAgent):
    """CUSTOM class created for experimental agent architecture. We will implement an agent 
    that includes a planner, coder, and reviewer. Each will have its own history and may require
    its own parser to read its output and convert to an appropriate file format. This was adapted
    from ShellAgent in extra/shell_agent.py
    
    Example useage:
    sweagent run \
        --config config/custom_configs/multi_agent.yaml \
        --env.repo.github_url=https://github.com/SWE-agent/test-repo \
        --problem_statement.text="Add a simple hello world function to a new file named hello.py."
    
    
    """
    
    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler,
        history_processors: list[HistoryProcessor],
        model: AbstractModel,
        max_requeries: int = 3,
        name: str = "main",
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
    ):
        """The agent handles the behaviour of the model and how it interacts with the environment.

        To run the agent, either call `self.run` or `self.setup` and then `self.step` in a loop.
        """
        self._catch_errors = _catch_errors
        self._always_require_zero_exit_code = _always_require_zero_exit_code
        self.name = name
        self.model = model
        self.templates = templates
        self.tools = tools
    #TODO will need to modify parser for each sub agent
        if isinstance(self.model, HumanThoughtModel):
            self.tools.config.parse_function = ThoughtActionParser()
        elif isinstance(self.model, HumanModel):
            self.tools.config.parse_function = ActionOnlyParser()
        self.history_processors = history_processors
        self.max_requeries = max_requeries
        self.logger = get_logger("swea-agent", emoji="📅")
        # Set in run method
        self._env: SWEEnv | None = None
        self._problem_statement: ProblemStatement | ProblemStatementConfig | None = None
        self.traj_path: Path | None = None

        #: The following three attributes collect the information about how the agent
        #: solved the problem.
        self.history = []
        self._trajectory = []
        self.info = AgentInfo()

        self._chook = CombinedAgentHook()

        self._replay_config: BaseModel | None = None
        """This can be set to a RunSingleConfig from the Run instance whenever possible.
        It can be used to replay the agent's trajectory in an environment.
        """

        self._action_sampler: AbstractActionSampler | None = None
        if action_sampler_config is not None:
            self._action_sampler = action_sampler_config.get(self.model, self.tools)

        #: Count how many timeout errors have occurred consecutively. Kills agent
        #: after 5 of them.
        self._n_consecutive_timeouts = 0
        self._total_execution_time = 0.0


    @classmethod
    def from_config(cls, config: MultiAgentConfig) -> Self:
        # To ensure that all models stay completely independent, we deepcopy the
        # model config, because it lives on as a property in the model, tools, etc.
        config = config.model_copy(deep=True)
        model = get_model(config.model, config.tools)
        return cls(
            templates=config.templates,
            tools=ToolHandler(config.tools),
            history_processors=config.history_processors,
            model=model,
            max_requeries=config.max_requeries,
            action_sampler_config=config.action_sampler,
        )
    
    def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        """Setup the agent for a new instance. This includes
        formatting the system message and adding demonstrations to the history.

        This method is called by `self.run`.
        """
        
        # output_dir.mkdir(parents=True, exist_ok=True)

        # # apply template configuration to multimodal problem statements
        # if hasattr(problem_statement, "type") and problem_statement.type == "swe_bench_multimodal":
        #     from sweagent.agent.problem_statement import SWEBenchMultimodalProblemStatement

        #     if isinstance(problem_statement, SWEBenchMultimodalProblemStatement):
        #         # apply the global disable_image_processing setting if it's not explicitly set
        #         if not problem_statement.disable_image_processing and self.templates.disable_image_processing:
        #             problem_statement.disable_image_processing = True

        # self._problem_statement = problem_statement
        self._env = env
        # iid = self._problem_statement.id
        # self.logger.info("Setting up agent for instance %s", iid)

        # # Save/reset some attributes
        # self.traj_path = output_dir / (self._problem_statement.id + ".traj")
        # self.logger.info("Trajectory will be saved to %s", self.traj_path)

        # self._chook.on_tools_installation_started()
        # self.tools.install(self._env)
        # self._chook.on_setup_attempt()
        # self.info = AgentInfo()
        # self.info["swe_agent_hash"] = get_agent_commit_hash()
        # self.info["swe_agent_version"] = __version__
        # self.info["swe_rex_version"] = get_rex_version()
        # self.info["swe_rex_hash"] = get_rex_commit_hash()
        # assert self._env is not None
        # assert self._problem_statement is not None
        # self._env.set_env_variables({"PROBLEM_STATEMENT": self._problem_statement.get_problem_statement_for_env()})
        # self.add_system_message_to_history()
        # self.add_demonstrations_to_history()
        # self.add_instance_template_to_history(state=self.tools.get_state(self._env))
        # self._chook.on_setup_done()
        
        pass
    
    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def save_trajectory(
        self,
    ) -> None:
        """Save the trajectory to disk.
        This includes the history, the environment state, and the model stats.
        """
        data = self.get_trajectory_data()
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(data, indent=2))

    def get_trajectory_data(self) -> dict[str, Any]:
        """Get all data that we save in .traj files."""

        assert self._env is not None
        # The deepcopy here is important because else the
        # data["info"]["model_stats"] update will create havoc!
        attempt_data = copy.deepcopy(
            {
                "trajectory": self.trajectory,
                "history": self.history,
                "info": self.info,
            }
        )
        attempt_data["replay_config"] = self.replay_config.model_dump_json() if self.replay_config is not None else None # type: ignore
        attempt_data["environment"] = self._env.name # type: ignore
        return attempt_data

    def step(self) -> StepOutput:
        """Run a step of the agent. This is a wrapper around `self.forward_with_handling`
        with additional bookkeeping:

        1. Update message history with performed action and observation
        2. Update trajectory with the final executed result
        3. Update the info dictionary

        Returns:
            step_output: step output (same as the output of `self.forward_with_handling`)
        """
        return StepOutput()

        # assert self._env is not None
        # self._chook.on_step_start()

        # n_step = len(self.trajectory) + 1
        # self.logger.info("=" * 25 + f" STEP {n_step} " + "=" * 25)
        # step_output = self.forward_with_handling(self.messages)
        # self.add_step_to_history(step_output)

        # self.info["submission"] = step_output.submission
        # self.info["exit_status"] = step_output.exit_status  # type: ignore
        # self.info.update(self._get_edited_files_with_context(patch=step_output.submission or ""))  # type: ignore
        # self.info["model_stats"] = self.model.stats.model_dump()

        # self.add_step_to_trajectory(step_output)

        # self._chook.on_step_done(step=step_output, info=self.info)
        # return step_output

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the agent on a problem instance. This method contains the
        main loop that repeatedly calls `self._step` until the problem is solved.
        
        Removed hooks for now.

        Args:
            setup_args: Arguments to pass to the agent's setup method.
            env: The environment to run the agent on.
            traj_dir: Directory to save the trajectory to
        """
        
        # This is where we will make the main modifications.
        # query planner model and parse out put into plan
        # for eash task in plan
        #   query coder model
        # after coder loop is finished, query reviewer model
        
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)

        # # Run action/observation loop
        # step_output = StepOutput()

        # while not step_output.done:
        #     step_output = self.step()
        #     self.save_trajectory()

        # self.logger.info("Trajectory saved to %s", self.traj_path)

        data = self.get_trajectory_data()
        # this checks that the data is correctly formatted so it can be used in evaluation later
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])
