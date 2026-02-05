# EECS545_HLSVZ
Team Project for EECS 545/ Machine Learning

This is a work in progress.

We modify SWE-agent (https://github.com/SWE-agent/SWE-agent) to include a new class, MultiAgent.
MultiAgent accomodates a new multi-role structure with separate historries. The intended roles 
are a planner, a coder, and a reviewer.

Planner:
The planner reads the initial problem statement and analyzes the repo. It then produces a step
by step plan for the coder to follow.

Coder:
Reads the plan and executes it step by step.

Reviewer:
The reviewer analyzes the coder's patch. It develops tests and validates the patch. If the patch
is successful, then it calls submit. If the coder never reached a conclusion, the reviewer 
determines if the plan was impossible to execute, which it then calls the planner to revise its 
plan. Or if the plan is valid, but the coder failed, the reviewer recalls the coder with an
explanation of the failure and suggests corrective actions.
