# ASH - The Answer Set Programming based Hierarchical Conformance Refinement Planner for Robots

ASH is an online autonomous task and high-level action planning system for complex discrete deterministic planning problems, built in Answer Set Programming (ASP).

ASH uses a novel divide-and-conquer based approach to online hierarchical planning, which enables it to generate and incrementally yield partial plans throughout plan execution.
This ability can reduces execution latency and total planning times exponentially over the existing state-of-the-art ASP based planners, and for first time places ASP as a practical tool for real-world/time robotics problems.

## HCR Planning at a glance

The concept of HCR planning is intuitive.
Plans are generated and progressively refined downwards over an abstraction hierarchy, under a constraint that requires those plans remain structurally similar and achieve the same effects at all levels.

This constraint is formed by a series of sub-goal stages, obtained from the effects of abstract actions planned at the high-levels, which serve to form a skeleton for solutions at the lower-levels.
At any refinement level, the existance of this skeleton structure allows a complete refinement problem to be divided into a sequence of exponentially simpler partial refinement problems, by any of a variety of problem division strategies.

This simple mechanism allows blindingly fast plan generation whilst requiring only the addition of the abstraction hierarchy to the robot's knowledge base.
The conjecture is that obtaining this hierarchy is simple because it requires only a removal of descriptive knowledge.

## ASP based domain and problem encodings

The primary desirable characteristic of ASP is its intuitive and highly elaboration tolerent language for knowledge representation and reasoning.
It provides the ability to represent a dynamic system through a set of intuitve axiomatic rules with define the fundamental physical laws of a that system.
This give a robot an understanding of the constraints that govern its reality and enable it to reason for itself about how to formulate plans.
For example, the following linguistic rules can be trivially translated to the language of ASP:
* __Action Effect__ - "When a robot moves, its location changes"
* __Action Precondition__ - "A robot can only grasp objects that share its location"
* __State Variable Relation__ - "Grasped objects continue to share a robot's location as it moves"

Constructing an abstraction hierarchy for HCR planning requires defining a series of abstract domain models.
An abstract domain model may; remove, generalise, or redefine any of these system laws, in order to obtain a simplified description of the domain and problem.
The intuition is that, a plan generated in an abstract model should be significantly easier to solve than the original model, and the abstract plan should give the robot enough of an understanding of what the structure of the original level plan might look like to guide its search for it.
When the time saved by this is more than the time taken to obtain the abstract solution, then we have benefited from such a mechanism.

There are three such abstract models currently supported by the theory and implementation.
* __Condensed Models__ - The state space is reduced, by automatically combining sets of detailed entities into abstract descriptors, this reduces the number of actions and state variables needed to represent the problem, and generalises planning constraints. Abstraction mappings are generated automatically.
* __Relaxed Models__ - A sub-set of action preconditions are removed, this removes significant constraints on planning. Abstraction mappings are generated automatically.
* __Tasking Models__ - The system laws are redefined to create a system representation that deals with abstract task descriptions, the resulting plan is a sequence of tasks to be completed in order. Abstraction mappings must be given manually by the designer, and tell the planner how it can complete tasks by reaching states of the original model.
