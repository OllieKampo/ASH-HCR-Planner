###########################################################################
###########################################################################
## Python script for generating plans with ASH                           ##
## Copyright (C)  2021  Oliver Michael Kamperis                          ##
## Email: o.m.kamperis@gmail.com                                         ##
##                                                                       ##
## This program is free software: you can redistribute it and/or modify  ##
## it under the terms of the GNU General Public License as published by  ##
## the Free Software Foundation, either version 3 of the License, or     ##
## any later version.                                                    ##
##                                                                       ##
## This program is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          ##
## GNU General Public License for more details.                          ##
##                                                                       ##
## You should have received a copy of the GNU General Public License     ##
## along with this program. If not, see <https://www.gnu.org/licenses/>. ##
###########################################################################
###########################################################################

from dataclasses import dataclass, field, fields
from functools import cached_property
import numpy
from tqdm import tqdm
from core.Strategies import DivisionPoint, DivisionPointPair, DivisionScenario, DivisionStrategy, Reaction, SubGoalRange
import enum
import logging
from typing import Any, Callable, Final, Iterable, Iterator, Literal, NamedTuple, Optional, Union, final
import statistics
import json
import _collections_abc

import ASP_Parser as ASP
from core.Helpers import AbstractionHierarchy, ReversableDict, center_text
import clingo
import os
import sys
import statistics
from scipy.optimize import curve_fit
import time

## Module logger
_planner_logger: logging.Logger = logging.getLogger(__name__)
_planner_logger.setLevel(logging.DEBUG)

## Type aliases
Number = Union[int, float]
HierarchicalNumber = Union[Number, dict[int, Number]]

## Custom exception handling
class ASH_Error(Exception): pass
class ASH_NoSolutionError(ASH_Error): pass
class ASH_InvalidPlannerState(ASH_Error): pass
class ASH_InternalError(ASH_Error): pass

class ASH_InvalidInputError(ASH_Error):
    def __init__(self, message: str, given: Optional[Any] = None) -> None:
        _message: str = message
        if given is not None:
            _message += f"Got; {given} of type {type(given)}."
        super().__init__(_message)

def log_and_raise(exception_type: type[BaseException],
                  *args: object, from_exception: Optional[BaseException] = None,
                  logger: logging.Logger = _planner_logger) -> None:
    """Commit a log (at debug level) and then raise an exception from a sequence of arbitrary arguments."""
    logger.debug(*args, exc_info=(from_exception is not None))
    raise exception_type(*args) from from_exception

#############################################################################################################################################
#############################################################################################################################################
###############################   █████  ███████ ██   ██               ██████  ██       █████  ███    ██ ███████  ###########################
###############################  ██   ██ ██      ██   ██               ██   ██ ██      ██   ██ ████   ██ ██       ###########################
###############################  ███████ ███████ ███████     █████     ██████  ██      ███████ ██ ██  ██ ███████  ###########################
###############################  ██   ██      ██ ██   ██               ██      ██      ██   ██ ██  ██ ██      ██  ###########################
###############################  ██   ██ ███████ ██   ██               ██      ███████ ██   ██ ██   ████ ███████  ###########################
#############################################################################################################################################
#############################################################################################################################################

@enum.unique
class ActionType(enum.Enum):
    """
    Enum class encapulating possible action types.
    
    Items
    -----
    `Locomotion = "locomotion"` - A locomotion action is one that changes the robot's location,
    and possibly the location of entities currently being grasped or transported by the robot.
    
    `Manipulation = "manipulation"` - A manipulation action is one that allows the robot to move,
    place, or grasp an entity.
    
    `Configuration = "configuration"` - A configuration action is one that changes the robot's own
    physical state, or the physical state of its component parts, without changing the wider system state.
    """
    
    Locomotion = "locomotion"
    Manipulation = "manipulation"
    Configuration = "configuration"

class Fluent(ASP.Atom):
    """
    Represents a fluent literal.
    
    Fluent literals are fluent function literals defining the current value of a state variable at a given time step.
    
    Fields
    ------
    `L : int` - An integer, greater than zero, defining the abstraction level of the fluent.
    
    `F : str` - A non-empty string defining the fluent symbol itself, usually a function symbol of the form `name(arg_1, arg_2, ... arg_n)`.
    
    `V : str` - A non-empty string defining the value assigned to the fluent.
    
    `S : int` - An integer, greater than or equal to zero, defining the discrete time step the fluent holds the given value at.
    """
    @classmethod
    def default_params(cls) -> tuple[str]:
        return ('L', 'F', 'V', 'S')
    
    @classmethod
    def default_cast(cls) -> Optional[dict[str, Callable[[clingo.Symbol], Any]]]:
        return {'L' : [str, int], 'F' : str, 'V' : str, 'S' : [str, int]}
    
    @classmethod
    def default_sort(cls) -> Optional[Union[str, tuple[str]]]:
        return ('L', 'S', 'F', 'V')
    
    @classmethod
    def predicate_name(cls) -> str:
        return "holds"

StateVariableMapping = dict[tuple[int, str], list[Fluent]]

class Action(ASP.Atom):
    """
    Represents an action literal.
    Whose encoding as an ASP symbol (an atom) of the form:
            occurs(al, robot, action, step)
    
    Actions are function symbols declaring operators usable by robots.
    Action literals are function symbols defining that robot has planned an action at a given abstraction level and time step.
    
    Fields
    ------
    `L : int` - An integer, greater than zero, defining the abstraction level of the action literal.
    
    `R : str` - A non-empty string defining the name of the executing robot of the action literal.
    
    `A : str` - A non-empty string defining the action itself, usually a function symbol of the form `name(arg_1, arg_2, ... arg_n)`.
    
    `S : int` - An integer, greater than zero, defining the discrete time step the action is planned to occur at.
    """
    @classmethod
    def default_params(cls) -> Optional[tuple[str]]:
        return ('L', 'R', 'A', 'S')
    
    @classmethod
    def default_cast(cls) -> Optional[dict[str, Callable[[clingo.Symbol], Any]]]:
        return {'L' : [str, int], 'R' : str, 'A' : str, 'S' : [str, int]}
    
    @classmethod
    def default_sort(cls) -> Optional[Union[str, tuple[str]]]:
        return ('L', 'S', 'R', 'A')
    
    @classmethod
    def predicate_name(cls) -> str:
        return "occurs"
    
    @property
    def type(self) -> "ActionType":
        "The action's type; locomotion, configuration, or manipulation."
        if str(self['A']).startswith("move"):
            return ActionType.Locomotion
        elif str(self['A']).startswith("configure"):
            return ActionType.Configuration
        else: return ActionType.Manipulation

class SubGoal(ASP.Atom):
    """
    Represents an action literal.
    Whose encoding as an ASP symbol (an atom) of the form:
            occurs(al, robot, action, step)
    
    Actions are function symbols declaring operators usable by robots.
    Action literals are function symbols defining that robot has planned an action at a given abstraction level and time step.
    
    Fields
    ------
    `L : int` - An integer, greater than zero, defining the abstraction level of the action literal.
    
    `R : str` - A non-empty string defining the name of the executing robot of the action literal.
    
    `A : str` - A non-empty string defining the action which created the sub-goal literal, usually a function symbol of the form `name(arg_1, arg_2, ... arg_n)`.
    
    `F : str` - A non-empty string defining the fluent symbol affected by the given action, usually a function symbol of the form `name(arg_1, arg_2, ... arg_n)`.
    
    `V : str` - A non-empty string defining the value assigned to the fluent by the effect of the given action.
    
    `I : int` - An integer, greater than zero, defining the discrete time step the action is planned to occur at.
    """
    @classmethod
    def default_params(cls) -> Optional[tuple[str]]:
        return ('L', 'R', 'A', 'F', 'V', 'I')
    
    @classmethod
    def default_cast(cls) -> Optional[dict[str, Callable[[clingo.Symbol], Any]]]:
        return {'L' : [str, int], 'R' : str, 'A' : str, 'F' : str, 'V' : str, 'I' : [str, int]}
    
    @classmethod
    def default_sort(cls) -> Optional[Union[str, tuple[str]]]:
        return ('L', 'I', 'R', 'A', 'F', 'V')
    
    @classmethod
    def predicate_name(cls) -> str:
        return "sub_goal"

class FinalGoal(ASP.Atom):
    """
    Represents a final goal literal.
    
    Fields
    ------
    `L : int` - An integer, greater than zero, defining the abstraction level of the action literal.
    
    `F : str` - A non-empty string defining the goal-fluent symbol, usually a function symbol of the form `name(arg_1, arg_2, ... arg_n)`.
    
    `V : str` - A non-empty string defining the final-goal literal value of the goal-fluent.
    
    `T : bool` - The truth of the goal, either; true or false, if true then the value must hold in the final-goal state, if false it must not.
    """
    @classmethod
    def default_params(cls) -> Optional[tuple[str]]:
        return ('L', 'F', 'V', 'T')
    
    @classmethod
    def default_cast(cls) -> Optional[dict[str, Callable[[clingo.Symbol], Any]]]:
        return {'L' : [str, int], 'F' : str, 'V' : str, 'T' : str}
    
    @classmethod
    def default_sort(cls) -> Optional[Union[str, tuple[str]]]:
        return ('L', 'F', 'V', 'T')
    
    @classmethod
    def predicate_name(cls) -> str:
        return "final_goal"

@dataclass(frozen=True)
class ASH_Statistics(ASP.Statistics):
    """
    Extension of ASP statistic to contain overhead time (the time spent reasoning
    about reaction problem divisions in sequential yield planning mode).
    """
    overhead_time: float = 0.0

@dataclass(frozen=True)
class ASH_IncrementalStatistics(ASP.IncrementalStatistics):
    overhead_time: float = 0.0
    
    @property
    def grand_totals(self) -> ASH_Statistics:
        """The grand total times and maximum memory usage, the sum of the base and all incremental times."""
        statistics: ASP.Statistics = super().grand_totals
        return ASH_Statistics(grounding_time=statistics.grounding_time,
                              solving_time=statistics.solving_time,
                              total_time=statistics.total_time + self.overhead_time,
                              memory=statistics.memory,
                              overhead_time=self.overhead_time)
    
    @classmethod
    def from_incremental_statistic(cls, incremental_statistic: ASP.IncrementalStatistics, overhead_time: float) -> "ASH_IncrementalStatistics":
        return cls(*(getattr(incremental_statistic, field.name) for field in fields(incremental_statistic)), overhead_time)



@dataclass(frozen=True)
class ConformanceMapping:
    """
    Represents a conformance mapping between a contiguous sequence of sub-goal stages
    produced from an abstract monolevel plan and an original level monolevel plan.
    
    A conformance mapping represents how adjacent levels of a refinement diagram are linked
    by defining the descending refined plan's child sub-plans and the matching child unique
    achievement steps for each sub-goal stage.
    
    The mapping also contains the greedy minimal unique sequential yield achievement steps
    of each sub-goal stage iff the plan was generated in sequential yield mode.
    
    Fields
    ------
    `constraining_sgoals : dict[int, list[SubGoal]]` - A list of sub-goal stages forming the constraint.
    
    `current_sgoals : ReversableDict[int, int]` - The current sub-goal stage at each plan step mapping,
    the reverse of this mapping is the sub-goal stage to refining sub-plan mapping.
    
    `sgoals_achieved_at : ReversableDict[int, int]` - The sub-goal stage achievement step mapping.
    
    `sequential_yield_steps : {dict[int, int] | None} = None` - The steps at which actions were
    yielded in sequential yield planning mode (used for calculating interleaving quantities).
    """
    
    ## The conformance mapping itself;
    ##      - The conformance constraint,
    ##      - The descending child sub-plans,
    ##      - The matching children (minimal unqiue achievement steps).
    constraining_sgoals: dict[int, list[SubGoal]]
    current_sgoals: ReversableDict[int, int]
    sgoals_achieved_at: ReversableDict[int, int]
    
    ## Variable for sequential yield planning
    sequential_yield_steps: Optional[dict[int, int]] = None
    
    @classmethod
    def from_answer(cls, constraining_sgoals: dict[int, list[SubGoal]], answer: ASP.Answer, sequential_yield_steps: Optional[dict[int, int]] = None) -> "ConformanceMapping":
        """
        Construct a conformance mapping for a given sequence of sub-goal stages from an answer set.
        """
        query: dict[int, list[dict[str, int]]]
        
        ## Find the steps upon which each sub-goal stage was current and was achieved (i.e. construct the refinement diagram)
        query = answer.fmodel.query("current_sub_goal_index", ['L', 'I', 'S'], True,
                                    sort_by='S', group_by='S', cast_to=[str, int])
        current_sgoals: ReversableDict[int, int] = ReversableDict({step : query[step][0]['I'] for step in query})
        
        query = answer.fmodel.query("sgoals_ach_at", ['L', 'I', 'S'], True,
                                    sort_by='I', group_by='I', cast_to=[str, int])
        
        sgoals_achieved_at: ReversableDict[int, int] = ReversableDict({index : query[index][0]['S'] for index in query})
        
        ## Construct the conformance mapping
        return ConformanceMapping(constraining_sgoals, current_sgoals, sgoals_achieved_at, sequential_yield_steps)
    
    def __str__(self) -> str:
        return f"Number of sub-goal stages = {len(self.constraining_sgoals)}, Index range = [{min(self.constraining_sgoals)}-{max(self.constraining_sgoals)}], Step range = [{min(self.current_sgoals)}-{max(self.current_sgoals)}]"
    
    @property
    def problem_size(self) -> int:
        "The size of the conformance refinement planning problem (number of sub-goal stages) represented by this conformance mapping."
        return len(self.constraining_sgoals)
    
    @property
    def constraining_sgoals_range(self) -> SubGoalRange:
        "The sub-goal stage indices range of the conformance refinement planning problem represented by this conformance mapping."
        return SubGoalRange(min(self.constraining_sgoals), max(self.constraining_sgoals))
    
    @property
    def total_sgoal_producing_actions(self) -> int:
        "The number of sub-goal producing actions from the abstract plan that this conformance mapping refines."
        return len(set((sgoal["I"], sgoal["R"], sgoal["A"]) for sgoals in self.constraining_sgoals.values() for sgoal in sgoals))
    
    @property
    def total_sgoal_literals(self) -> int:
        "The total sum of all individual constraining sub-goal literals over all stages of the problem."
        return sum(len(sgoals) for sgoals in self.constraining_sgoals.values())
    
    @property
    def length_expansion_factor(self) -> float:
        "The factor by which the refined plan expands in length with respect to the given refinement problem size (equivalent to the sub-goal stage producing abstract plan length)."
        return len(self.current_sgoals) / len(self.constraining_sgoals)
    
    @property
    def length_expansion_deviation(self) -> float:
        """
        The standard deviation of the sub-plan length expansion factors over all conformance constraining sub-goal stages refined by this plan.
        
        This measures the variation or spread, of the matching child sub-goal stage achieving state transitions.
        A value of zero indicates perfectly balanced refinement trees, such that the refining child sub-plan lengths are exactly equal for all refinement trees.
        """
        if len(self.sgoals_achieved_at) > 1:
            return statistics.stdev(self.get_subplan_length(index) for index in self.sgoals_achieved_at)
        return 0.0
    
    def get_subplan_length(self, index: int) -> int:
        """
        Get the sub-plan length expansion factor of a given conformance constraining sub-goal stage.
        
        Parameters
        ----------
        `index : int` - The goal sequence index of the sub-goal stage.
        
        Returns
        -------
        `int` - The sub-plan length expansion factor, an integer in the range [1-infinity].
        
        Raises
        ------
        `ValueError` - If the given sub-goal stage index is not in the range of the conformance refinement planning problem represented by this mapping.
        """
        if index not in self.sgoals_achieved_at:
            raise ValueError(f"The sub-goal index {index} is not in the range of conformance mapping {self!s}.")
        return len(self.current_sgoals(index))

class Expansion(NamedTuple):
    """
    Tuple class encapsulating expansion factors, deviations, or balance.
    
    Fields
    ------
    `length : float` - The plan length expansion.
    
    `action : float` - The action quantity expansion.
    """
    length: float
    action: float
    
    def __str__(self) -> str:
        return f"(LE = {self.length:>3.1f}, AC = {self.action:>3.1f})"

@dataclass(frozen=True)
class MonolevelPlan(_collections_abc.Mapping):
    """
    A monolevel plan is a mapping between a contiguous sequence of time steps and corresponding actions sets planned to occur on those steps.
    
    Fields
    ------
    `level: int` - The abstraction level of the monolevel plan.
    
    `states: dict[int, list[Fluent]]` -
    
    `actions: dict[int, list[Action]]` -
    
    `produced_sgoals: dict[int, list[SubGoal]]` -
    
    `is_final: bool` -
    
    `planning_statistics: ASH_IncrementalStatistics` -
    
    `conformance_mapping: {ConformanceMapping | None} = None` -
    
    `problem_divisions: {list[DivisionPoint] | None} = None` -
    
    `total_choices: int = 0` -
    This is None iff the plan is concatenated.
    
    `preemptive_choices: int = 0` -
    This is None iff the plan is concatenated.
    
    `fgoal_ordering_correct: {bool | None} = None` - Whether the final-goal intermediate order preferences were achieved in the correct order.
    This is None iff the plan is not top-level (i.e. not the most abstract classical plan)
    """
    ## Universal plan variables
    level: int
    model_type: "DomainModelType"
    states: dict[int, list[Fluent]]
    actions: dict[int, list[Action]]
    produced_sgoals: dict[int, list[SubGoal]]
    is_final: bool
    planning_statistics: ASH_IncrementalStatistics
    
    ## Conformance refinement plan variables
    conformance_mapping: Optional[ConformanceMapping] = None
    problem_divisions: Optional[list[DivisionPoint]] = None
    
    ## Final-goal preemptive achievement
    total_choices: int = 0
    preemptive_choices: int = 0
    
    ## Final-goal intermediate achievement ordering
    fgoals_achieved_at: list[int] = field(default_factory=list)
    fgoal_ordering_correct: Optional[bool] = None
    
    def __str__(self) -> str:
        gt_time: ASH_Statistics = self.planning_statistics.grand_totals
        return (f"Lvl = {self.level:>1d} ({self.problem_type}), LE = {self.plan_length:>3d}, AC = {self.total_actions:>3d}, CF = {self.compression_factor:>3.1f}, " # Plan quality
                f"GT = {gt_time.grounding_time:>6.2f}s, ST = {gt_time.solving_time:>6.2f}s, OT = {gt_time.overhead_time:>4.2f}s, TT = {gt_time.total_time:>6.2f}s, " # Planning times
                f"EF = {self.get_plan_expansion_factor()!s}, ED = {self.get_expansion_deviation()!s}, EB = {self.get_degree_of_balance()!s}") # Refinement tree balancing
    
    ##########################
    ## Mapping methods: Time Step -> Action Set
    
    def __getitem__(self, step: int) -> list[Action]:
        return self.actions[step]
    
    def __contains__(self, step: int) -> bool:
        return step in self.actions
    
    def __iter__(self) -> Iterator[int]:
        yield from self.actions
    
    def __len__(self) -> int:
        return len(self.actions)
    
    @property
    def action_step_range(self) -> range:
        return range(self.action_start_step, self.end_step + 1)
    
    @property
    def state_step_range(self) -> range:
        return range(self.state_start_step, self.end_step + 1)
    
    @property
    def action_start_step(self) -> int:
        "The plan's start step, inclusive of states but exclusive of actions."
        return min(self.actions)
    
    @property
    def state_start_step(self) -> int:
        "The plan's start step, inclusive of states but exclusive of actions."
        return min(self.states)
    
    @property
    def end_step(self) -> int:
        "The plan's end step, inclusive of both state and actions."
        return max(self.states)
    
    ##########################
    ## Plan quality measures
    
    @property
    def plan_length(self) -> int:
        "The length of the plan."
        return len(self)
    
    @property
    def total_actions(self) -> int:
        "The total number of actions in the plan."
        return sum(len(self[step]) for step in self)
    
    @property
    def compression_factor(self) -> float:
        "The compression factor of the plan, representing the percentage reduction in the plan length achieved by concurrent action planning."
        return self.plan_length / self.total_actions
    
    @property
    def total_produced_sgoals(self) -> int:
        "The total number of sub-goals produced by this abstract plan."
        return sum(len(sgoals) for sgoals in self.produced_sgoals.values())
    
    def calculate_plan_quality(self, optimal_length: int, optimal_actions: int) -> float:
        "Calculate the quality of this plan relative to the plan length and action quantity of another."
        return statistics.mean(optimal_length / self.plan_length, optimal_actions / self.total_actions)
    
    ####################
    ## Plan properties
    
    @property
    def is_abstract(self) -> bool:
        "Whether this plan was generated in an abstract model. An abstract plan cannot be executed and produces sub-goal stages to allow it to be refined in the original model."
        return self.level > 1
    
    @property
    def is_ground(self) -> bool:
        "Whether this plan was generated in a ground model. A ground plan can be executed and cannot be refined."
        return self.level == 1
    
    @property
    def is_refined(self) -> bool:
        "Whether this is a refinement of an abstract plan, and thus has a conformance mapping to that abstract plan's sub-goal stages."
        return self.conformance_mapping is not None
    
    @property
    def is_initial(self) -> bool:
        "Whether the plan starts in the initial state."
        return self.state_start_step == 0
    
    @property
    def is_complete(self) -> bool:
        "Whether the plan completes its abstraction level, such that it is both initial and final."
        return self.is_initial and self.is_final
    
    @property
    def has_trailing_plan(self) -> bool:
        "Whether the plan is a refinement and has a trailing part (a sub-plan that does not map upwards to a sub-goal stage)."
        return self.is_refined and self.is_complete and self.end_step not in self.conformance_mapping.current_sgoals
    
    @property
    def planning_mode(self) -> Literal["offline", "online"]:
        "The planning mode used to generate this plan, 'offline' iff the plan is complete and contains no problem divisions (complete conformance refinement or classical), otherwise 'online'."
        return "offline" if self.is_complete and not self.problem_divisions else "online"
    
    @property
    def problem_type(self) -> Literal["classic", "com-ref", "par-ref"]:
        "The monolevel problem type this plan solves, 'classic' iff the problem was classical, 'com-ref' iff the problem was complete conformance refinement, otherwise 'par-ref' (partial conformance refinement)."
        return ("classic" if not self.is_refined
                else ("com-ref" if self.planning_mode == "offline"
                      else "par-ref"))
    
    ###################
    ## Planning costs
    
    @property
    def grand_totals(self) -> ASP.Statistics:
        "The grand total timing/memory statistics for generating this plan."
        return self.planning_statistics.grand_totals
    
    @staticmethod
    def __regress(steps: list[int], times: list[float]) -> tuple[Callable[[int, float, float], float], numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Fits a exponential function by non-linear least squares regression, to model planning times against search length.
        
        Parameters
        ----------
        `steps : list[int]` - A list of intergers defining the steps over which the plan to regress ranges.
        
        `times : list[float]` - A list of floating points defining the time expended (grounding, solving, or total planning) per step.
        
        Returns
        -------
        `tuple[Callable[[int, float, float], float], numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]` - A tuple whose items are;
            - A function of the form `a * e ^ (x * b)` where `x` is a time step (the independent variable), and `a` and `b` are the regression parameters,
            - An array conversion of the steps,
            - An array conversion of the times,
            - An array containing the optimal values for the parameters `a` and `b` such that the sum of squared residuals is minimal for `(a * e ^ (x * b)) - y` where `y` is the time expended for the given step `x`,
            - An array containing the estimated covariance of the optimal parameter values.
        
        Also See
        --------
        Documentation of regression method used: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        """
        x_points: numpy.ndarray = numpy.array(steps)
        y_points: numpy.ndarray = numpy.array(times)
        
        func = lambda x, a, b: a*numpy.exp(x*b)
        try:
            popt, pcov = curve_fit(func, x_points, y_points, [0.025, 0.025])
        except RuntimeError:
            popt, pcov = numpy.array([0.0, 0.0]), numpy.array([0.0, 0.0])
        
        return func, x_points, y_points, popt, pcov
    
    @cached_property
    def regress_grounding_time(self) -> tuple[Callable, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        "Fits a exponential function by regression, to model the grounding time against search length."
        steps: list[int] = [step for step in self]
        times: list[float] = [statistics.grounding_time for statistics in self.planning_statistics.incremental.values()]
        return self.__regress(steps, times)
    
    @cached_property
    def regress_solving_time(self) -> tuple[Callable, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        "Fits a exponential function by regression, to model the solving time against search length."
        steps: list[int] = [step for step in self]
        times: list[float] = [statistics.solving_time for statistics in self.planning_statistics.incremental.values()]
        return self.__regress(steps, times)
    
    @cached_property
    def regress_total_time(self) -> tuple[Callable[[int, float, float], float], numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        "Fits a exponential function by regression, to model the total planning time against search length."
        steps: list[int] = [step for step in self]
        times: list[float] = [statistics.total_time for statistics in self.planning_statistics.incremental.values()]
        return self.__regress(steps, times)
    
    @property
    def interleaving(self) -> tuple[tuple[int, float], tuple[int, float]]:
        """
        Calculate and return the interleaving quantities and score as a tuple of two-tuples;
            - number of interleaved sub-plans, percentage of interleaved sub-plans
            - number of interleaved plan steps, percentage of interleaved steps.
        """
        total_interleaved_sub_plans: int = 0
        total_interleaving_quantity: int = 0
        if (self.is_refined
            and self.conformance_mapping.sequential_yield_steps is not None):
            for index in self.conformance_mapping.constraining_sgoals:
                interleaving_quantity: int = self.interleaving_quantity(index)
                if interleaving_quantity:
                    total_interleaved_sub_plans += 1
                    total_interleaving_quantity += interleaving_quantity
        return ((total_interleaved_sub_plans, total_interleaved_sub_plans / self.conformance_mapping.problem_size),
                (total_interleaving_quantity, total_interleaving_quantity / self.plan_length))
    
    def interleaving_quantity(self, index: int) -> int:
        """
        Get the step interleaving quantity of the sub-plan that refines a given sub-goal stage index in the conformance constraint of this plan.
        The step interleaving quantity of a sub-plan is the difference in the minimal achievement and final achievement length of the sub-goal stage index.
        This shows the change in sub-plan length caused by interleaving.
        """
        if (self.is_refined
            and (yield_steps := self.conformance_mapping.sequential_yield_steps) is not None):
            sub_plan_length: int = len(self.conformance_mapping.current_sgoals(index))
            yield_length: int = (yield_steps[index] - yield_steps.get(index - 1, 0))
            if index > 1: yield_length = yield_length - self.interleaving_quantity(index - 1)
            return sub_plan_length - yield_length
        return 0
    
    @classmethod
    def __matches(cls, action: Action, action_type: ActionType) -> bool:
        "Simple class method for determining the types of actions."
        if action_type.value == ActionType.Locomotion:
            return (str(action['A']).startswith("move")
                    or str(action['A']).startswith("push"))
        elif action_type.value == ActionType.Configuration:
            return str(action['A']).startswith("configure")
        elif action_type.value == ActionType.Manipulation:
            return (not cls.__matches(action, ActionType.Locomotion)
                    and not cls.__matches(action, ActionType.Configuration))
    
    def get_action_type(self, step: int) -> ActionType:
        """Get the most common action type in the action set planned at the given time step."""
        return statistics.mode([action.type for action in self[step]])
    
    def get_sub_plan_type(self, index: int) -> ActionType:
        """The most common action type in the sub-plan that refines the sub-goal stage at the given index."""
        if not self.is_refined:
            raise RuntimeError("Cannot get sub-plan types of classical plan.")
        return statistics.mode([self.get_action_type(step) for step in self.conformance_mapping.current_sgoals(index)])
    
    def get_plan_expansion_factor(self, action_type: Optional[ActionType] = None) -> Expansion:
        """
        The plan expansion factor of this monolevel plan.
        Equivalent to the average expansion factor over all conformance constraining sub-goal stages refined by this plan.
        
        Parameters
        ----------
        `action_type: {ActionType | None} = None` - An optional action type.
        
        Returns
        -------
        `(float, float)` - A two tuple of floats defining;
            - The length expansion factor (index 0),
            - The action expansion factor (index 1).
        """
        if self.is_refined:
            if action_type is None:
                return Expansion(self.plan_length / self.conformance_mapping.problem_size,
                                 self.total_actions / self.conformance_mapping.total_sgoal_producing_actions)
            matching_steps: int = 0
            matching_actions: int = 0
            for step in self:
                counted: bool = False
                for action in self[step]:
                    if self.__matches(action, action_type):
                        if not counted:
                            matching_steps +=1
                            counted = True
                        matching_actions += 1
            return Expansion(matching_steps / self.conformance_mapping.problem_size,
                             matching_actions / self.conformance_mapping.total_sgoal_producing_actions)
        return Expansion(1.0, 1.0)
    
    def get_expansion_factor(self,
                             indices: Optional[Union[int, range]] = None,
                             action_type: Optional[ActionType] = None,
                             accu_step: Optional[int] = None
                             ) -> Expansion:
        """
        Get the expansion factor of a given conformance constraining sub-goal stage index refined by this plan.
        Returns 1.0 if the plan is classical.
        
        This method is a proxy for the method of the same name provided by the conformance mapping class.
        """
        if self.is_refined:
            if (isinstance(indices, int) and indices not in self.conformance_mapping.constraining_sgoals):
                raise ValueError(f"The sub-goal index {indices} is not refined by {self!s}.")
            
            if indices is None or isinstance(indices, range):
                _range: Iterable
                if indices is None: _range = self.conformance_mapping.constraining_sgoals.keys()
                elif isinstance(indices, range): _range = indices
                
                length: float = 0.0; action: float = 0.0
                for index in _range:
                    expansion: Expansion = self.get_expansion_factor(index, action_type, accu_step)
                    length += expansion.length; action += expansion.action
                return Expansion(length / len(_range), action / len(_range))
            
            steps: list[int] = self.conformance_mapping.current_sgoals(indices)
            if (max(self.conformance_mapping.constraining_sgoals_range) == indices
                and self.has_trailing_plan):
                steps.extend([step for step in range(max(steps), self.end_step)])
            
            if action_type is None:
                return Expansion(len([step for step in self.conformance_mapping.current_sgoals(indices)
                                      if accu_step is None or step <= accu_step]),
                                 sum(len(self[step]) for step in self.conformance_mapping.current_sgoals(indices)
                                     if accu_step is None or step <= accu_step) / len(self.conformance_mapping.constraining_sgoals[indices]))
            
            return Expansion(len([step for step in self.conformance_mapping.current_sgoals(indices)
                                  if any(self.__matches(action, action_type) for action in self.get(step, []))]),
                             sum(len([action for action in self.get(step, []) if self.__matches(action, action_type)])
                                 for step in self.conformance_mapping.current_sgoals(indices)) / len(self.conformance_mapping.constraining_sgoals[indices]))
        
        return Expansion(1.0, 1.0)
    
    def get_expansion_deviation(self,
                                indices: Optional[Union[int, range]] = None,
                                action_type: Optional[ActionType] = None,
                                accu_step: Optional[int] = None
                                ) -> Expansion:
        """
        The expansion factor standard deviation over all conformance constraining sub-goal stages refined by this plan.
        
        This measures the variation or spread, of the matching child sub-goal stage achieving transitions.
        A value of zero indicates perfectly balanced refinement trees, such that the refining child sub-plan lengths are exactly equal for all refinement trees.
        """
        if self.is_refined and len(self.conformance_mapping.constraining_sgoals) > 1:
            _range: Iterable
            if indices is None: _range = self.conformance_mapping.constraining_sgoals.keys()
            elif isinstance(indices, range): _range = indices
            elif isinstance(indices, int): _range = [indices]
            else: raise TypeError(f"Indices must be None, and int, or a range. Got; {indices} or type {type(indices)}.")
            
            expansion_factors: list[Expansion] = []
            for index in _range:
                if (factor := self.get_expansion_factor(index, action_type, accu_step)).length != 0:
                    expansion_factors.append(factor)
            
            if len(expansion_factors) >= 2:
                return Expansion(statistics.stdev([factor[0] for factor in expansion_factors]),
                                 statistics.stdev([factor[1] for factor in expansion_factors]))
        
        return Expansion(0.0, 0.0)
    
    def get_degree_of_balance(self, indices: Optional[Union[int, range]] = None, action_type: Optional[ActionType] = None, accu_step: Optional[int] = None) -> Expansion:
        """
        The degree to which the refinement trees of the plan are balanced.
        This is the coefficient of deviation, and is good when comaparing the balancing of plans with different length, as it is normalised on the plan length.
        
        A value of zero indicates perfect balance, such that the length of the sub-plans are equal.
        This is important because its indicates that the matching child actions of the conformance refinement a evenly spread over the refined plan's length.
        Such conformance constraints tend to be stronger and more consistently restrict the search space/reduce search time over the plan length.
        """
        if self.is_refined:
            return Expansion(*(dev / fac for dev, fac in zip(self.get_expansion_deviation(indices, action_type, accu_step),
                                                             self.get_expansion_factor(indices, action_type, accu_step))))
        return Expansion(0.0, 0.0)

@dataclass(frozen=True)
class RefinementSchema:
    """
    Encapsulates a refinement schema.
    
    Fields
    ------
    `constraining_sgoals: dict[int, list[SubGoal]]`
    
    `problem_divisions: list[DivisionPoint] = []`
    """
    
    constraining_sgoals: dict[int, list[SubGoal]]
    problem_divisions: list[DivisionPoint] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"(Level = {self.level}, Sub-goals range = [{min(self.constraining_sgoals)}-{max(self.constraining_sgoals)}], Total problem divisions = {len(self.problem_divisions)})"
    
    @property
    def level(self) -> int:
        return int(self.constraining_sgoals[1][0]['L'])
    
    @classmethod
    def from_json(cls, json: dict) -> "RefinementSchema":
        return cls({int(index) : [SubGoal(sgoal) for sgoal in eval(repr(sgoals))] for index, sgoals in json["sub_goal_stages"].items()},
                   [eval(repr(point)) for point in json["problem_divisions"]])
    
    @property
    def serialisable_dict(self) -> dict[int, dict[int, dict[str, str]]]:
        "Convert the refinement schema to a json serialisable dictionary."
        _dict: dict[str, dict[int, str]] = {}
        
        _dict["sub_goal_stages"] = {}
        for index, sgoals in self.constraining_sgoals.items():
            _dict["sub_goal_stages"][index] = [sgoal._dict for sgoal in sgoals]
        
        _dict["problem_divisions"] = [repr(division) for division in self.problem_divisions]
        
        return _dict

@dataclass(frozen=True)
class HierarchicalPlan(_collections_abc.Mapping, AbstractionHierarchy):
    """
    Encapsulates a hierarchical plan.
    An immutable dataclass.
    This is returned for every experiemnetal run, so should contain all the plans and every statistic that is needed.
    
    These together form the online planning diagram;
         - Contains all the partial plans at each level,
         - The problem division tree,
         - The links between these depicting;
             - The planning increments (how many levels are in each increment and when the planner changes level),
             - The hierarchical progression on each increment,
             - When each problem templated by the scenarios in the problem division tree is solved.
    
    Fields
    ------
    `concatenated_plans : dict[int, MonolevelPlan]` - A concatenated monolevel plan at each abstraction level.
    If this is a hierarchical conformance refinement plan, these are the concatenations of the partial plans at the same level.
    Otherwise, if this hierarchical plan contains only classical plans, these are classical complete plans.
    
    `partial_plans : dict[int, dict[int, MonolevelPlan]]` -
    
    `problem_division_tree : dict[int, list[DivisionScenario]]` -
    """
    concatenated_plans: dict[int, MonolevelPlan]                ## maps: abstraction level -> concatenated (complete) monolevel plan
    partial_plans: dict[int, dict[int, MonolevelPlan]]          ## maps: abstraction level x online increment -> (partial) monolevel plan
    problem_division_tree: dict[int, list[DivisionScenario]]    ## maps: abstraction level x online increment -> division scenario
    
    ######
    ## Make the hierarchical plan a mapping from abstraction levels to concatenated monolevel plans
    
    def __getitem__(self, level: int) -> MonolevelPlan:
        return self.concatenated_plans[level]
    
    def __contains__(self, level: int) -> bool:
        return level in self.concatenated_plans
    
    def __iter__(self) -> Iterator[int]:
        yield from self.concatenated_plans.keys()
    
    def __len__(self) -> int:
        return len(self.concatenated_plans)
    
    def get_plan_sequence(self, level: int) -> list[MonolevelPlan]:
        "Get the monolevel plan sequence at a given abstraction level in this hierarchical plan."
        if self.is_hierarchical_refinement:
            return list(self.partial_plans[level].values())
        return [self.concatenated_plans[level]]
    
    ######
    ## Make the hierarchical plan an abstraction hierarchy
    
    @property
    def top_level(self) -> int:
        "The top abstraction level of the hierarchical plan, always greater than or equal to the bottom level."
        return max(self.concatenated_plans)
    
    @property
    def bottom_level(self) -> int:
        "The bottom abstraction level of the hierarchical plan, always greater than or equal to 1 (the ground level)."
        return min(self.concatenated_plans)
    
    @property
    def is_hierarchical_refinement(self) -> bool:
        """
        Whether this is a hierarchical refinement plan.
        This is true iff it was generated by the hierarchical planning algorithm with conformance constraints enabled.
        
        A hierarchical refinement plan contains at each level both;
            - a possibly complete concatenated conformance refinement monolevel plan,
            - and a non-empty sequence of possibly partial conformance refinement monolevel plans,
        Otherwise, it only contains complete classical plans, and no partial plans.
        """
        return bool(self.partial_plans) and self.concatenated_plans[self.bottom_level].is_refined
    
    ######
    ## Hierarchical plan statistics
    
    def get_minimum_execution_time(self, level: int, problem_number: int, per_action: bool = False) -> float:
        """
        The minimum required execution time for non-final partial plans,
        that would allow the robot to continue without ever pausing,
        based on the wait time until the next partial plan at the given level is yielded.
        """
        if not self.is_hierarchical_refinement:
            return 0.0
        if problem_number == len(self.partial_plans[level]):
            return 0.0
        raw_wait_time: float = self.get_wait_time(level, problem_number + 1)
        if per_action:
            return raw_wait_time / self.partial_plans[level][problem_number].total_actions
        return raw_wait_time
    
    def get_average_minimum_execution_time(self, level: int, per_action: bool = False) -> float:
        """The mean minimum exectuion time over all partial problems."""
        if not self.is_hierarchical_refinement:
            return 0.0
        if len(self.partial_plans[level]) == 1:
            return 0.0
        return statistics.mean(self.get_minimum_execution_time(level, problem_number, per_action)
                               for problem_number, increment_number
                               in enumerate(self.partial_plans[level], start=1)
                               if problem_number != len(self.partial_plans[level]))
    
    def get_yield_time(self, level: int, problem_number: int) -> float:
        """
        The yield time of the partial plan that solves a given problem number at a given level in this hierarchical plan.
        If this hierarchical plan was not generated by the hierarchical conformance refinement planning algorithm,
        then if the plan at the given level exists, it was generated as a independent classical plan,
        and its yield time is identical to completion time.
        """
        if not self.is_hierarchical_refinement:
            return self.concatenated_plans[level].grand_totals.total_time
        increment_sequence: list[int] = list(self.partial_plans[level])
        yield_increment: int = increment_sequence[problem_number - 1]
        return sum(self.partial_plans[_level][increment].grand_totals.total_time
                   for _level in reversed(self.constrained_level_range(bottom_level=level))
                   for increment in self.partial_plans[_level]
                   if increment <= yield_increment)
    
    def get_wait_time(self, level: int, problem_number: int, per_action: bool = False) -> float:
        """
        The wait time of the partial plan that solves a given problem number at a given level in this hierarchical plan.
        This is the time difference between yielding the previous partial plan at the given level, and yielding the given.
        """
        raw_wait_time: float = 0.0
        if problem_number == 1:
            raw_wait_time = self.get_yield_time(level, problem_number)
        else: raw_wait_time = self.get_yield_time(level, problem_number) - self.get_yield_time(level, problem_number - 1)
        if per_action:
            return raw_wait_time / self.partial_plans[level][problem_number].total_actions
        return raw_wait_time
    
    def get_average_wait_time(self, level: int, exclude_initial: bool = False, per_action: bool = False) -> float:
        "The average partial plan wait time at a given level in this hierarchical plan."
        if not self.is_hierarchical_refinement:
            return self.get_completion_time(level)
        if exclude_initial and len(self.partial_plans[level]) == 1:
            return 0.0
        return statistics.mean(self.get_wait_time(level, problem_number, per_action)
                               for problem_number, increment_number
                               in enumerate(self.partial_plans[level], start=1)
                               if (not exclude_initial or problem_number > 1))
    
    def get_latency_time(self, level: int) -> float:
        "The latency time (yield time of the first partial plan) at a given level in this hierarchical plan."
        return self.get_yield_time(level, 1)
    
    def get_completion_time(self, level: int) -> float:
        "The completion time (sum of latency time and total planning time of all non-initial partial plans) at a given level in this hierarchical plan."
        _top_level: Optional[int] = None
        if not self.concatenated_plans[level].is_refined:
            _top_level = level
        return sum(self.concatenated_plans[_level].grand_totals.total_time
                   for _level in reversed(self.constrained_level_range(bottom_level=level, top_level=_top_level)))
    
    def get_grand_totals(self, level: int) -> ASH_Statistics:
        "The grand total timing/memory statistics of the concatenated complete monolevel plan at a given level in this hierarchical plan."
        return self.concatenated_plans[level].planning_statistics.grand_totals
    
    def get_hierarchical_problem_sequence(self) -> list[tuple[int, int, int, int]]:
        """
        An ordered list of (sequence number - level - increment - problem number) four-tuples,
        defining the problem sequence ordering over the hierarchy, useful when constructing an online planning diagram.
        """
        problem_sequence: list[tuple[int, int, int, int]] = []
        if self.is_hierarchical_refinement:
            for iteration in range(1, max(self.partial_plans[self.bottom_level]) + 1):
                for level in reversed(self.level_range):
                    for problem_number, inner_iteration in enumerate(self.partial_plans[level], start=1):
                        if inner_iteration == iteration:
                            sequence_number: int = len(problem_sequence) + 1
                            problem_sequence.append((sequence_number, level, iteration, problem_number))
        return problem_sequence
    
    @property
    def execution_latency_time(self) -> float:
        "The execution latency time (ground level latency time) of this hierarchical plan."
        return self.get_latency_time(self.bottom_level)
    
    @property
    def absolution_time(self) -> float:
        "The absolution time (ground level completion time) of this hierarchical plan (equivalent to overall grand total time taken to complete all levels)."
        return self.get_completion_time(self.bottom_level)
    
    @property
    def required_memory(self) -> ASP.Memory:
        "The minimum amount of memory required to complete all levels in the hierarchical plan."
        return max(plan.planning_statistics.grand_totals.memory for plan in self.concatenated_plans.values())
    
    ######
    ## Problem division
    
    @property
    def average_divisions_per_scenario(self) -> float:
        if self.problem_division_tree:
            return statistics.mean(scenario.get_total_divisions(shifting_only=False) for level in self.problem_division_tree for scenario in self.problem_division_tree[level])
        return 0.0
    
    @property
    def average_divisions_per_level(self) -> float:
        if self.problem_division_tree:
            return statistics.mean(sum(scenario.get_total_divisions(shifting_only=False) for scenario in self.problem_division_tree[level]) + (len(self.problem_division_tree[level]) - 1) for level in self.problem_division_tree)
        return 0.0
    
    def get_division_points(self, level: int, produced_from: bool = True) -> list[DivisionPoint]:
        """
        Get a list of all division points that divide the abstract plan at the given level.
        
        Parameters
        ----------
        `level : int` - An integer defining the abstraction level at which to get the division points.
        
        `produced_from : bool` - A Boolean, True to get the division points produced from the given level, False to get the division points committed to the given level.
        Division points are produced from a level iff it is an abstract level and a division strategy was used to divide the refinement planning problem of that abstract plan.
        Division points are committed to a level iff the level is not the top-level and invloved refining one such abstract plan.
        
        Returns
        -------
        `list[DivisionPoint]` - A complete list of all division points that have currently been committed to or produced from the given level.
        """
        _level: int = level if produced_from else level + 1
        # if _level not in self.problem_division_tree:
        #     raise ASH_InvalidInputError(f"No problem divisions {'produced from' if produced_from else 'committed to'} planning at level {_level}.")
        
        ## Make a list of all the division points across all the scenarios;
        ##      - Fabricate the inherited divisions for each one,
        ##      - Overwrite the last inherited division point each time,
        ##          - Since this will be a duplicate of the first inherited division of the next scenario.
        division_points: list[DivisionPoint] = []
        for scenario in self.problem_division_tree.get(_level, []):
            division_points[(len(division_points) - 1):] = scenario.get_division_points(shifting_only=False, fabricate_inherited=True)
        return division_points
    
    def get_refinement_schema(self, level: int) -> RefinementSchema:
        """
        Get the refinement problem schema at the given abstraction level.
        
        A refinement schema includes a sequence of sub-goal stages and a merged division scenario.
        This is all the required data needed to solve a refinement planning problem at the given level, with out having to first generate the abstract plans at the previous levels.
        The initial states and final-goals are not stored in the schema, and must be re-generated from the problem definition.
        
        Parameters
        ----------
        `level : int` - An integer defining the abstraction level at which to get the schema.
        
        Returns
        -------
        `RefinementSchema` - An object encapsulating a refinement schema.
        
        Raises
        ------
        `ASH_InvalidInputError` - If there is either;
            - No plan at the given level,
            - The plan is not refined.
        """
        if (plan := self.concatenated_plans.get(level, None)) is None:
            raise ASH_InvalidInputError(f"Plan at level {level} does not exist.")
        if not plan.is_refined:
            raise ASH_InvalidInputError(f"Plan at level {level} is not a refined plan.")
        return RefinementSchema(plan.conformance_mapping.constraining_sgoals,
                                self.get_division_points(level, produced_from=False))
    
    ######
    ## Serialisation
    
    @property
    def serialisable_dict(self) -> dict[int, dict[int, dict[str, Union[str, list[dict[str, str]]]]]]:
        "Convert the hierarchical plan to a json serialisable dictionary."
        _dict = {}
        for level, plan in self.concatenated_plans.items():
            _dict[level] = {}
            for step in plan.actions:
                _dict[level][step] = {}
                _dict[level][step]["actions"] = [action.select('R', 'A') for action in plan.actions[step]]
                if plan.is_abstract:
                    _dict[level][step]["produced_sgoals"] = [sgoals.select('R', 'A', 'F', 'V') for sgoals in plan.produced_sgoals[step]]
                if plan.is_refined:
                    _dict[level][step]["current_sgoals_index"] = str(current_sgoals := plan.conformance_mapping.current_sgoals.get(step))
                    _dict[level][step]["achieved_sgoals_index"] = str(plan.conformance_mapping.sgoals_achieved_at.get(current_sgoals))
        return _dict

def format_actions(actions: dict[int, list[Action]], current_sub_goals: dict[int, list[SubGoal]]) -> list[str]:
    """Function used to verbose format actions and the current sub-goals of the conformance mapping."""
    output: list[str] = []
    for step in set(current_sub_goals.keys()) | set(actions.keys()):
        output.append(f"Step {step}:")
        if current_sub_goals:
            output.append("    Current Sub-goals:")
            if (sub_goals := current_sub_goals.get(step, None)) is not None:
                for sub_goal in sub_goals:
                    output.append(f"        [Index = {sub_goal['I']}] {sub_goal['R']} : {sub_goal['A']} -> {sub_goal['F']} = {sub_goal['V']}")
                output.append("    Achieved Sub-goals:")
                next_sub_goals = current_sub_goals.get(step + 1, [])
                for sub_goal in sub_goals:
                    if sub_goal not in next_sub_goals:
                        output.append(f"        [Index = {sub_goal['I']}] {sub_goal['R']} : {sub_goal['A']} -> {sub_goal['F']} = {sub_goal['V']}")
        output.append("    Planned actions:")
        for action in actions.get(step, []):
            output.append(f"        {action['R']} : {action['A']}")
    return output

#############################################################################################################################################
#############################################################################################################################################
####################   █████  ███████ ██   ██               ██████   ██████  ███    ███  █████  ██ ███    ██ ███████ ########################
####################  ██   ██ ██      ██   ██               ██   ██ ██    ██ ████  ████ ██   ██ ██ ████   ██ ██      ########################
####################  ███████ ███████ ███████     █████     ██   ██ ██    ██ ██ ████ ██ ███████ ██ ██ ██  ██ ███████ ########################
####################  ██   ██      ██ ██   ██               ██   ██ ██    ██ ██  ██  ██ ██   ██ ██ ██  ██ ██      ██ ########################
####################  ██   ██ ███████ ██   ██               ██████   ██████  ██      ██ ██   ██ ██ ██   ████ ███████ ########################
#############################################################################################################################################
#############################################################################################################################################

## Constants defining the program parts in domain files
_DOMAIN_PROGRAM_PARTS: Final[set[str]] = {"domain_sorts",
                                          "action_effects(t)",
                                          "action_preconditions(t)",
                                          "variable_relations(t)",
                                          "abstraction_mappings(t)"}

_PROBLEM_PROGRAM_PARTS: Final[set[str]] = {"entities",
                                           "static_state",
                                           "initial_state",
                                           "goal_state"}

@enum.unique
class DomainModelType(enum.Enum):
    """
    An enumeration defining the three possible abstract model types currently supported by ASH.
    
    Items
    -----
    `Tasking = "tasking"`
    
    `Condensed = "condensed"`
    
    `Reduced = "reduced"`
    
    `Ground = "ground"`
    """
    Tasking = "tasking"
    Condensed = "condensed"
    Reduced = "reduced"
    Ground = "ground"

class PlanningDomain(AbstractionHierarchy):
    """
    Represents a hierarchical planning domain definition used by ASH.
    This is a two-tuple of:
        - Domain Definition = {classes,
                               sorts = {fluents, actions, statics},
                               rules = {effects, preconditions, relations}, abstraction mappings}
        - World Structure = {static state,
                             entity declarations,
                             ancestry relation declarations}
    
    Properties
    ----------
    `name : str` - A string defining an arbitrary name for this planning domain.
    The domain is given a name of the form '<name> #n' where n is an ordinal number.

    `domain_files : list[str]` - A list of strings defining the files from which this domain was loaded.
    
    `optional_parts : list[str]` - A list of strings defining names of optional
    program parts from `ASH.OPTIONAL_PROGRAM_PARTS` that this planning domain contains.
    If the planning domain does not contain an optional part, it will need to be provided
    explicitly so solve a planning problem with this domain.
    
    `top_level : int` - An integer defining the maximum abstraction level of this planning domain.
    The abstraction range is [1-top_level].
    """
    
    __slots__ = ("__name",
                 "__domain_files",
                 "__has_optional_part",
                 "__top_level",
                 "__model_types")
    
    __domain_counter: dict[str, int] = {}
    @staticmethod
    def __get_domain_number(name: str) -> int:
        PlanningDomain.__domain_counter[name] = PlanningDomain.__domain_counter.get(name, 0) + 1
        return PlanningDomain.__domain_counter[name]
    
    def __init__(self, domain_files: list[str], name: Optional[str] = None):
        """
        Instantiate a planning domain with an optional name from a list of domain files.
        
        Parameters
        ----------
        `domain_files : list[str]` - A list of strings defining ASP (.lp) damain files to load.
        
        `name : {str | None} = None` - A string defining an arbitrary name for this planning domain.
        The string is converted to the form 'name #n' where n is an ordinal number.
        If not given, or None, the program is given a name of the form 'Anon #n'.
        
        Raises
        ------
        `ValueError` - If either a file in the list of domain files could not be read or is not an ASP (.lp) format.
        """
        formatted_files: str = "Domain files = [{0}]".format(('\n' + (' ' * len('Domain files = ['))).join(map(str, domain_files)))
        _planner_logger.debug(f"Instantiating new planning domain:\nName = {name}\n{formatted_files}")
        
        if len(domain_files) != len(set(domain_files)):
            log_and_raise(ValueError, f"The given domain files contain a duplicate:\n{formatted_files}")
        
        self.__name: str = f"{name if name is not None else 'Anon'} #{PlanningDomain.__get_domain_number(str(name))}"
        checked_required_parts: list[bool] = [False for i in range(len(_DOMAIN_PROGRAM_PARTS))]
        self.__has_optional_part: dict[str, bool] = {part : False for part in _PROBLEM_PROGRAM_PARTS}
        
        ## Load domain files
        for domain_file in domain_files:
            if domain_file.endswith(".lp"):
                try: ## Check each file exists and is a valid ASP format
                    with open(domain_file, "r") as file_reader:
                        ## Check the domain files contain the correct program parts
                        file_: str = file_reader.read()
                        for i, part in enumerate(_DOMAIN_PROGRAM_PARTS):
                            checked_required_parts[i] = checked_required_parts[i] or (f"#program {part}." in file_)
                        for part in _PROBLEM_PROGRAM_PARTS:
                            self.__has_optional_part[part] = self.__has_optional_part[part] or (f"#program {part}." in file_)
                except OSError as error:
                    log_and_raise(ValueError, f"Unable to create a planning domain. Cannot read the domain file {domain_file}.", from_exception=error)
            else: log_and_raise(ValueError, f"Unable to create a planning domain. The file {domain_file} is not an ASP (.lp) format.")
        if not all(checked_required_parts):
            log_and_raise(ValueError, f"Unable to create a planning domain. The domain files do not contain all of the required program parts.")
        self.__domain_files: list[str] = domain_files
        _planner_logger.debug(f"The domain files {repr(domain_files)} were loaded successfully, "
                              f"contain all domain program parts, and contain problem program parts: {self.__has_optional_part}.")
        
        ## Find the maximum abstraction level
        program = ASP.LogicProgram.from_files(["./core/ASH.lp"] + domain_files, silent=True, warnings=False)
        answer: ASP.Answer = program.solve([ASP.Options.threads(1)],
                                           base_parts=[ASP.BasePart(name="abstraction_levels",
                                                                    args=[0, "hierarchical"]),
                                                       ASP.BasePart(name="domain_sorts")])
        self.__top_level: int = max([al.arguments[0].number for al in answer.fmodel.get_atoms("al", 1, True)])
        
        model_types: list[dict] = answer.fmodel.query("model_type", ['L', 'T'],
                                                      cast_to={'L' : lambda symbol: int(str(symbol)),
                                                               'T' : lambda symbol: DomainModelType(str(symbol))})
        self.__model_types: dict[int, DomainModelType] = {}
        for level in self.level_range:
            for model_type in model_types:
                if model_type['L'] == level:
                    self.__model_types[model_type['L']] = model_type['T']
            if level not in self.__model_types:
                raise ASH_InternalError(f"No model type found for level {level}.")
        
        _planner_logger.debug(f"{self} instantiated successfully")
    
    def __str__(self) -> str:
        return f"Planning Domain {self.__name} :: Abstraction range = [1-{self.__top_level}]"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__domain_files}, {self.__name})"
    
    def copy(self, rename: Optional[str] = None) -> "PlanningDomain":
        """
        Create a copy of this hierarchical planning domain definition.
        This is an expensive operation because ASH will re-check if the domain files exist.
        
        Parameters
        ----------
        `rename : {str, None}` - An optional string to name the copy of this planning domain.
        If not given or None, the copy will have name 'Copy of <name> #n' where 'n' is an integer.
        
        Returns
        -------
        `PlanningDomain` - An exact copy of this planning domain.
        """
        return PlanningDomain(self.domain_files, name=rename if rename else f"Copy of <{self.name}>")
    
    @property
    def name(self) -> str:
        "A string defining the name of the planning domain instance of the form 'name #n' where n is an integer."
        return self.__name
    
    @property
    def domain_files(self) -> list[str]:
        "A list of strings defining the file paths from which this domain was loaded."
        return self.__domain_files
    
    @property
    def top_level(self) -> int:
        return self.__top_level
    
    @property
    def has_tasking_model(self) -> bool:
        return DomainModelType.Tasking in self.__model_types.values()
    
    def get_model_type(self, level: int) -> DomainModelType:
        "Get the domain model type at a given abstraction level in this hierarchical planning domain definition."
        if level not in self.level_range:
            raise ValueError(f"Abstraction level {level} is not in the range of {self}.")
        return self.__model_types[level]

#############################################################################################################################################
#############################################################################################################################################
#######################   █████  ███████ ██   ██         ██████  ██       █████  ███    ██ ███    ██ ███████ ██████  ########################
#######################  ██   ██ ██      ██   ██         ██   ██ ██      ██   ██ ████   ██ ████   ██ ██      ██   ██ ########################
#######################  ███████ ███████ ███████  █████  ██████  ██      ███████ ██ ██  ██ ██ ██  ██ █████   ██████  ########################
#######################  ██   ██      ██ ██   ██         ██      ██      ██   ██ ██  ██ ██ ██  ██ ██ ██      ██   ██ ########################
#######################  ██   ██ ███████ ██   ██         ██      ███████ ██   ██ ██   ████ ██   ████ ███████ ██   ██ ########################
#############################################################################################################################################
#############################################################################################################################################

VerbosityValue = NamedTuple("VerbosityValue", (("level", int), ("log", int)))
@enum.unique
class Verbosity(enum.Enum):
    """
    Enumeration defining the verbosity of a hierarchical planner's output to the CLI.
    
    Items
    -----
    Verbose = 4 - Log everything to the CLI.
    
    Standard = 3 - Log planned actions.
    
    Simple = 2 - Do no print out plans or problem specifics.
    
    Minimal = 1 - Only progress bars and statistics.
    
    Disable = 0 - Log nothing.
    """
    Verbose = VerbosityValue(4, logging.INFO)
    Standard = VerbosityValue(3, logging.INFO)
    Simple = VerbosityValue(2, logging.DEBUG)
    Minimal = VerbosityValue(1, logging.DEBUG)
    Disable = VerbosityValue(0, logging.DEBUG)

@enum.unique
class OnlineMethod(enum.Enum):
    """
    Enumeration defining the two types of online method used in refinement planning.
    
    The top-level of the current increment's level range must be between the highest and lowest valid
    planning levels (inclusive), and the bottom-level is between the top-level and ground level.
    
    Items
    -----
    `GroundFirst = "ground-first"` -
    
    `CompleteFirst = "complete-first"` -
    
    `Hybrid = "hybrid"` -
    """
    GroundFirst = "ground-first"
    CompleteFirst = "complete-first"
    Hybrid = "hybrid"

@enum.unique
class StateRepresentation(enum.Enum):
    """
    Enumeration defining the three types of state represenation that can be used over a hierarchical planning domain definition.
    
    Items
    -----
    `Classical = "classical"` - The classical approach represents the state at the current planning level only.
    
    `Refinement = "refinement"` - The conformance refinement approach represents the state at the current planning level and previous adjacent abstraction level simultaneously.
    This is equivalent to maintaining the state represenation in an original domain model and its abstract model.
    Such a representation is required to allow reasoning about conformance between plans generated in such models during refinement planning.
    This is because the sub-goal stages produced from the effects of the abstract plan's actions are defined by the abstract state representation, which may be different to the original state representation.
    Thus, in order to reason about the achievement of those sub-goal stages, we must maintain the abstract state representation when refining the abstract plan in the original model.
    This is done by mapping the original state upwards, using the state abstraction mapppings introduced by the abstract model.
    These mappings map each original level state to exactly one abstract state (called its parent state), but any given abstract state may be mapped to from many original states (called its child states).
    
    `Hierarchical = "hierarchical"` - The hierarchical approach represents the state at all available abstraction levels simultaneously.
    This is used to generate complete conforming initial states and final-goals over the entire hierarchy prior to planning.
    Thus, it is used to ensure that plans at every level in the hierarchy, start and end in states that are conforming, i.e. they map to each other.
    """
    
    Classical = "classical"
    Refinement = "refinement"
    Hierarchical = "hierarchical"

@enum.unique
class ConformanceType(enum.Enum):
    """
    Enumeration defining the two types of conformance constraint enforcement used in refinement planning.
    
    Items
    -----
    `SequentialAchievement = "sequential"` - Inidividual sub-goals of a stage can be achieved by any child of the refined sub-plan of a sub-goal producing abstract action.
    Only a sub-set of the sub-goals have to be satisfied in the matching child.
    
    `SimultaneousAchievement = "simultaneous"` - All sub-goals of a stage must be achieved in the matching child of the refined sub-plan of a sub-goal producing abstract action.
    """
    
    SequentialAchievement = "sequential"
    SimultaneousAchievement = "simultaneous"

@enum.unique
class PreemptMode(enum.Enum):
    """
    The final-goal preemptive achievement enforcement mode.
    
    Items
    -----
    `Heuristic = "heuristic"` - Preemptive achievement is applied as a domain heuristic to the ASP solver.
    This affects the ASP program solving process throughout all search steps in the planning progression.
    This avoids optimising the answer sets of the program at the end of search, but can reduce search efficiency for long plan lengths.
    
    `Optimise = "optimise"` - Preemptive achievement is applied as a model optimisation process.
    This affects only the generation/selection of the optimal answer sets of the program at the end of search.
    This avoids affecting the ASP based heuristic, but there is an additional overhead on generating the final answer set.
    """
    
    Heuristic = "heuristic"
    Optimise = "optimise"

class ConsistencyCheck(NamedTuple):
    """Used for hierarchical problem consistency checks."""
    
    consistent: StateVariableMapping
    inconsistent: StateVariableMapping

@dataclass(frozen=True)
class Solution:
    """
    
    Fields
    ------
    `answer : ASP.Answer` - The ASP answer object returned for the search.
    
    `last_achieved_sgoals : int` - The last sub-goal stage sequence index achieved by the search, 1 if classical planning was used.
    
    `overhead_time : float = 0.0` - The total overhead time spent in sequential yield planning.
    
    `sequential_yield_steps : {dict[int, int] | None} = None` - The planning steps upon which each sub-goal stage index was minimally greedily achieved (Maps: sub-goal stage index -> satisfying state time step).
    
    `reactive_divisions : list[DivisionPoint] = []` - A list of reactive division made during planning, in the order in which they were committed.
    """
    answer: ASP.Answer
    last_achieved_sgoals: int
    overhead_time: float = 0.0
    sequential_yield_steps: Optional[dict[int, int]] = None
    reactive_divisions: list[DivisionPoint] = field(default_factory=list)

@dataclass
class MonolevelProblem:
    """
    Encapsulates a problem specification.
    
    Fields
    ------
    `level: int` - The abstraction level of the problem.
    
    `concurrency: bool` - Whether action concurrency is enabled, if disabled actions can only be planned sequentially.
    
    `conformance: bool` - Whether to apply conformance constraints, enabling conformance refinement planning mode.
    
    `conformance_type: ConformanceType` - The conformance constraint application type; simultaneous or sequential sub-goal stage achievement.
    
    `first_sgoals: int` - The first in sequence sub-goal stage index value (inclusive) of a conformance refinement planning problem.
    
    `last_sgoals: int`- The last in sequence sub-goal stage index value (inclusive) of a conformance refinement planning problem.
    
    `start_step: int` - The start step of the problem, the first action is planned on the step following the start.
    
    `is_initial: bool` - Whether the problem starts in the initial state, true iff the start step is zero.
    
    `is_final: bool` - Whether the problem is final, if true a solution to the problem must achieve the final-goal.
    
    `complete_planning: bool` - Whether the problem is complete, true iff the problem is both initial and final.
    
    `sequential_yield: bool` - Whether the conformance refinement planning problem should be solved in sequential yield mode.
    
    `reactive_divisions: bool` - Whether reactive divisions may occur during sequential yield planning.
    
    `use_search_length_bound: bool` - Whether to use the minimum search length bound.
    
    `search_length_bound: int` - The value of the minimum search length bound.
    
    Properties
    ----------
    `sgoals_range : range` - The contiguous sub-goal stage indices range of the problem, equivalent to [first_sgoals-last_sgoals].
    """
    
    ## Basic parameters
    level: int
    concurrency: bool
    
    ## Planning mode and refinement parameters
    conformance: bool
    conformance_type: ConformanceType
    first_sgoals: int
    last_sgoals: int
    
    ## Problem type parameters
    start_step: int
    is_initial: bool
    is_final: bool
    complete_planning: bool
    
    ## Optional parameters
    sequential_yield: bool
    reactive_divisions: bool
    use_search_length_bound: bool
    search_length_bound: int
    
    def __str__(self) -> str:
        return "\n".join(f"{field.name} = {getattr(self, field.name)!s}" for field in fields(self))
    
    @property
    def problem_description(self) -> str:
        "A formatted multi-line description of the monolevel problem specification."
        if self.conformance:
            conformance_description: str = (f"{'complete' if self.complete_planning else 'partial'} conformance refinement "
                                            f"({self.conformance_type.value}) with sgoals range [{self.first_sgoals}-{self.last_sgoals}]")
        return (f"[Level = {self.level}] {conformance_description if self.conformance else 'classical'} : "
                f"Concurrency {'enabled' if self.concurrency else 'disabled'} : "
                f"Minimum search length bound {'enabled' if self.use_search_length_bound else 'disabled'} with value {self.search_length_bound}")
    
    @property
    def sgoals_range(self) -> range:
        "The contiguous sub-goal stage indices range of the problem, equivalent to [first_sgoals-last_sgoals]."
        return range(self.first_sgoals, self.last_sgoals + 1)
    
    def create_program_parts(self,
                             save_grounding: bool,
                             minimise_actions: bool,
                             order_fgoal_achievement: bool,
                             preempt_positive_fgoals: bool,
                             preempt_negative_fgoals: bool,
                             preempt_mode: PreemptMode
                             ) -> ASP.ProgramParts:
        """
        Create the program parts used to make a new planning logic program from this problem specification given optimisation options.
        
        Parameters
        ----------
        `save_grounding : bool` - Whether the logic program's grounding will be saved.
        
        `minimise_actions : bool` - Whether to enable the action minimisation statement.
        
        `order_fgoal_achievement : bool` - Whether to enable the final-goal intermediate achievement ordering preference optimisation statement.
        
        `preempt_positive_fgoals : bool` - Whether to enable positive final-goal preemptive achievement.
        
        `preempt_negative_fgoals : bool` - Whether to enable negative final-goal preemptive achievement.
        
        `preempt_mode : PreemptMode` - The mode by which the final-goal preemptive achievement is applied.
        
        Returns
        -------
        `ASP_Parser.ProgramParts` - The created program parts.
        """
        ## Base program parts are grounded only once in the start state.
        base_parts = [ASP.BasePart("base"),
                      ## ASH Modules
                      ASP.BasePart(name="abstraction_levels",
                                   args=(self.level,
                                         (StateRepresentation.Refinement
                                          if self.conformance else
                                          StateRepresentation.Classical).value)),
                      ASP.BasePart(name="instance_module"),
                      
                      ## Domain definition sorts definition
                      ASP.BasePart(name="domain_sorts"),
                      
                      ## Problem specifics for initial state
                      ASP.BasePart(name="entities"),
                      ASP.BasePart(name="static_state"),
                      ASP.BasePart(name="ash_initial_state"),
                      ASP.BasePart(name="ash_goal_state")]
        
        ## Incremental parts are grounded on every search step.
        inc_parts = [## ASH Modules for state representation, planning, and optimisation
                     ASP.IncPart(name="state_module",
                                 args=("#inc",
                                       self.start_step)),
                     ASP.IncPart(name="plan_module",
                                 args=("#inc",
                                       self.start_step,
                                       str(self.concurrency).lower(),
                                       str(self.is_final and not self.sequential_yield).lower())),
                     ASP.IncPart(name="optimisation_module",
                                 args=("#inc",
                                       self.start_step,
                                       str(minimise_actions).lower(),
                                       str(order_fgoal_achievement).lower(),
                                       str(preempt_positive_fgoals).lower(),
                                       str(preempt_negative_fgoals).lower(),
                                       str(preempt_mode.value).lower())),
                     ## Domain definition system law and abstraction mapping elements
                     ASP.IncPart(name="action_effects"),
                     ASP.IncPart(name="action_preconditions"),
                     ASP.IncPart(name="variable_relations"),
                     ASP.IncPart(name="abstraction_mappings")]
        
        if self.conformance:
            ## Add the conformance module iff conformance is enabled and a saved grounding is unavailable
            ## (we wouldn't be creating program parts if there was a saved grounding being used);
            ##      - Give the sub-goal stage range to refine;
            ##          - The first index is always given,
            ##          - The last index is given only if the grounding will not be saved such that this index will never be changed,
            ##      - Give the sub-goal stage achievement type applied by the conformance constraint,
            ##      - Define whether sequential yield mode is enabled.
            inc_parts.extend([ASP.IncPart(name="conformance_module",
                                          args=("#inc",
                                                self.start_step,
                                                self.first_sgoals,
                                                self.last_sgoals if not save_grounding else "none",
                                                self.conformance_type.value,
                                                str(self.sequential_yield).lower()))])
        
        return ASP.ProgramParts(base_parts, inc_parts)



@final
class HierarchicalPlanner(AbstractionHierarchy):
    """
    Hierarchical planners use conformance refinement to generate plans over
    an abstraction hierarchy according to a given planning domain and problem.
    """
    
    __slots__ = (## Variables for logging
                 "__name",                      # str
                 "__logger",                    # logging.Logger
                 
                 ## Represents domain dynamics and structure
                 "__domain",                    # PlaningDomain
                 "__domain_logic_program",      # ASP_Parser.LogicProgram
                 "__saved_groundings",          # dict[int, ASP_Parser.LogicProgram]
                 "__total_last_sgoals",         # dict[int, int]
                 
                 ## Represents hierarchical problem
                 "__initial_states",            # dict[int, list[Fluent]]
                 "__final_goals",               # dict[int, list[FinalGoal]]
                 
                 ## Represents contained plans
                 "__actions",                   # dict[int, dict[int, list[Action]]]        :: maps: abstraction level x time step -> action set
                 "__states",                    # dict[int, dict[int, list[Fluent]]]        :: maps: abstraction level x time step -> dynamic state
                 "__complete_plan",             # ReversableDict[int, bool]                 :: maps: level -> completeness of concatenated plan
                 "__sgoals",                    # dict[int, dict[int, list[SubGoal]]]       :: maps: abstraction level x time step -> sub-goal stage
                 "__statistics",                # dict[int, IncrementalStatistics]          :: maps: abstraction level x incremental statistics
                 "__partial_plans",             # dict[int, dict[int, MonolevelPlan]]       :: maps: abstraction level x online increment -> (partial) monolevel plan
                 
                 ## Represents refinement diagram
                 "__current_sgoals",            # dict[int, ReversableDict[int, int]]       :: maps: level x step -> index
                 "__sgoals_achieved_at",        # dict[int, ReversableDict[int, int]]       :: maps: level x index -> step
                 "__sequential_yield_steps",    # dict[int, dict[int, int]]                 :: maps: level x index -> step
                 "__division_scenarios",        # dict[int, list[DivisionScenario]]         :: maps: level x increment -> scenario that divides the refinement of the plan generated on that increment
                 
                 ## Miscellaneous
                 "__threads",                   # int
                 "__verbosity",                 # Verbosity
                 "__silence_clingo")            # bool
    
    @classmethod
    def __get_instance_number(cls, name: str) -> int:
        if not hasattr(cls, f"__{cls.__name__}_instance_counter"):
            cls.__instance_counter: dict[str, int] = {}
        cls.__instance_counter[name] = cls.__instance_counter.get(name, 0) + 1
        return cls.__instance_counter[name]
    
    def __init__(self, files: list[str], name: Optional[str] = None,
                 threads: int = 1, verbosity: Verbosity = Verbosity.Standard, silence_clingo: bool = True):
        """
        Instantiate a hierarchical planner from either a planning domain or a list of domain files to load.
        
        Parameters
        ----------
        `domain : {PlanningDomain, list[str]}` - Either a pre-constructed planning domain (see ASH.PlanningDomain) which the planner makes a copy of,
        or a list of strings defining the file paths of domain files to load, from which the planner will construct its own planning domain.
        
        `name : Optional[str] = None` - An arbitrary name for the planner instance.
        
        `threads : int = 1` - The number of solver threads to use.
        
        `verbosity : Verbosity = Verbosity.Standard` - A verbosity mode defining the output format from this planner to the CLI, see (ASH.Planner.Verbosity).
        
        `silence_clingo : bool` - A Boolean, True to disable all output from Clingo, False otherwise.
        
        Raises
        ------
        `TypeError` - If any argument is not of the correct type.
        """
        ## Variables for holding domains, problems and plans
        self.__domain = PlanningDomain(files)
        
        ## Load the logic program
        self.__domain_logic_program = ASP.LogicProgram.from_files(["./core/ASH.lp"] + self.__domain.domain_files,
                                                                  name="ASH", silent=silence_clingo,
                                                                  enable_tqdm=verbosity != Verbosity.Disable)
        
        ## Variables for holding saved program groundings and their current refinement goal sequence progress
        self.__saved_groundings: dict[int, ASP.LogicProgram] = {}
        self.__total_last_sgoals: dict[int, int] = {}
        
        ## Variables for logging
        self.__name: str = f"{name if name else 'Anon'} #{self.__get_instance_number(str(name))}"
        self.__logger: logging.Logger = logging.getLogger(f"ASH Planner {self.__name}")
        self.__logger.debug(f"Instantiated with program:\n{repr(self.__domain_logic_program)}")
        
        ## Maps each level to its specific initial state and final goals
        self.__initial_states: dict[int, list[Fluent]] = {} # Maps: abstraction level -> initial state
        self.__final_goals: dict[int, list[FinalGoal]] = {} # Maps: abstraction level -> final goals
        
        ## Variables for storing monolevel plans
        self.__actions: dict[int, dict[int, list[Action]]] = {} # Maps: abstraction level x time step -> action set
        self.__states: dict[int, dict[int, list[Fluent]]] = {} # Maps: abstraction level x time step -> dynamic state
        self.__complete_plan: dict[int, bool] = {}
        self.__statistics: dict[int, ASH_IncrementalStatistics] = {}
        self.__partial_plans: dict[int, dict[int, MonolevelPlan]] = {}
        
        ## Variables for storing conformance mappings and representing refinement diagrams/trees
        self.__sgoals: dict[int, dict[int, list[SubGoal]]] = {} # Maps: abstraction level x time step -> sub-goal stage
        self.__current_sgoals: dict[int, ReversableDict[int, int]] = {} # Maps: abstraction level x time step -> sub-goal index
        self.__sgoals_achieved_at: dict[int, ReversableDict[int, int]] = {} # Maps: abstraction level x sub-goal index -> time step
        self.__sequential_yield_steps: dict[int, dict[int, int]] = {} # Maps: abstraction level x sub-goal index -> time step
        
        ## Variables for problem division handling and representing problem division trees
        self.__division_scenarios: dict[int, list[DivisionScenario]] = {}
        
        ## Variables for solver options
        self.__threads: int = threads
        self.set_thread_count(threads)
        self.__verbosity: Verbosity = verbosity
        self.__silence_clingo: bool = silence_clingo
        self.set_verbosity_mode(verbosity, silence_clingo)
    
    def __str__(self) -> str:
        return f"ASH Hierarchical Planner {self.__name} : Planning domain = {self.__domain}"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({repr(self.__domain)}, {self.__name}, "
                f"{self.__threads}, {self.__verbosity}, {self.__silence_clingo})")
    
    @property
    def name(self) -> str:
        "The planner's name."
        return self.__name
    
    @property
    def domain(self) -> PlanningDomain:
        "The currently loaded planning domain."
        return self.__domain
    
    @property
    def top_level(self) -> int:
        return self.__domain.top_level
    
    @property
    def problem_initialised(self) -> bool:
        "A Boolean, True iff the planner contains an initialised problem instance, False otherwise."
        return bool(self.__initial_states) and bool(self.__final_goals)
    
    def set_thread_count(self, threads: int) -> None:
        """
        Set the number of solver threads used by the underlying Clingo ASP system.
        
        Parameters
        ----------
        `threads : int` - The number of solver threads to use.
        """
        if not isinstance(threads, int):
            log_and_raise(TypeError, f"Thread count value must be an integer. Got {threads} of type {type(threads)}.", logger=self.__logger)
        if threads > os.cpu_count(): # type: ignore
            self.__threads = os.cpu_count() # type: ignore
        elif threads < 1: self.__threads = 1
        else: self.__threads = threads
    
    def set_verbosity_mode(self, verbosity: Verbosity, silence_clingo: bool) -> None:
        """
        Sets the silence mode of ASH and the underlying Clingo ASP solver.
        
        Parameters
        ----------
        `verbosity : Verbosity` - A verbosity mode defining the output format from this planner to the CLI, see (ASH.Planner.Verbosity).
        
        `silence_clingo : bool` - A Boolean, True to disable all output from Clingo, False otherwise.
        
        Raises
        ------
        `TypeError` - If the type of either argument is not a Boolean
        """
        if not isinstance(verbosity, Verbosity):
            log_and_raise(TypeError, f"Verbosity must be from the Verbosity enum. Got {verbosity} of type {type(verbosity)}.", logger=self.__logger)
        if not isinstance(silence_clingo, bool):
            log_and_raise(TypeError, f"Clingo silence mode must be a Boolean. Got {silence_clingo} of type {type(silence_clingo)}.", logger=self.__logger)
        self.__verbosity = verbosity
        self.__silence_clingo = silence_clingo
    
    def __log_level(self, minimum: Verbosity, level: int = logging.INFO, default: int = logging.DEBUG) -> int:
        """
        Get a logging level according to whether this logger's verbosity meets a minimum requirement.
        
        Parameters
        ----------
        `minimum : Verbosity` - The minimum verbosity to log at the given level.
        
        `level : int` - The level to log at if this hierarchical planner's verbosity is equal to or above the minimum.
        
        `default : int` - The level to log at if this hierarchical planner's verbosity is less than the minimum.
        
        Returns
        -------
        `int` - The resulting logging level.
        """
        if self.__verbosity.value.level >= minimum.value.level:
            return level
        return default
    
    ################################################################################
    #### Plan extraction
    ################################################################################
    
    def purge_solutions(self) -> None:
        """
        Purge all solutions currently stored in this hierarchical planner.
        This clears; all monolevel plans, the conformance mappings, and problem divisions.
        """
        ## Clear monolevel plans
        self.__states = {}
        self.__actions = {}
        self.__complete_plan = {}
        self.__statistics = {}
        self.__partial_plans = {}
        
        ## Clear refinement diagram
        self.__sgoals = {}
        self.__current_sgoals = {}
        self.__sgoals_achieved_at = {}
        
        ## Clear problem division tree
        self.__division_scenarios = {}
    
    def get_executable_plan(self) -> MonolevelPlan:
        """
        Get the currently executable possibly partial concatenated ground level monolevel plan.
        A concatenated plan is either; a single partial or complete plan, or a possibly chained together sequence of partial plans.
        An empty plan is returned if the ground level planning problem has not yet been started or solved.
        """
        try:
            return self.get_monolevel_plan(1, 0)
        except ValueError:
            return MonolevelPlan()
    
    def get_monolevel_plan(self, level: int, start_step: int = 0, end_step: Optional[int] = None) -> MonolevelPlan:
        """
        Get the currently stored executable monolevel plan at a given abstraction level.
        The step range of the plan can optionally be constrained within a contiguous range.
        
        Parameters
        ----------
        `level: int` - The abstraction level at which to get the plan.
        
        `start_step: int = 0` - The inclusive start step of the plan to get.
        Note that the actions on the start step are not included,
        as actions follow the start state of a state transition by one step
        to occur at the same step as the end state of the transition,
        so the first action set occurs on (start_step + 1).
        
        `end_step: Optional[int] = None` - The inclusive end step of the plan to get.
        If None, the plan includes all steps greater than or equal to the start step.
        
        Returns
        -------
        `MonolevelPlan` - The resulting monolevel plan containing;
            - all actions and states of the plan,
            - all produced sub-goal stages,
            - the conformance mapping,
            - plan quality and planning time statistics.
        
        Raises
        ------
        `ValueError` - If no plan exists at the given abstraction level.
        """
        if level not in self.__actions:
            raise ValueError(f"No plan exists at abstraction level {level}.")
        
        ## A plan is refined iff there are sub-goal stages at the previous level and there are achieved sub-goals at the current level
        refined_plan: bool = ((level + 1) in self.__sgoals
                              and level in self.__sgoals_achieved_at)
        
        if (refined_plan
            and start_step != 0
            and start_step not in self.__sgoals_achieved_at[level].values()):
            self.__logger.log(self.__log_level(Verbosity.Minimal, logging.WARNING),
                              f"Extracted plan at non-initial start step {start_step} does not start on a matching child step...\n"
                              f"These are: {list(self.__sgoals_achieved_at[level].values())}")
        
        ## The elements that make up a monolevel plan
        states: dict[int, list[Fluent]] = {}
        actions: dict[int, list[Action]] = {}
        produced_sgoals: dict[int, list[SubGoal]] = {}
        
        ## Elements that make up a conformance mapping between this plan and the abstract plan that it refines
        constraining_sgoals: dict[int, list[SubGoal]] = {}
        current_sgoals: ReversableDict[int, int] = ReversableDict()
        sgoals_achieved_at: ReversableDict[int, int] = ReversableDict()
        sequential_yield_steps: dict[int, int] = {}
        
        ## The current total concatenated plan length is the number of planned actions
        total_plan_length: int = len(self.__actions[level])
        
        ## The plan's step range goes between;
        ##      - The start step (inclusive of states, exclusive of actions),
        ##      - The minimum of; the given end step and total plan length (inclusive of both states and actions).
        step_range = range(start_step,
                           (min(end_step + 1, total_plan_length + 1)
                           if end_step is not None else
                           total_plan_length + 1))
        
        self.__logger.debug(f"Extracting monolevel plan at level {level}:\n"
                            f"Current total plan length = {total_plan_length}, chosen step range to extract = {step_range}")
        self.__logger.debug(f"Current concatenated monolevel plan progression:\n"
                            + "\n".join(f"Level [{level}]: "
                                        f"Length = {len(self.__actions.get(level, {}))}, "
                                        f"Total actions = {sum(len(actions) for actions in self.__actions.get(level, {}).values())}, "
                                        f"Produced sub-goal stages = {len(self.__sgoals.get(level, {}))}, "
                                        f"Produced sub-goal literals = {sum(len(sgoals) for sgoals in self.__sgoals.get(level, {}).values())}"
                                        for level in reversed(self.level_range)))
        
        trailing_plan: bool = False
        
        for step in step_range:
            states[step] = self.__states[level][step]
            
            ## The start step does not include any actions
            if step != start_step:
                actions[step] = self.__actions[level][step]
                
                ## Get the produced sub-goal stages if this plan is abstract (non-ground)
                if level != 1:
                    produced_sgoals[step] = self.__sgoals[level][step]
                
                if (refined_plan
                    and not trailing_plan):
                    ## Get the sub-goal stage index that is current of the given step;
                    ##      - If there is no current sub-goal stage index then the given step is inside the trailing plan,
                    ##      - Inside a trailing plan there is no conformance mapping to the constraining sub-goal stages produced from the abstract plan at the previous level.
                    index: int = self.__current_sgoals[level].get(step, -1)
                    if index == -1:
                        trailing_plan = True
                        continue
                    
                    ## Update the conformance mapping for the given step-index pair;
                    ##      - The constraining sub-goal at the current index defines the head of a refinement tree,
                    ##      - The current sub-goal stage function maps each sub-goal stage index to the steps at which that stage was current in its refinement,
                    ##          - This links the tree's head action and produced sub-goal stage to its refined sub-plan, containing all its descending child actions (of which there may be many),
                    ##      - The sub-goal stage achievement function maps each sub-goal stage index to the step at which that stage was uniquely achieved in its refinement,
                    ##          - This links the tree's head action and produced sub-goal stage to its matching child action (at the end of its refined sub-plan),
                    ##      - The sub-goal stage sequential yield achievement maps each sub-goal stage index to the step at which that stage was minimally uniquely achieved during sequential yield refinement planning mode.
                    constraining_sgoals[index] = self.__sgoals[level + 1][index]
                    current_sgoals[step] = index
                    sgoals_achieved_at[index] = self.__sgoals_achieved_at[level][index]
                    if (yield_steps := self.__sequential_yield_steps.get(level, None)) is not None:
                        sequential_yield_steps[index] = yield_steps[index]
        
        conformance_mapping: Optional[ConformanceMapping] = None
        if refined_plan:
            _sequential_yield_steps: Optional[dict[int, int]] = sequential_yield_steps
            if not _sequential_yield_steps: _sequential_yield_steps = None
            conformance_mapping = ConformanceMapping(constraining_sgoals, current_sgoals, sgoals_achieved_at, _sequential_yield_steps)
        is_final: bool = self.__complete_plan.get(level, False) and (step_range.stop - 1) == total_plan_length
        return MonolevelPlan(level, self.__domain.get_model_type(level), states, actions, produced_sgoals, is_final, self.__statistics[level], conformance_mapping)
    
    def get_hierarchical_plan(self, bottom_level: int = 1, top_level: Optional[int] = None, ignore_missing: bool = False) -> HierarchicalPlan:
        """
        Get the currently stored hierarchical plan over a given abstraction level range.
        The plan will include all levels in the given range that are in the currently loaded planning domain and have been generated.
        
        Parameters
        ----------
        `bottom_level: int = 1` - The bottom level of the abstraction level range to include in the plan.
        
        `top_level: Optional[int] = None` - The top level of the abstraction level range to include in the plan.
        If None, then all levels above the bottom level are included.
        
        `ignore_missing: bool = False` - Whether to ignore levels in the given range at which a monolevel plan does not yet exist.
        If False, then an error will be raised if no plan exists at some level in the range.
        
        Returns
        -------
        `HierarchicalPlan` - The resulting hierarchical plan containing for each level;
            - a concatenated monolevel plan (always initial and combined, possibly final/complete),
            - iff the hierarchical plan was generated by the hierarchical planning algorithm,
                - a sequence of (possibly) partial monolevel plans,
                - a sequence of division scenarios (representing a level of the problem division tree).
        """
        concatenated_plans: dict[int, MonolevelPlan] = {}
        partial_plans: dict[int, list[MonolevelPlan]] = {}
        problem_division_tree: dict[int, list[DivisionScenario]] = {}
        
        for level in self.constrained_level_range(bottom_level, top_level):
            if (ignore_missing
                and level not in self.__actions):
                continue
            
            concatenated_plans[level] = self.get_monolevel_plan(level)
            if level in self.__partial_plans:
                partial_plans[level] = self.__partial_plans[level]
            if level in self.__division_scenarios:
                problem_division_tree[level] = self.__division_scenarios[level]
        
        return HierarchicalPlan(concatenated_plans, partial_plans, problem_division_tree)
    
    def total_plan_length(self, level: int) -> int:
        """
        Get the total length of the currently stored executable concatenated monolevel plan at a given abstraction level.
        
        Parameters
        ----------
        `level : int` - An integer defining the level at which to get the plan length.
        
        Returns
        -------
        `int` - An integer defining the plan length at the given level.
        """
        if not isinstance(level, int):
            raise TypeError(f"Abstraction level must be an integer. Got; {level} or type {type(level)}.")
        return len(self.__actions.get(level, {}))
    
    def total_produced_sgoals(self, level: int) -> int:
        """
        Get the total number sub-goal stages that have been produced from a given abstraction level.
        
        Parameters
        ----------
        `level : int` - An integer defining the level at which to get the number of sub-goal stages.
        
        Returns
        -------
        `int` - An integer defining the number of sub-goal stages that have been produced from the given level.
        """
        if not isinstance(level, int):
            raise TypeError(f"Abstraction level must be an integer. Got; {level} or type {type(level)}.")
        return len(self.__sgoals.get(level, {}))
    
    def total_achieved_sgoals(self, level: int) -> int:
        """
        Get the total number of sub-goal stages achieved by abstract plan conformance refinement at a given abstraction level, that were produced from planning at the previous abstract level.
        This is equivalent to the current progression of conformance refinement planning at the given level.
        
        Parameters
        ----------
        `level : int` - An integer defining the level at which to get the number of achieved sub-goal stages.
        
        Returns
        -------
        `int` - An integer defing the number of sub-goal stages achieved by plan refinement at the given level.
        """
        if not isinstance(level, int):
            raise TypeError(f"Abstraction level must be an integer. Got; {level} or type {type(level)}.")
        if level in self.__sgoals_achieved_at:
            return max(self.__sgoals_achieved_at[level])
        return 0
    
    def get_current_division_scenario(self, level: int) -> Optional[DivisionScenario]:
        """
        Get the current division scenario at a given abstraction level.
        
        The current division scenario is the earliest in sequence scenario that is unsolved.
        Where a division scenario is solved iff all of its templated (partial) conformance refinement planning problems are solved, otherwise it is unsolved.
        All the scenario's partial problems are solved iff all of its sub-goal stages, produced from the abstract plan that it divides, have been refined (achieved at the next level).
        
        Parameters
        ----------
        `level : int` - An integer defining the level at which to get the division scenario for.
        
        Returns
        -------
        `{DivisionScenario | None}` - The current division scenario, or None if either; there are no scenarios at the given level or, all scenarios at the given level have been solved.
        """
        if not isinstance(level, int):
            raise TypeError(f"Abstraction level must be an integer. Got; {level} or type {type(level)}.")
        if level not in self.level_range:
            raise ASH_InvalidInputError(f"The abstraction level {level} is not in the range of {self}.")
        
        ## Iterate through the division scenarios at the given level;
        ##      - Return the first one whose last sub-goal stage index has not yet been refined.
        for scenario in self.__division_scenarios.get(level, []):
            if scenario.last_index > self.total_achieved_sgoals(level - 1):
                return scenario
        
        ## Return None iff there are no current division scenarios
        return None
    
    def get_valid_planning_level(self, highest: bool, bottom_level: int = 1, top_level: Optional[int] = None) -> Optional[int]:
        """
        Determine the highest or lowest valid abstraction level that can currently be planned at.
        
        A level can be planned at if both;
                - It is incomplete such that the final-goal has not been achieved,
                - It is the top-level or there are unachieved sub-goal stages at the previous level.
        
        Parameters
        ----------
        `highest : bool` - Whether to search for the highest or lowest planning level.
        True to search for the highest planning level, False to search for the lowest.
        
        `bottom_level : int = 1` - TODO
        
        `top_level : {int | None} = None` - TODO
        
        Returns
        -------
        `{int | None}` - The valid planning level, or None if no valid planning level exists in the given range.
        No valid level exists iff all levels in the range are complete.
        """
        level_range: Iterable[int] = self.constrained_level_range(bottom_level, top_level)
        _top_level: int = max(level_range)
        
        ## Iterate over the hierarchy's levels in;
        ##      - Descending order if searching for the higheset valid level,
        ##      - Otherwise, ascending ordering to search for the lowest valid level.
        if highest: level_range = reversed(level_range)
        
        ## Search the hierarchy for abstraction levels that are valid to plan at;
        ##      - Easiest way to tell if a level can be planned at is if either;
        ##          - It is the top-level and it is not complete,
        ##          - There are unachieved sub-goal stages at the previous level.
        for level in level_range:
            if ((level == _top_level and not self.__complete_plan.get(level, False))
                or (len(self.__sgoals.get(level + 1, {})) > len(self.__sgoals_achieved_at.get(level, {})))):
                return level
        
        ## Return None if there are no valid planning levels;
        ##      - This should be true iff all abstraction levels in the range have been completed.
        if not all(self.__complete_plan.get(level, False) for level in level_range):
            raise ASH_InternalError(f"Found no valid planning levels in range {level_range} that are not all complete:\n"
                                    + "\n".join(f"Level [{level}]: Is final = {self.__complete_plan.get(level, False)}" for level in level_range))
        return None
    
    ################################################################################
    #### Problem specifics
    ################################################################################
    
    def get_initial_state(self, level: Union[int, None]) -> Optional[Union[list[Fluent], dict[int, list[Fluent]]]]:
        """
        Get the initial state of the currently loaded planning problem.
        If no planning problem is loaded, or the initial state has not be generated, this method returns None.
        
        Parameters
        ----------
        `level : {int | None}` - The abstraction level to get the initial state for, or None to get the initial state at all levels.
        
        Returns
        -------
        `{list[Fluent], dict[int, list[Fluent]] | None}` - Either, a list of fluent literals defining the complete valid initial state for the given level,
        a dictionary mapping all abstraction levels to their respective initial state, or None if no planning problem is loaded or the initial state has not be generated.
        """
        if not self.in_range(level):
            raise ValueError(f"Abstraction level {level} is not in the range of {self}.")
        if level is not None:
            return self.__initial_states.get(level, None)
        return self.__initial_states.copy()
    
    def get_final_goal(self, level: int) -> Optional[list[FinalGoal]]:
        """
        Get the final-goal of the currently loaded planning problem.
        If no planning problem is loaded, or the final-goal has not be generated, this method returns None.
        
        Parameters
        ----------
        `level : {int | None}` - The abstraction level to get the final-goal for, or None to get the final-goal at all levels.
        
        Returns
        -------
        `{list[FinalGoal], dict[int, list[FinalGoal]] | None}` - Either, a list of fluent literals defining the complete valid final-goal for the given level,
        a dictionary mapping all abstraction levels to their respective final-goal, or None if no planning problem is loaded or the final-goal has not be generated.
        """
        if not self.in_range(level):
            raise ValueError(f"Abstraction level {level} is not in the range of {self}.")
        if level is not None:
            return self.__final_goals.get(level, None)
        return self.__final_goals.copy()
    
    def initialise_problem(self, find_inconsistencies: bool = False) -> None:
        """Initialise the currently loaded planning problem."""
        
        self.__logger.log(self.__log_level(Verbosity.Simple),
                          "Attempting to initialise hierarchical planning problem...")
        
        init_valid, init_unique, init_consistency = self.__generate_initial_states(find_inconsistencies)
        goal_valid, goal_unique, goal_consistency = self.__generate_final_goals(find_inconsistencies)
        
        def fvalid(valid: bool) -> str:
            return "VALID" if valid else "INVALID"
        def funique(unique: bool) -> str:
            return "UNIQUE" if unique else "NON-UNIQUE"
        
        if (valid := init_valid and goal_valid):
            self.__logger.log(self.__log_level(Verbosity.Simple),
                              "Hierarchical planning problem successfully initialised:\n"
                              f"Initial State is {funique(init_unique)}, Final-Goal is {funique(goal_unique)}")
        else: raise ASH_NoSolutionError("Hierarchical planning problem failed to initialise:\n"
                                        f"Initial State is {fvalid(init_valid)}, Final-Goal is {fvalid(goal_valid)}")
        
        if init_unique and goal_unique:
            self.__logger.log(self.__log_level(Verbosity.Verbose),
                              "The given problem specification has a unique interpretation (exactly one stable model exists).")
            if find_inconsistencies:
                self.__logger.log(self.__log_level(Verbosity.Minimal),
                                  "No inconsistencies were found in the problem specification.")
            
        else:
            if not find_inconsistencies:
                self.__logger.log(self.__log_level(Verbosity.Minimal),
                                  "The given problem specification has multiple possible interpretation (more than one stable models exist).\n"
                                  "A check for inconsistencies is advised (use CLI argument --operation=find-problem-inconsistencies).")
            
            else:
                variable_mappings: dict = {}
                
                for name, consistency in [("initial state", init_consistency),
                                          ("final-goal", goal_consistency)]:
                    
                    variable_mappings[name] = {"consistent" : ASP.Atom.nest_grouped_atoms(consistency.consistent, as_standard_dict=True),
                                               "inconsistent" : ASP.Atom.nest_grouped_atoms(consistency.inconsistent, as_standard_dict=True)}
                    
                    self.__logger.log(self.__log_level(Verbosity.Minimal),
                                      f"Inconsistencies found :: There are {len(consistency.consistent)} consistent and {len(consistency.inconsistent)} inconsistent variables.\n"
                                      f"Note that inconsistent defined final-goal literals indicate inconsistencies in the defining fluents, or that the defining fluents are not goal fluents.\n"
                                      f"The following fluent state variables can take multiple possible values in the {name}:\n\n"
                                      + json.dumps(variable_mappings[name]["inconsistent"].copy(), indent=4))
                    
                    if isinstance(find_inconsistencies, str):
                        with open(find_inconsistencies, 'w') as file_writer:
                            file_writer.write(json.dumps(variable_mappings, indent=4))
    
    def __generate_initial_states(self, find_inconsistencies: Union[bool, str] = False) -> tuple[bool, bool, ConsistencyCheck]:
        """Generate the problem's initial states."""
        
        self.__logger.log(self.__verbosity.value.log, "Generating initial states...")
        
        ## Create a local copy of the logic program
        local_program: ASP.LogicProgram = self.__domain_logic_program.copy(rename=f"{self.__name} || Generate Initial States")
        
        ## Solve in initial state mode;
        ##      - Find the union of all models to find inconsistencies (variables with multiple literals),
        ##      - Otherwise simply generate up to two models only the last of which is used as the intial state for the problem,
        ##          - However, if two are found it means more than one interpretation of the initial state exists and the problem specification is inconsistent.
        answer: ASP.Answer = local_program.solve(solver_options=[ASP.Options.models(0 if find_inconsistencies else 2),
                                                                 ASP.Options.enumeration(ASP.Options.EnumerationMode.Union
                                                                                         if find_inconsistencies else
                                                                                         ASP.Options.EnumerationMode.Auto),
                                                                 ASP.Options.threads(self.__threads),
                                                                 ASP.Options.warn(self.__verbosity == Verbosity.Verbose)],
                                                 base_parts=[## ASH Modules for state representation at initial step only
                                                             ASP.BasePart("abstraction_levels", [1, "hierarchical"]),
                                                             ASP.BasePart("instance_module", []),
                                                             ASP.BasePart("state_module", [0, 0]),
                                                             
                                                             ## Problem specifics for initial state
                                                             ASP.BasePart("entities", []),
                                                             ASP.BasePart("static_state", []),
                                                             ASP.BasePart("initial_state", []),
                                                             
                                                             ## Domain definitions elements relevent to initial state
                                                             ASP.BasePart("domain_sorts", []),
                                                             ASP.BasePart("variable_relations", [0]),
                                                             ASP.BasePart("abstraction_mappings", [0])])
        
        self.__logger.log(self.__verbosity.value.log,
                          "Initial states {0}:\n{1!s}".format('generated successfully'
                                                              if answer.result.satisfiable else
                                                              'failed to generate',
                                                              answer))
        
        is_valid: bool = False
        is_unique: bool = False
        consistency: Optional[ConsistencyCheck] = None
        
        if (is_valid := answer.result.satisfiable):
            if not find_inconsistencies:
                self.__initial_states = answer.fmodel.query(Fluent, group_by='L')
                
                for level in reversed(self.level_range):
                    header: str = center_text(f"Initial state at abstraction level {level}", framing_width=48, centering_width=60)
                    self.__logger.log(self.__log_level(Verbosity.Standard),
                                      f"\n\n{header}\n\n" + "\n".join(map(str, self.__initial_states[level])))
                
                is_unique = len(answer.base_models) == 1
            
            else:
                ## Obtain the state variable to literal value assignment mappings: 
                ##      - Maps: level X state variable -> list[state literal]
                fluents: StateVariableMapping = answer.fmodel.query(Fluent, group_by=['L', 'F'])
                consistent: StateVariableMapping = {}
                inconsistent: StateVariableMapping = {}
                
                for variable in fluents:
                    if len(fluents[variable]) > 1:
                        inconsistent[variable] = fluents[variable] ## TODO add select
                    else: consistent[variable] = fluents[variable]
                
                consistency = ConsistencyCheck(consistent, inconsistent)
                is_unique = not bool(inconsistent)
        
        return is_valid, is_unique, consistency
    
    def __generate_final_goals(self, find_inconsistencies: bool = False) -> tuple[bool, bool, ConsistencyCheck]:
        """Generate the problem's final-goals."""
        
        self.__logger.log(self.__verbosity.value.log, "Generating final-goals...")
        
        ## Create a local copy of the logic program
        local_program: ASP.LogicProgram = self.__domain_logic_program.copy(rename=f"{self.__name} || Generate Final Goals")
        
        ## Solve in goal abstraction mode
        answer: ASP.Answer = local_program.solve(solver_options=[ASP.Options.models(0 if find_inconsistencies else 2),
                                                                 ASP.Options.enumeration(ASP.Options.EnumerationMode.Union
                                                                                         if find_inconsistencies else
                                                                                         ASP.Options.EnumerationMode.Auto),
                                                                 ASP.Options.threads(self.__threads),
                                                                 ASP.Options.warn(self.__verbosity == Verbosity.Verbose)],
                                                 base_parts=[## ASH Modules for final-goal representation as state on initial step only
                                                             ASP.BasePart("abstraction_levels", [1, "hierarchical"]),
                                                             ASP.BasePart("instance_module", []),
                                                             ASP.BasePart("goal_abstraction_module", []),
                                                             
                                                             ## Problem specifics for final-goal
                                                             ASP.BasePart("entities", []),
                                                             ASP.BasePart("static_state", []),
                                                             ASP.BasePart("goal_state", []),
                                                             
                                                             ## Domain definitions elements relevent to final-goal
                                                             ASP.BasePart("domain_sorts", []),
                                                             ASP.BasePart("variable_relations", [0]),
                                                             ASP.BasePart("abstraction_mappings", [0])])
        
        self.__logger.log(self.__verbosity.value.log,
                          "Final-goals {0}:\n{1!s}".format('generated successfully'
                                                           if answer.result.satisfiable else
                                                           'failed to generate',
                                                           answer))
        
        is_valid: bool = False
        is_unique: bool = False
        consistency: Optional[ConsistencyCheck] = None
        
        if (is_valid := answer.result.satisfiable):
            if not find_inconsistencies:
                self.__final_goals = answer.fmodel.query(FinalGoal, group_by='L')
                
                for level in reversed(self.level_range):
                    header: str = f"Final-goals at abstraction level {level}"
                    pos_header: str = center_text("Positive " + header, framing_width=48, centering_width=60)
                    self.__logger.log(self.__log_level(Verbosity.Standard),
                                      f"\n\n{pos_header}\n\n" + "\n".join(str(literal) for literal in self.__final_goals[level]
                                                                          if literal['T'] == "true"))
                    neg_header: str = center_text("Negative " + header, framing_width=48, centering_width=60)
                    self.__logger.log(self.__log_level(Verbosity.Verbose),
                                      f"\n\n{neg_header}\n\n" + "\n".join(str(literal) for literal in self.__final_goals[level]
                                                                          if literal['T'] == "false"))
                
                is_unique = len(answer.base_models) == 1
            
            else:
                ## Obtain the state variable to positive final-goal literal value assignment mappings: 
                ##      - Maps: level X state variable -> list[final-goal literal]
                fluents: StateVariableMapping = answer.fmodel.query(FinalGoal, group_by=['L', 'F'], param_constrs={'T' : "true"})
                consistent: StateVariableMapping = {}
                inconsistent: StateVariableMapping = {}
                
                for variable in fluents:
                    if len(fluents[variable]) > 1:
                        inconsistent[variable] = fluents[variable]
                    else: consistent[variable] = fluents[variable]
                
                consistency = ConsistencyCheck(consistent, inconsistent)
                is_unique = not bool(inconsistent)
        
        return is_valid, is_unique, consistency
    
    ################################################################################
    #### Plan generation
    ################################################################################
    
    def __create_problem(self,
                         level: int,
                         concurrency: bool,
                         conformance: bool,
                         conformance_type: Optional[ConformanceType],
                         first_sgoals: Optional[int],
                         last_sgoals: Optional[int],
                         sequential_yield: bool,
                         division_strategy: Optional[DivisionStrategy],
                         use_search_length_bound: bool
                         ) -> MonolevelProblem:
        """
        Processes inputs to the monolevel planning algorithm and generates a monolevel planning problem specification.
        
        Internal use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `level: int` - The abstraction level of the problem.
        
        `concurrency: bool` - Whether action concurrency should be enabled, if disabled actions can only be planned sequentially.
        
        `conformance: bool` - Whether to apply conformance constraints, enabling conformance refinement planning mode.
        
        `conformance_type: ConformanceType` - The conformance constraint application type; simultaneous or sequential sub-goal stage achievement.
        
        `first_sgoals: int` - The requested first in sequence sub-goal stage index value (inclusive) of a conformance refinement planning problem.
        
        `last_sgoals: int`- The requested last in sequence sub-goal stage index value (inclusive) of a conformance refinement planning problem.
        
        `sequential_yield: bool` - Whether the conformance refinement planning problem should be solved in sequential yield mode.
        
        `division_strategy : {DivisionStrategy | None}` - The division strategy being used to divide the planning problem.
        
        `use_search_length_bound: bool` - Whether to use the minimum search length bound.
        
        Returns
        -------
        `MonolevelProblem` - A specification for the monolevel planning problem.
        
        Raises
        ------
        `ASH_InvalidPlannerState` - Iff conformance is requested but no sub-goal stages exist at the previous level.
        """
        
        ## It is always possible to enable concurrency;
        ##      - True classical planning however is only sequential.
        _concurrency: bool = concurrency
        
        ## Conformance is enabled iff;
        ##      - Conformance is explicitly enforced and,
        ##      - The planning level is not the top-level.
        _conformance: bool = level != self.top_level and conformance
        
        ## Ensure that subgoals exist at the previous level if conformance is enabled
        if _conformance and not self.total_produced_sgoals(level + 1):
            log_and_raise(ASH_InvalidPlannerState,
                          f"Cannot enforce conformance of plan at level {level} with previous level {level + 1}, as no subgoals stages exist at that level.",
                          logger=self.__logger)
        
        ## Determine the conformance type;
        ##      - If not given or None then;
        ##          - Sequential achievement is used if concurrency is enabled to ensure solution existance,
        ##          - Otherwise, simultaneous is used to provide a stronger constraint.
        _conformance_type: Optional[ConformanceType] = conformance_type
        if _conformance and conformance_type is None:
            _conformance_type = (ConformanceType.SequentialAchievement if concurrency
                                 else ConformanceType.SimultaneousAchievement)
        
        ## Find the limits of the subgoal ranges that exist at the previous level
        ##      - Maximum number of subgoal stages at previous level
        ##      - Number of achieved subgoal stages
        all_sgoals: int = self.total_produced_sgoals(level + 1)
        achieved_sgoals: int = self.total_achieved_sgoals(level)
        
        ## Find the sub-goal stage range to refine;
        ##      - If the number of sub-goal stages was not given or None then;
        ##           - add all the sub-goals from the previous level that have not been achieved,
        ##      - Otherwise only add sub-goals whose sequence indices fall between than the given range.
        ##          - The first sub-goal stage is;
        ##              - the maximum of 1 and,
        ##                  - the minimum of the total number of sub-goal stages that have been achieved at the given level and,
        ##                  - the given first sub-goal stage index value.
        ##          - The last sub-goal stage is;
        ##              - the maximum of the first and,
        ##                  - the minimum of the total number of sub-goal stages that have currently been produced from the previous level and,
        ##                  - the given last sub-goal stage index value.
        _first_sgoals: int = achieved_sgoals + 1
        if first_sgoals is not None:
            _first_sgoals = max(1, min(first_sgoals, _first_sgoals))
            if _first_sgoals < first_sgoals:
                self.__logger.log(self.__log_level(Verbosity.Minimal, logging.WARNING),
                                  f"The first requested sub-goal stage index value {first_sgoals} was larger than the maximum possible value of {_first_sgoals} "
                                  "(the sub-goal index immediately following that which was last achieved at this level) and was revised to this value.")
            if _first_sgoals > first_sgoals:
                self.__logger.log(self.__log_level(Verbosity.Minimal, logging.WARNING),
                                  f"First sub-goal stage index value {first_sgoals} was smaller than 1 and revised to {_first_sgoals}.")
        _last_sgoals: int = max(1, all_sgoals)
        if last_sgoals is not None:
            _last_sgoals = max(_first_sgoals, min(last_sgoals, _last_sgoals))
            if _last_sgoals < last_sgoals:
                self.__logger.log(self.__log_level(Verbosity.Minimal, logging.WARNING),
                                  f"The last requested sub-goal stage index value {last_sgoals} was larger than the maximum possible value of {_last_sgoals} "
                                  "(the total number of sub-goals that have currently been produced from the previous level) and was revised to this value.")
            if _last_sgoals > last_sgoals:
                self.__logger.log(self.__log_level(Verbosity.Minimal, logging.WARNING),
                                  f"Last sub-goal stage index value {last_sgoals} was smaller than the first and was revised to {_last_sgoals}.")
        
        ## Find the step to start planning from;
        ##      - Assume no revision of the existing plan and extend from its end step,
        ##          - The end step is the achievement step of the last achieved sub-goal stage.
        ##      - If any of the existing plan is to be revised (by blending) then;
        ##          - Start from the step at which the sub-goal stage that preceeds the first in the current sequence was achieved,
        ##          - This is because that state is need as the start state even though the first sub-goal stage is not current until the following state.
        ##      - Note that if either;
        ##          - A preemptive reactive division was made on a saved grounding,
        ##          - A left blend was made over a saved grounding,
        ##              - The current search length of that grounding that has been reached so far, may be greater than the start step of the current problem,
        ##              - This is valid because the plan cannot possibly get shorter than the currently stored search length.
        _start_step: int = self.total_plan_length(level)
        if _first_sgoals <= achieved_sgoals:
            _start_step = self.__sgoals_achieved_at[level].get(_first_sgoals - 1, 0)
        
        ## The problem is initial iff the start step is the initial step
        _is_initial: bool = _start_step == 0
        
        ## The problem is final iff either;
        ##      - Conformance is not enabled such that complete classical plan must be generated,
        ##      - Conformance is enabled and both;
        ##          - The plan at the previous level is complete,
        ##          - The problem includes all the last sub-goal stage from the previous level;
        ##              - This is thus the final sub-goal stage of the previous level and thus makes the refinement problem at the current level final.
        _is_final: bool = (not _conformance
                           or (self.__complete_plan.get(level + 1, False)
                               and _last_sgoals == all_sgoals))
        
        ## The planning mode is;
        ##      - Complete iff the problem is both initial and final,
        ##      - Partial otherwise.
        _complete_planning: bool = _is_initial and _is_final
        
        ## Sequential yield iff conformance is enabled
        _sequential_yield: bool = _conformance and sequential_yield
        
        ## Reactive divisions can be made only iff;
        ##      - Sequential yield planning mode is enabled,
        ##      - The division strategy was given and not None,
        ##      - There is a valid (reactive) bound at the given level
        _reactive_divisions: bool = (_sequential_yield
                                     and division_strategy is not None
                                     and any(division_strategy.get_bound(bound, level, -1) != -1 for bound in division_strategy.bounds))
        
        ## Determine the minimum step bound for the plan length at the current planning level;
        ##      - The step bound is zero if conformance is disabled,
        ##      - If conformance is enabled;
        ##          - If partial planning;
        ##              - The step bound is the maximum of;
        ##                  - The length of the existing plan that is not being revised (by blending) plus the number of sub-goal stages being refined in the current problem,
        ##                  - The length of the existing plan, since blending cannot possibly make the child sub-plans of the revised sub-goal stages any shorter.
        ##          - If complete or final-partial planning, the step bound is the maximum of the prior and the length of the plan at the previous level.
        _search_length_bound: int = 0
        if _conformance:
            _search_length_bound = max(self.total_plan_length(level),
                                       _start_step + (_last_sgoals - _first_sgoals))
            if _is_final:
                _search_length_bound = max(_search_length_bound,
                                           self.total_plan_length(level + 1))
        _use_search_length_bound: bool = (use_search_length_bound
                                          and _search_length_bound > _start_step
                                          and not _sequential_yield)
        
        return MonolevelProblem(## Basic parameters
                                level, _concurrency,
                                ## Planning mode and refinement parameters
                                _conformance, _conformance_type, _first_sgoals, _last_sgoals,
                                ## Problem type parameters
                                _start_step, _is_initial, _is_final, _complete_planning,
                                ## Optional parameters
                                _sequential_yield, _reactive_divisions, _use_search_length_bound, _search_length_bound)
    
    def monolevel_plan(
                       ## Required arguments
                       self,
                       level: int,
                       concurrency: bool,
                       conformance: bool,
                       /,
                       
                       ## Common arguments
                       conformance_type: Optional[ConformanceType] = None,
                       first_sgoals: Optional[int] = None,
                       last_sgoals: Optional[int] = None, ## TODO Should take last_sgoals_right_blend as well, so it knows about the blend and discard it before returning.
                       sequential_yield: bool = False,
                       division_strategy: Optional[DivisionStrategy] = None,
                       save_grounding: bool = False,
                       use_saved_grounding: Optional[bool] = None,
                       use_search_length_bound: bool = True,
                       make_observable: bool = False,
                       *,
                       
                       ## Optimisation options
                       minimise_actions: Optional[bool] = None,
                       order_fgoals_achievement: bool = False,
                       preempt_pos_fgoals: Optional[bool] = None,
                       preempt_neg_fgoals: Optional[bool] = False,
                       preempt_mode: PreemptMode = PreemptMode.Heuristic,
                       
                       ## Performance analysis options
                       detect_interleaving: bool = False,
                       generate_search_space: bool = False,
                       generate_solution_space: bool = False,
                       
                       ## Search limits
                       time_limit: Optional[int] = None,
                       length_limit: Optional[int] = None
                       
                       ## Return the monolevel plan
                       ) -> MonolevelPlan:
        """
        Generate a monolevel plan that solves a given monolevel classical or conformance refinement planning problem.
        
        Parameters
        ----------
        `level: int` - The abstraction level of the problem.
        
        `concurrency: bool` - Whether action concurrency should be enabled, if disabled actions can only be planned sequentially.
        
        `conformance: bool` - Whether to apply conformance constraints, enabling conformance refinement planning mode.
        
        `conformance_type: ConformanceType` - The conformance constraint application type; simultaneous or sequential sub-goal stage achievement.
        
        `first_sgoals: int` - The requested first in sequence sub-goal stage index value (inclusive) of a conformance refinement planning problem.
        
        `last_sgoals: int`- The requested last in sequence sub-goal stage index value (inclusive) of a conformance refinement planning problem.
        
        `sequential_yield: bool` - Whether the conformance refinement planning problem should be solved in sequential yield mode.
        
        `division_strategy : {DivisionStrategy | None}` - The division strategy being used to divide the planning problem.
        
        `save_grounding : bool` - Whether the logic program's grounding will be saved.
        
        `use_search_length_bound: bool` - Whether to use the minimum search length bound.
        
        `make_observable: bool = False` - Deprecated, to be removed.
        
        `minimise_actions : bool` - Whether to enable the action minimisation statement.
        
        `order_fgoal_achievement : bool` - Whether to enable the final-goal intermediate achievement ordering preference optimisation statement.
        
        `preempt_pos_fgoals : bool` - Whether to enable positive final-goal preemptive achievement.
        
        `preempt_neg_fgoals : bool` - Whether to enable negative final-goal preemptive achievement.
        
        `preempt_mode : PreemptMode` - The mode by which the final-goal preemptive achievement is applied.
        
        `detect_interleaving : bool = False` - Deprecated, to be removed.
        
        `generate_search_space : bool = False` - Deprecated, to be removed.
        
        `generate_solution_space : bool = False` - Deprecated, to be removed.
        
        `time_limit: {int | None} = None` - The search time limit in seconds, if reached cancels search.
        
        `length_limit: {int | None} = None` - The search length limit, if reached cancels search.
        """
        
        ## Log the function call and input arguments for debugging purposes
        self.__logger.debug("Starting monolevel planning with input arguments\n\t"
                            + "\n\t".join(map(str, locals().items())))
        
        #############################################################################
        #### Check Planner State
        #############################################################################
        
        ## Check the planner has an initial state and final goal pre-generated
        if not self.__initial_states:
            log_and_raise(ASH_InvalidPlannerState,
                          "Cannot generate a plan, no initial state exists.",
                          logger=self.__logger)
        if not self.__final_goals:
            log_and_raise(ASH_InvalidPlannerState,
                          "Cannot generate a plan, no final goal exists.",
                          logger=self.__logger)
        
        #############################################################################
        #### Input Validity Checking
        #############################################################################
        
        ## Check planning level is valid
        if not isinstance(level, int):
            log_and_raise(ASH_InvalidInputError,
                          f"Planning level must be an integer. Got {level} of type {type(level)}.",
                          logger=self.__logger)
        if not self.__domain.in_range(level):
            tl: int = self.__domain.top_level
            log_and_raise(ASH_InvalidInputError,
                          f"Planning level of {level} is invalid. "
                          f"The level must be an integer in the range; [1-{tl}] : {tl} = top_level.",
                          logger=self.__logger)
        
        ## Check that the subgoal stage range to refine is valid
        if first_sgoals is not None and last_sgoals is not None and last_sgoals < first_sgoals:
            self.__logger.warn(f"The maximum value of the refined subgoal stage range {last_sgoals}, "
                               f"is less than the minimum value {first_sgoals}.")
        
        #############################################################################
        #### Obtain the problem specification
        #############################################################################
        
        problem: MonolevelProblem = self.__create_problem(level, concurrency,
                                                          conformance, conformance_type, first_sgoals, last_sgoals,
                                                          sequential_yield, division_strategy, use_search_length_bound)
        
        self.__logger.log(self.__log_level(Verbosity.Standard, logging.INFO),
                          f"Generating monolevel plan:\n{problem.problem_description}")
        self.__logger.log(self.__log_level(Verbosity.Verbose, logging.INFO),
                          f"Problem specification obtained: {problem!s}")
        
        ## Warn if the given step limit is less than the determined min search length bound
        if (problem.use_search_length_bound
            and length_limit is not None
            and length_limit < problem.search_length_bound):
            self.__logger.log(self.__log_level(Verbosity.Standard, logging.WARNING),
                              f"The specified step limit of {length_limit} is less than the minimum step bound of {problem.search_length_bound}.")
        
        ## Determine if problem spaces are to be generated;
        ##      - These are special modes that don't change the problem specification but there are some facets;
        ##          - Search spaces cannot be generated in sequential yield mode,
        ##          - Solution spaces cannot be generated at the same time as search spaces are as it conflicts with enumerating the optimal models,
        ##          - The goal-wise search spaces are obtained simply by checking the conformance mapping and taking the problem spaces on matching children,
        ##          - Step-wise solutions spaces are not supported.
        _generate_search_space: bool = generate_search_space and not problem.sequential_yield
        _generate_solution_space: bool = generate_solution_space and not _generate_search_space
        _generate_problem_space: bool = _generate_search_space or _generate_solution_space
        
        #############################################################################
        #### Obtain the optimisation details
        #############################################################################
        
        ## Action minimisation is enabled iff;
        ##      - Search space generation is disabled and either;
        ##          - Explicitly enabled or,
        ##          - Not given or None and concurrency is enabled.
        _minimise_actions: bool = ((bool(minimise_actions)
                                    or (minimise_actions is None
                                        and problem.concurrency))
                                   and not _generate_search_space)
        
        ## Final-goal intermediate achievement ordering preferences are enabled iff explicitly enabled
        _order_fgoals_achievement: bool = (bool(order_fgoals_achievement)
                                           and self.__domain.get_model_type(level) == DomainModelType.Tasking)
        
        ## Partial-planning is enabled if either;
        ##      - A complete plan is not being generated or,
        ##      - Reactive problem divisions are possible.
        partial_planning_enabled: bool = (not problem.is_final
                                          or problem.reactive_divisions)
        
        ## Preemptive positive final-goal achievement is enabled iff either;
        ##      - Explicitly enabled or,
        ##      - Not given or None and problem divisions may be applied.
        _preempt_pos_fgoals: bool = (partial_planning_enabled
                                     and (bool(preempt_pos_fgoals)
                                          or preempt_pos_fgoals is None))
        
        ## Preemptive negative final-goal achievement is enabled iff explicitly enabled
        _preempt_neg_fgoals: bool = (partial_planning_enabled
                                     and bool(preempt_neg_fgoals))
        
        ## Determine the mode by which the final-goal preemptive achievement is enforced;
        ##      - By default heuristic mode is used if not given or None.
        _preempt_mode: PreemptMode = PreemptMode.Heuristic
        if preempt_mode is not None:
            if not isinstance(preempt_mode, PreemptMode):
                log_and_raise(ASH_InvalidInputError,
                              "Invalid final-goal preemptive achievement mode. "
                              f"Got; {preempt_mode} of type {type(preempt_mode)}.",
                              logger=self.__logger)
            _preempt_mode = preempt_mode
        
        self.__logger.log(self.__log_level(Verbosity.Standard),
                          "Optimisation details:\n"
                          + "\n".join([f"Action minimisation = {_minimise_actions}",
                                       f"Final-goal intermediate achievement ordering preferences = {_order_fgoals_achievement}",
                                       f"Positive final-goal preemptive achievement = {_preempt_pos_fgoals}",
                                       f"Negative final-goal preemptive achievement = {_preempt_neg_fgoals}",
                                       f"Final-goal preemptive achievement mode = {_preempt_mode}"]))
        
        #############################################################################
        #### Logic Program Construction
        #############################################################################
        
        ## Use the saved grounding iff specified and one is available
        _saved_ground_available: bool = level in self.__saved_groundings
        _use_saved_grounding: bool = (use_saved_grounding and _saved_ground_available)
        _save_grounding: bool = (save_grounding and not problem.is_final)
        self.__logger.debug("Grounding Options:\n"
                            f"Continue existing saved grounding: requested = {use_saved_grounding}, available = {_saved_ground_available}, chosen = {_use_saved_grounding}"
                            f"Save current grounding on completion: requested = {save_grounding}, possible = {not problem.is_final}, chosen = {_save_grounding}")
        
        ## Obtain a local copy of the logic program;
        ##      - Either getting the saved monolevel problem logic program grounding,
        ##      - Or creating a new 'clean' copy of the domain logic program.
        local_program: ASP.LogicProgram
        if _use_saved_grounding:
            local_program = self.__saved_groundings[level]
            self.__logger.debug(f"Using saved grounding: {local_program!s}")
        else:
            local_program = self.__domain_logic_program.copy(rename=f"{self.__name} :: Generate monolevel plan")
            self.__logger.debug(f"Using new logic program: {local_program!s}")
        
        ## Assign/update the modifiable total last sub-goal stage index value iff the grounding is being saved or a saved grounding is being contiued.
        ## Using a saved grounding means that we already have the start state and all the base program parts;
        ##      - But we need to tell the planner that we are extending the existing program with more sub-goals,
        ##      - This may also mean that the final-goal may now need to be included too.
        if _save_grounding or _use_saved_grounding:
            self.__total_last_sgoals[level] = problem.last_sgoals
        
        ## Add the start state and final goal iff a saved grounding is not being used;
        ##      - The start state is the initial state only iff the problem is initial such that either;
        ##          - Conformance is not enabled or,
        ##          - The first sub-goal stage is the initial stage.
        ##      - Otherwise, the start state is the state at the start step (which will be non-zero).
        ##      - The final-goal is always added (to allow for final-goal premeptive achievement heuristics), but its acheivement is only enforced if the problem is final.
        if not _use_saved_grounding:
            if not problem.conformance or problem.is_initial:
                self.__logger.debug("Adding initial state as problem start state.")
                local_program.add_rules(self.__initial_states[level] + self.__initial_states.get(level + 1, []),
                                        program_part="ash_initial_state")
            else:
                self.__logger.debug(f"Adding intermediate state at step {problem.start_step} as problem start state.")
                local_program.add_rules(self.__states[level][problem.start_step],
                                        program_part="ash_initial_state")
            local_program.add_rules(self.__final_goals.get(level, []) + self.__final_goals.get(level + 1, []),
                                    program_part="ash_goal_state")
            
            ## Create a list of necessary program parts
            program_parts: ASP.ProgramParts = problem.create_program_parts(_save_grounding,
                                                                           _minimise_actions,
                                                                           _order_fgoals_achievement,
                                                                           _preempt_pos_fgoals,
                                                                           _preempt_neg_fgoals,
                                                                           _preempt_mode)
        
        _optimise: bool = (_minimise_actions
                           or _order_fgoals_achievement
                           or (_preempt_mode is PreemptMode.Optimise
                               and (_preempt_pos_fgoals
                                    or _preempt_neg_fgoals)))
        
        ## Determine solver options
        solver_options = [ASP.Options.program_heuristics(),
                          ASP.Options.statistics(),
                          ASP.Options.threads(self.__threads),
                          ASP.Options.warn(not self.__silence_clingo),
                          ASP.Options.optimise(ASP.Options.OptimiseMode.EmunerateOptimal
                                               if _generate_problem_space else
                                               (ASP.Options.OptimiseMode.FindOptimum
                                                if _optimise else
                                                ASP.Options.OptimiseMode.Ignore))]
        if _generate_problem_space:
            solver_options.append(ASP.Options.models(0))
        if problem.use_search_length_bound:
            solver_options.extend(["-c", f"minimum_search_length_bound={problem.search_length_bound}"])
        self.__logger.debug(f"Solver options determined:\n{solver_options}")
        
        ## Add subgoal stages to the program if conformance is enabled
        if problem.conformance:
            _sgoals: dict[int, list[SubGoal]] = self.__sgoals[level + 1]
            
            ## Obtain the sub-goal stage sequence range to add to the program;
            ##      - If a new logic program is being used, this is the whole range,
            ##      - If a saved grounding is being used, we only need to insert those have haven't previously been inserted,
            ##      - If there was an iterrupting division, then we will be reinserting some, but there is not way to avoid this.
            sgoals_range: range = problem.sgoals_range
            if problem.first_sgoals <= self.total_achieved_sgoals(level) and _use_saved_grounding: 
                sgoals_range = range(self.total_achieved_sgoals(level) + 1, problem.last_sgoals + 1)
            
            local_program.add_rules([sgoal for index in _sgoals
                                     for sgoal in _sgoals[index]
                                     if index in sgoals_range],
                                    program_part="base")
        
        #############################################################################
        #### Perform the search for a plan that solves the given problem
        #############################################################################
        
        ## Solve the logic program in planning mode
        solve_signal: ASP.SolveSignal
        solution: Solution
        try:
            ## If we are using the saved grounding;
            ##      - Resume solving the saved program,
            ##      - Add the previous actions and fluents to the program.
            if _use_saved_grounding:
                self.__logger.log(self.__log_level(Verbosity.Verbose),
                                  f"Resuming saved grounding: {local_program}")
                
                ## The existing incrementor and step bounds are used;
                ##      - We may be farther in the search length than the achievement step of the first sgoals iff a preemptive reactive division was made.
                with local_program.resume() as solve_signal:
                    
                    self.__fix_plan_to_grounding(solve_signal,
                                                 (action for step in self.__actions[level]
                                                  for action in self.__actions[level][step]
                                                  if step <= problem.start_step),
                                                 (fluent for step in self.__states[level]
                                                  for fluent in self.__states[level][step]
                                                  if step <= problem.start_step))
                    self.__logger.log(self.__log_level(Verbosity.Verbose),
                                      "Existing plan added to saved grounding.")
                    
                    ## To account for blends in sequential yield planning we have to consider how many sub-goals have already been achieved in the goal sequence during search in this saved grounding
                    first_sgoals_accounting_for_blends = max(problem.first_sgoals, self.total_achieved_sgoals(level) + 1)
                    
                    solution = self.__search(solve_signal, level, problem.start_step, first_sgoals_accounting_for_blends, problem.last_sgoals, problem.is_final,
                                             problem.sequential_yield, detect_interleaving, generate_search_space, make_observable, division_strategy)
                    
                    self.__handle_groundings(solve_signal, local_program, level, _save_grounding, problem.is_final)
            
            ## Otherwise, start a new program.
            else:
                self.__logger.log(self.__log_level(Verbosity.Verbose),
                                  f"Starting new logic program: {local_program!s}")
                
                ## A new incrementor is needed because a new program grounding is being created
                incrementor = ASP.SolveIncrementor(step_start=problem.start_step,
                                                   step_increase_initial=((problem.search_length_bound - problem.start_step) + 1) if problem.use_search_length_bound else 2,
                                                   step_end_min=(problem.search_length_bound + 1),
                                                   step_end_max=length_limit if length_limit is not None else None,
                                                   stop_condition=(ASP.SolveResult.Satisfiable
                                                                   if (not problem.sequential_yield
                                                                       and not _generate_search_space)
                                                                   else None),
                                                   cumulative_time_limit=time_limit)
                
                ## If this grounding is going to be saved and continued later then setup the external context function for updating the total last sub-goal stage index
                def get_total_last_sgoals(problem_level: clingo.Symbol) -> clingo.Symbol:
                    if _save_grounding or _use_saved_grounding:
                        return clingo.Number(self.__total_last_sgoals[int(str(problem_level))])
                    else: return clingo.Number(problem.last_sgoals)
                context: Iterable[Callable[..., clingo.Symbol]] = [get_total_last_sgoals]
                
                ## A new incrementor will be used in this case
                with local_program.start(solver_options=solver_options,
                                         count_multiple_models=_generate_problem_space,
                                         context=context,
                                         solve_incrementor=incrementor,
                                         base_parts=program_parts.base_parts,
                                         inc_parts=program_parts.inc_parts) as solve_signal:
                    
                    solution = self.__search(solve_signal, level, problem.start_step, problem.first_sgoals, problem.last_sgoals, problem.is_final,
                                             problem.sequential_yield, detect_interleaving, generate_search_space, make_observable, division_strategy)
                    
                    self.__handle_groundings(solve_signal, local_program, level, _save_grounding, problem.is_final)
            
        except Exception as exception:
            log_and_raise(ASH_NoSolutionError, "Exception during search.", from_exception=exception, logger=self.__logger)
        
        ## Raise a no solution error if no answer set was found;
        ##      - This occurs if the program was unsatisfiable,
        ##      - or the search length or time limit was reached.
        if ((unsat := solution.answer.result.unsatisfiable)
            or (limit := (solve_signal.halt_reason in [ASP.HaltReason.StepMaximum, ASP.HaltReason.TimeLimit]
                          or solution.answer.statistics.grand_totals.total_time > time_limit))):
            log_and_raise(ASH_NoSolutionError,
                          f"The monolevel planning problem at level {level} does not have solution within given limits. Reason: "
                          + ("The problem is unsatisfiable" if unsat else "The search length or time limit was reached." if limit else "Unknown."),
                          logger=self.__logger)
        
        #############################################################################
        #### Record the solution to the given problem
        #############################################################################
        
        ## Determine whether the requested problem specification was actually solved;
        ##      - This should only not occur iff interrupting reactive problem division is enabled.
        solved_requested: bool = solution.last_achieved_sgoals == problem.last_sgoals
        achieved_finalised: bool = problem.is_final and solved_requested
        
        ## The answer set that solved the monolevel problem
        answer: ASP.Answer = solution.answer
        
        ## Save the timing and memory statistics
        partial_statistics = ASH_IncrementalStatistics.from_incremental_statistic(answer.statistics, solution.overhead_time)
        if (not _use_saved_grounding
            and (existing_statistics := self.__statistics.get(level, None)) is not None):
            self.__statistics[level] = ASH_IncrementalStatistics.from_incremental_statistic(existing_statistics.combine_with(answer.statistics, shift_increments=problem.start_step),
                                                                                            existing_statistics.overhead_time + solution.overhead_time)
        else: self.__statistics[level] = partial_statistics
        
        ## Obtain the generated plan and its produced sub-goal stages
        states: dict[int, list[Action]] = answer.fmodel.query(Fluent, param_constrs={'L' : [level, level + 1]}, group_by='S')
        actions: dict[int, list[Action]] = answer.fmodel.query(Action, param_constrs={'L' : level}, group_by='S')
        sgoals: dict[int, list[SubGoal]] = answer.fmodel.query(SubGoal, param_constrs={'L' : level}, group_by='I')
        
        ## Update the internally stored concatenated plans
        self.__states.setdefault(level, {}).update(states)
        self.__actions.setdefault(level, {}).update(actions)
        self.__sgoals.setdefault(level, {}).update(sgoals)
        
        ## Mark the current planning level as complete if the final goal state was achieved
        self.__complete_plan[level] = achieved_finalised
        
        ## Save the conformance mapping if conformance was enabled
        conformance_mapping: Optional[ConformanceMapping] = None
        if problem.conformance:
            self.__logger.debug("Previous conformance mappings:\n"
                                + "\n\n".join(f"Level = {level} >> "
                                              f"Current sub-goals: {self.__current_sgoals.get(level, {})}\n"
                                              f"Sub-goal achievement steps: {self.__sgoals_achieved_at.get(level, {})}"
                                              for level in self.level_range))
            
            ## Obtain the range of sub-goal stages that were actually achieved
            constraining_sgoals: dict[int, list[SubGoal]] = {index : sgoals for index, sgoals in self.__sgoals[level + 1].items()
                                                             if index in range(problem.first_sgoals, solution.last_achieved_sgoals + 1)}
            
            ## Construct the conformance mapping
            conformance_mapping = ConformanceMapping.from_answer(constraining_sgoals, answer, solution.sequential_yield_steps)
            
            ## Update the internally stored concatenated conformance mapping
            self.__current_sgoals.setdefault(level, ReversableDict()).update(conformance_mapping.current_sgoals)
            self.__sgoals_achieved_at.setdefault(level, ReversableDict()).update(conformance_mapping.sgoals_achieved_at)
            if (yield_steps := conformance_mapping.sequential_yield_steps) is not None:
                self.__sequential_yield_steps.setdefault(level, {}).update(yield_steps)
            
            self.__logger.debug("Updated conformance mappings:\n"
                                + "\n\n".join(f"Level = {level} >> "
                                              f"Current sub-goals: {self.__current_sgoals.get(level, {})}\n"
                                              f"Sub-goal achievement steps: {self.__sgoals_achieved_at.get(level, {})}"
                                              for level in self.level_range))
        
        fgoals_achieved_at: list[int] = []
        fgoal_ordering_correct: Optional[bool] = None
        if _order_fgoals_achievement:
            desired_order = answer.fmodel.query("goal_order", ['L', 'F', 'V', 'B', 'O'],
                                                sort_by='O', group_by='O', cast_to={'O' : [str, int]})
            goals_satisfied = answer.fmodel.query("goal_satisfied", ['L', 'F', 'V', 'B', 'S'],
                                                  sort_by='S', group_by='S', cast_to={'S' : [str, int]})
            _goals_satisfied = {}
            added = []
            for step in goals_satisfied:
                for goal_satisfied in goals_satisfied[step]:
                    for order in desired_order:
                        for goal_order in desired_order[order]:
                            if (goal_satisfied['F'] == goal_order['F']
                                and goal_satisfied['V'] == goal_order['V']
                                and goal_order not in added):
                                added.append(goal_order)
                                _goals_satisfied.setdefault(step, []).append(goal_order)
            goals_satisfied = _goals_satisfied
            
            fgoals_achieved_at = list(goals_satisfied.keys())
            
            def check_order() -> bool:
                current_order: int = 0
                for step in goals_satisfied:
                    for goal_satisfied in goals_satisfied[step]:
                        for order in desired_order:
                            for goal_order in desired_order[order]:
                                if (goal_satisfied['F'] == goal_order['F']
                                    and goal_satisfied['V'] == goal_order['V']):
                                    if order < current_order:
                                        return False
                                    current_order = order
                return True
            
            fgoal_ordering_correct = check_order()
            
            self.__logger.log(self.__log_level(Verbosity.Standard),
                              "Final-goal intermediate ordering:\n"
                              f"Achievement steps = {fgoals_achieved_at}\n"
                              f"Correct = {fgoal_ordering_correct}")
        
        ## Determine the number of final-goal preemptive achievement choices taken
        total_choices: int = 0
        preemptive_choices: int = 0
        if _preempt_mode == PreemptMode.Heuristic:
            clingo_stats: dict = answer.statistics.incremental[answer.statistics.calls].clingo_stats
            total_choices = int(clingo_stats["accu"]["solving"]["solvers"]["choices"])
            preemptive_choices = int(clingo_stats["accu"]["solving"]["solvers"]["extra"]["domain_choices"])
        elif _preempt_mode == PreemptMode.Optimise:
            if _preempt_pos_fgoals and not _preempt_neg_fgoals:
                preemptive_choices -= answer.fmodel.cost[-1]
            if _preempt_neg_fgoals and not _preempt_pos_fgoals:
                preemptive_choices += answer.fmodel.cost[-1]
            if _preempt_pos_fgoals and _preempt_neg_fgoals:
                preemptive_choices -= answer.fmodel.cost[-2]
                preemptive_choices += answer.fmodel.cost[-1]
        
        ## Construct the resulting monolevel plan
        monolevel_plan = MonolevelPlan(level=level,
                                       model_type=self.__domain.get_model_type(level),
                                       states=states,
                                       actions=actions,
                                       produced_sgoals=sgoals,
                                       is_final=achieved_finalised,
                                       planning_statistics=partial_statistics,
                                       conformance_mapping=conformance_mapping,
                                       problem_divisions=solution.reactive_divisions,
                                       total_choices=total_choices,
                                       preemptive_choices=preemptive_choices,
                                       fgoals_achieved_at=fgoals_achieved_at,
                                       fgoal_ordering_correct=fgoal_ordering_correct)
        
        ## Record how much of the requested problem was actually solved
        description: str
        if solution.last_achieved_sgoals < problem.last_sgoals:
            description = "Search was interrupted by a reactive division and a smaller than requested partial problem was solved."
        else: description = "Search finished as expected, the requested partial problem has been solved entirely."
        self.__logger.log(self.__log_level(Verbosity.Standard),
                          "Search ended :: "
                          f"Last achieved goal index = {solution.last_achieved_sgoals}, "
                          f"Last requested goal index = {problem.last_sgoals}, "
                          f"The problem was {(((solution.last_achieved_sgoals - problem.first_sgoals) + 1) / ((problem.last_sgoals - problem.first_sgoals) + 1)) * 100}% solved:"
                          f"\n{description}")
        
        ## Declare success
        self.__logger.log(self.__log_level(Verbosity.Standard),
                          "Monolevel plan generated successfully:\n" +
                          "\n".join(map(str, [f"{solution.answer.result} : {'COMPLETE PLAN OBTAINED' if achieved_finalised else 'PARTIAL PLAN OBTAINED'}",
                                              solution.answer.statistics, solution.answer.fmodel])))
        
        ## Log the plan
        header: str = (center_text(f"Plan at abstraction level {level}", append_blank_line=True, framing_width=40, centering_width=60) +
                       center_text(f"Steps = {monolevel_plan.plan_length} :: Actions = {monolevel_plan.total_actions}",
                                    framing_width=28, frame_before=False, framing_char='-', centering_width=60))
        plan: str = "\n".join(format_actions(self.__actions[level], {step : self.__sgoals[level + 1][index] for step, index in self.__current_sgoals.get(level, {}).items()}))
        self.__logger.log(self.__log_level(Verbosity.Verbose), f"\n\n{header}\n\n" + plan)
        if self.__verbosity.value.level == Verbosity.Standard.value.level:
            print(f"{header}\n\n" + "\n".join([f"{step:<3d} => [ " + ", ".join(f"{action['R']}: {action['A']}" for action in actions) + " ]"
                                               for step, actions in self.__actions[level].items()]) + "\n")
        
        ## Return the plan
        return monolevel_plan
    
    def hierarchical_plan(
                          ## Required arguments
                          self,
                          bottom_level: Optional[int] = None,
                          top_level: Optional[int] = None,
                          concurrency: bool = True,
                          conformance: bool = True,
                          /,
                          
                          ## Common arguments
                          conformance_type: Optional[ConformanceType] = None,
                          sequential_yield: bool = False,
                          division_strategy: Optional[DivisionStrategy] = None,
                          online_method: OnlineMethod = OnlineMethod.GroundFirst,
                          save_grounding: bool = False,
                          use_search_length_bound: bool = True,
                          avoid_refining_sgoals_marked_for_blending: bool = True,
                          make_observable: bool = False,
                          *,
                          
                          ## Optimisation options
                          minimise_actions: Optional[bool] = None,
                          order_fgoals_achievement: bool = False,
                          preempt_pos_fgoals: Optional[bool] = None,
                          preempt_neg_fgoals: Optional[bool] = False,
                          preempt_mode: PreemptMode = PreemptMode.Heuristic,
                          
                          ## Performance analysis options
                          detect_interleaving: bool = False,
                          generate_search_space: bool = False,
                          generate_solution_space: bool = False,
                          
                          ## Search limits
                          time_limit: Optional[HierarchicalNumber] = None,
                          length_limit: Optional[HierarchicalNumber] = None,
                          
                          ## Debugging options
                          pause_on_level_change: bool = False,
                          pause_on_increment_change: bool = False
                          
                          ) -> HierarchicalPlan:
        """
        Generate a hierarchical plan that solves the current hierarchical conformance refinement planning problem.
        
        Parameters
        ----------
        `bottom_level: int` - The abstraction level of the hierarchical problem.
        
        `top_level: int` - The abstraction level of the hierarchical problem.
        
        `concurrency: bool` - Whether action concurrency should be enabled, if disabled actions can only be planned sequentially.
        
        `conformance: bool` - Whether to apply conformance constraints, enabling conformance refinement planning mode.
        
        `conformance_type: ConformanceType` - The conformance constraint application type; simultaneous or sequential sub-goal stage achievement.
        
        `sequential_yield: bool` - Whether the conformance refinement planning problem should be solved in sequential yield mode.
        
        `division_strategy : {DivisionStrategy | None}` - The division strategy being used to divide the planning problem.
        
        `online_method: OnlineMethod = OnlineMethod.GroundFirst` - The online planning method used to traverse the problem division tree.
        
        `save_grounding : bool` - Whether the logic program's grounding will be saved.
        
        `use_search_length_bound: bool` - Whether to use the minimum search length bound.
        
        `avoid_refining_sgoals_marked_for_blending: bool = True` - Deprecated, to be removed.
        
        `make_observable: bool = False` - Deprecated, to be removed.
        
        `minimise_actions : bool` - Whether to enable the action minimisation statement.
        
        `order_fgoal_achievement : bool` - Whether to enable the final-goal intermediate achievement ordering preference optimisation statement.
        
        `preempt_pos_fgoals : bool` - Whether to enable positive final-goal preemptive achievement.
        
        `preempt_neg_fgoals : bool` - Whether to enable negative final-goal preemptive achievement.
        
        `preempt_mode : PreemptMode` - The mode by which the final-goal preemptive achievement is applied.
        
        `detect_interleaving : bool = False` - Deprecated, to be removed.
        
        `generate_search_space : bool = False` - Deprecated, to be removed.
        
        `generate_solution_space : bool = False` - Deprecated, to be removed.
        
        `time_limit: {int | None} = None` - The search time limit in seconds, if reached cancels search.
        
        `length_limit: {int | None} = None` - The search length limit, if reached cancels search.
        
        `pause_on_level_change: bool = False` - Whether to pause until user input on a hierarchical level change.
        
        `pause_on_increment_change: bool = False` - Whether to pause until user input on a online planning increment change.
        """
        
        self.__logger.debug("Arguments:\n\t" + "\n\t".join([str(local) for local in locals().items()]))
        
        ## Get the level range
        level_range: range = self.constrained_level_range(bottom_level, top_level)
        
        ## Online planning mode is only enabled if a division strategy is given and not None
        online: bool = division_strategy is not None
        
        self.__logger.log(self.__log_level(Verbosity.Simple),
                          f"Generating hierarchical plan : LEVELS [{min(level_range)}-{max(level_range)}] : {'ONLINE' if online else 'OFFLINE'} MODE")
        
        ## Declare local variables
        monolevel_plan: MonolevelPlan
        last_achieved_sgoals: int
        problems: dict[int, int] = {}
        increments: int = 0
        
        ## Simple function for converting hierarchical arguments
        def convert(arg: Optional[Union[Number, dict[int, Number]]],
                    level: int, default: Optional[Number] = None) -> Optional[int]:
            return (default if arg is None
                    else arg if isinstance(arg, (int, float))
                    else convert(arg.get(level, default), default))
        
        ## Log if conformance is enabled, the given top-level is not the domain's top-level, and their are no sub-goals at the previous level
        if conformance and max(level_range) != self.top_level and not self.__sgoals.get(max(level_range) + 1, {}):
            self.__logger.debug(f"Starting hierarchical planning at level {max(level_range)} "
                                "which is not the top-level and no sub-goals exist at previous level.")
        
        total_increments_prediction: Optional[int] = None
        if online: total_increments_prediction = division_strategy.total_increments_prediction(self.__domain, online_method)
        planning_increment_bar = tqdm(desc="Online increments", unit="increment",
                                      disable=(self.__verbosity not in [Verbosity.Simple, Verbosity.Minimal]) or not online,
                                      total=total_increments_prediction, initial=0, leave=False, ncols=180, miniters=1, colour="green")
        
        ## Loop that deals with the left-to-right online incremental progression over partial-problems;
        ##      - Continue incrementing until the ground-level is complete (there is only one increment in offline mode).
        while not self.__complete_plan.get(min(level_range), False):
            
            self.__logger.log(self.__log_level(Verbosity.Verbose),
                              "Current online planning diagram progression:\n"
                              + "\n".join(f"Level = {level} : "
                                          f"Solved problems = {problems.get(level, 0)} : "
                                          f"Total constraining sub-goal stages = {len(self.__sgoals.get(level + 1, {}))} : "
                                          f"Goals achieved = {len(self.__sgoals_achieved_at.get(level, {}))} : "
                                          f"Goals unachieved = {len(self.__sgoals.get(level + 1, {})) - len(self.__sgoals_achieved_at.get(level, {}))} : "
                                          f"Complete = {self.__complete_plan.get(level, False)}"
                                          for level in reversed(level_range)))
            
            ## A new online planning increment has started
            increments += 1
            
            ## Determine the currently valid planning level range
            current_level_range: range
            lowest_planning_level: int = self.get_valid_planning_level(False, min(level_range), max(level_range))
            highest_planning_level: int = self.get_valid_planning_level(True, min(level_range), max(level_range))
            self.__logger.log(self.__log_level(Verbosity.Verbose),
                              f"Current valid planning levels: Lowest = {lowest_planning_level}, Highest = {highest_planning_level}.")
            
            ## Determine the level range for the current planning increment as defined by the given online planning method;
            ##      - Ground-first solves downwards from the lowest planning level to the ground level,
            ##      - Complete-first solves only the highest planning level,
            ##      - Hybrid uses ground-first one the first increment only and the complete-first thereafter.
            if (online_method == OnlineMethod.GroundFirst
                or (online_method == OnlineMethod.Hybrid
                    and increments == 1)):
                current_level_range = range(min(level_range), lowest_planning_level + 1)
            elif (online_method == OnlineMethod.CompleteFirst
                  or (online_method == OnlineMethod.Hybrid
                      and increments > 1)):
                current_level_range = range(highest_planning_level, highest_planning_level + 1)
            else: raise ASH_InvalidInputError(f"Could not interpret given online method {online_method} of type {type(online_method)}.")
            self.__logger.log(self.__log_level(Verbosity.Verbose),
                              f"Chosen level range for online planning increment {increments} "
                              f"by method {online_method.value} is [{min(current_level_range)}-{max(current_level_range)}].")
            
            ## Progress bar for tracking the hierarchical progression on the current planning increment
            planning_level_tracking_progress_bar = tqdm(desc="Hierarchical progression", unit="level",
                                                        disable=(self.__verbosity not in [Verbosity.Simple, Verbosity.Minimal]),
                                                        total=max(current_level_range),
                                                        initial=0, leave=False, ncols=180, miniters=1, colour="yellow")
            
            ## Loop that deals with the top-to-bottom hierarchical progression loop;
            ##      - Solve current (first valid) partial problem at each abstraction level in the current range in descending order.
            for level in reversed(current_level_range):
                
                ## Keep track of the number of problems solved per level
                if level not in problems:
                    problems[level] = 0
                problems[level] += 1
                
                ## Only enforce conformance if there are sub-goal stages at the previous level to use for conformance
                _conformance: bool = (conformance and (level + 1) in self.__sgoals)
                
                ## Use the pre-existing scenarios to divide the problem;
                ##      - Determine the sub-goal stage range to define the current planning problem
                dividing_scenario: Optional[DivisionScenario] = None
                first_sgoals: Optional[int] = None
                last_sgoals: Optional[int] = None
                
                ## Determine the possibly partial monolevel problem's sub-goal stage range to refine
                if _conformance:
                    ## Assume offline planning and thus include the total range of sub-goal stages in order to refine the entire plan from the previous level as a complete problem
                    first_sgoals = 1
                    last_sgoals = len(self.__sgoals.get(level + 1, {}))
                    
                    ## If the division strategy was given and not None then;
                    ##      - Online planning is enabled and the division scenarios generated from the strategy at previous levels are used to form partial refinement problems at the current,
                    ##      - Each partial problem is defined by a sub-sequence of the sub-goal stages from the previous level.
                    if division_strategy is not None:
                        dividing_scenario = self.get_current_division_scenario(level + 1)
                        if dividing_scenario is None:
                            raise ASH_InternalError(f"No valid current division scenario for problem {problems[level]} at level {level}.")
                        
                        self.__logger.log(self.__log_level(Verbosity.Verbose),
                                          f"Using division scenario from previous level {level + 1} to proactively divide planning problem {problems[level]} at level {level}:\n{dividing_scenario}")
                        
                        if problems[level] not in dividing_scenario.problem_range:
                            raise ASH_InternalError(f"Invalid problem number for current division scenario: number = {problems[level]}, scenario range = {dividing_scenario.problem_range}.")
                        
                        ## TODO The sub-goal stage range should include the true division points, and the calculated blend points
                        sgoals_range: SubGoalRange = dividing_scenario.get_subgoals_indices_range(problems[level])
                        first_sgoals, last_sgoals = sgoals_range.first_index, sgoals_range.last_index
                        self.__logger.log(self.__log_level(Verbosity.Verbose),
                                          f"Proactively chosen sgoals range = [{first_sgoals}-{last_sgoals}]")
                
                ## Find the search limits
                _time_limit: Optional[float] = convert(time_limit, level)
                if _conformance:
                    current_hierarchical_plan: HierarchicalPlan = self.get_hierarchical_plan(bottom_level=level,
                                                                                             ignore_missing=True)
                    total_planning_time_at_previous_level: float = current_hierarchical_plan.get_completion_time(current_hierarchical_plan.bottom_level)
                    _time_limit = _time_limit - total_planning_time_at_previous_level
                _length_limit: Optional[Number] = convert(length_limit, level)
                if isinstance(_length_limit, float):
                    if not _conformance:
                        log_and_raise(ASH_InvalidInputError, "Cannot use an expansion factor as a length limit for classical planning.", logger=self.__logger)
                    _length_limit = _length_limit * len(self.__actions[level + 1])
                
                ## Plan at the current level
                try:
                    if _conformance:
                        self.__logger.debug(f"Starting refinement planning: level = {level}, sgoals range = [{first_sgoals}-{last_sgoals}], problem number = {problems[level]}, increment number = {increments}")
                    else: self.__logger.debug(f"Starting top-level classical planning: level = {level}")
                    
                    ## Attempt to solve the requested partial problem
                    monolevel_plan = self.monolevel_plan(## Required arguments
                                                         level,
                                                         concurrency,
                                                         _conformance,
                                                         
                                                         ## Common arguments
                                                         conformance_type=conformance_type,
                                                         first_sgoals=first_sgoals,
                                                         last_sgoals=last_sgoals,
                                                         sequential_yield=sequential_yield,
                                                         division_strategy=division_strategy,
                                                         save_grounding=save_grounding,
                                                         use_saved_grounding=save_grounding,
                                                         use_search_length_bound=use_search_length_bound,
                                                         make_observable=make_observable,
                                                         
                                                         ## Optimisation options
                                                         minimise_actions=minimise_actions,
                                                         order_fgoals_achievement=order_fgoals_achievement,
                                                         preempt_pos_fgoals=preempt_pos_fgoals,
                                                         preempt_neg_fgoals=preempt_neg_fgoals,
                                                         preempt_mode=preempt_mode,
                                                         
                                                         ## Performance analysis options
                                                         detect_interleaving=detect_interleaving,
                                                         generate_search_space=generate_search_space,
                                                         generate_solution_space=generate_solution_space,
                                                         
                                                         ## Search limits
                                                         time_limit=_time_limit,
                                                         length_limit=_length_limit)
                    
                    ## Save the solution if one was found
                    self.__partial_plans.setdefault(level, {})[problems[level]] = monolevel_plan
                    
                except ASH_NoSolutionError as error:
                    ## An error will be raise if a solution to the planning problem was not found
                    planning_level_tracking_progress_bar.close()
                    planning_increment_bar.close()
                    if division_strategy is not None:
                        division_strategy.reset()
                    log_and_raise(ASH_NoSolutionError, f"The hierarchical planning problem over levels [{min(level_range)}-{max(level_range)}] does not have a valid solution.",
                                  from_exception=error, logger=self.__logger)
                
                ## If the monolevel plan is a refined plan (true iff it has a conformance mapping);
                ##      - Check whether the last sub-goal stage index that was actually achieved is that which was requested;
                ##      - The last index will be less than the last index that was requested to be achieved iff an interrupting reactive division was made.
                if (conformance_mapping := monolevel_plan.conformance_mapping) is not None:
                    last_achieved_sgoals: int = conformance_mapping.constraining_sgoals_range.last_index
                    interrupted: bool = last_achieved_sgoals < last_sgoals
                    
                    ## Obtain achievement step of last sub-goal stage and the concatenated plan length
                    last_achievement_step: int = conformance_mapping.sgoals_achieved_at[last_achieved_sgoals]
                    plan_length: int = monolevel_plan.end_step
                    
                    ## If the problem was final then the last goal index included the final-goal;
                    ##      - The achievement step of the final-goal might not be the same as the achievement step of the last sub-goal stage,
                    ##      - So the achievement step of the last goal index becomes the end step of the concatenated plan.
                    if monolevel_plan.is_final:
                        if last_achievement_step < plan_length:
                            trailing_plan_length = plan_length - last_achievement_step
                            self.__logger.debug(f"Final plan has trailing plan length of {trailing_plan_length}.")
                        last_achievement_step = plan_length
                    
                    if last_achievement_step != plan_length:
                        raise ASH_InternalError(f"Last achievement step {last_achievement_step} not equal to plan length {plan_length}.")
                    
                    ## The problem was divided preemptively iff both;
                    ##      - The final-goal was not achieved, TODO
                    ##      - The achievement step of the last achieved sub-goal stage was less than the search length.
                    statistics: ASH_IncrementalStatistics = monolevel_plan.planning_statistics
                    search_length: int = (statistics.incremental[statistics.calls].step_range.stop - 1)
                    preemptive: bool = last_achievement_step < search_length
                
                ## If the problem is was divided proactively by a division scenario and either;
                ##      - The problem was divided reactively by;
                ##          - One or more continuous divisions,
                ##          - An interrupting division such that a smaller than requested partial problem was solved then,
                ##      - The scenario must be updated to account for these unknown reactive divisions,
                ##      - If an interrupting division was made, this will affect the hierarchical and incremental progression, and thus the online planning diagram by adding an additional problem transition.
                if (dividing_scenario is not None
                    and monolevel_plan.problem_divisions):
                    self.__logger.debug(f"Updating existing division scenario for reactive divisions:\n{dividing_scenario}")
                    
                    for division_point in monolevel_plan.problem_divisions:
                        self.__logger.debug(f"Updating with reactive division:\n{division_point}")
                        dividing_scenario.update_reactively(division_point)
                    
                    self.__logger.debug(f"Updated division scenario:\n{dividing_scenario}")
                    
                    ## If a preemptive interrupting divison was made then;
                    ##      - Note that the search length beyond the achievement step of the sub-goal stage at the index of division;
                    ##          - Is wasted if a saved grounding is not being used and those steps will be re-grounded/solved,
                    ##          - Is preserved is a saved grounding is being used and you can start from the reached search length.
                    if interrupted and preemptive:
                        if save_grounding:
                            self.__logger.log(self.__log_level(Verbosity.Verbose),
                                              "Search length difference beyond preemptive interrupting reactive division point on saved grounding:\n"
                                              f"Plan length = {plan_length}, search length = {search_length}, difference = {search_length - plan_length}")
                        else: self.__logger.log(self.__log_level(Verbosity.Verbose),
                                                "Search length wasted by preemptive interrupting reactive division without saved grounding:\n"
                                                f"Plan length = {plan_length}, search length = {search_length}, difference = {search_length - plan_length}")
                
                ## Obtain the start and end steps of the plan that is about to be divided at the current level;
                ##      - Starts from the step equal to the number of achieved sub-goal stages at the next level (this will be the step of the end state of the matching child of the last achieved sgoal),
                ##      - This would be the abstract partial plan that was just generated (extending from the last inherited division at the next level) for ground-first,
                ##      - or it will be the abstract complete plan that was just generated for complete-first.
                start_step: int = self.total_achieved_sgoals(level - 1)
                end_step: Optional[int] = None
                self.__logger.debug(f"Default value of start step and end steps of abstract plan to divide: {start_step=}, {end_step=}")
                
                ## Determine number of previously solved problems and account for;
                ##      - Parts of the plan at the current level that have not been refined but have already been divided;
                ##          - This cannot happen in ground-first or complete-first, since we never ascend back to a level unless all its produced sgoals have been refined,
                ##          - But other more general methods, like hybrid, we need to be able to account for this possibility.
                previously_solved_problems: int = problems.get(level - 1, 0)
                if (scenarios := self.__division_scenarios.get(level, [])):
                    start_step = max(start_step, scenarios[-1].last_index)
                    previously_solved_problems = max(previously_solved_problems, max(scenarios[-1].problem_range))
                    self.__logger.debug(f"Value of start step of abstract plan to divide adjusted for parts of the existing plan at the current level that have already been divided: {start_step=}")
                
                ## If this planning problem was possibly partial (it was divided by a division scenario);
                ##      - The blends of the division points must be considered.
                if dividing_scenario is not None:
                    division_points: DivisionPointPair = dividing_scenario.get_division_point_pair(problems[level])
                    
                    ## If the right division point of the current partial problem is not the last in the dividing scenario;
                    ##      - Do not divide (or refine) the part of this abstract plan that achieves sgoals in the right blend,
                    ##      - This essentially just avoids refining the sub-goal stages in the right-blend of the solution to the current monolevel problem,
                    ##          - If it is the last, then the sub-goal stage producing actions of the last sub-goal stage of this scenario, divides the entire plan that current exists at the previous level, or divides up to the achievement of the final-goal,
                    ##          - In this case, we have to divide the whole plan, and cannot use the last achievement step as the end step, as there may be a trailing plan which does not descend from any sub-goal stages from the previous level.
                    if (right_point := division_points.right) is not None:
                        end_step = self.__sgoals_achieved_at[level][right_point.index]
                        self.__logger.debug(f"Value of end step of abstract plan to divide adjusted for right blend of current problem: {end_step=}")
                    
                    ## To avoid dividing/refining sub-goal stages marked for blending;
                    ##      - This essentially further avoids refining the sub-goal stages in the left-blend of the solution to the monolevel problem,
                    ##          - Do not divide the plan at the current level that will be revised by the left blend of the next partial problem,
                    ##          - This avoids refining those sub-goal stages multiple times, since they will be revised by blending and must be refined again anyway,
                    ##              - Start from the step that the left blend point of the right division point of the current partial problem was achieved,
                    ##              - All indices less than that point cannot be revised by blending so can only ever be refined once, avoiding refining them multiple times,
                    ##                but giving us smaller problems at the lower levels, which may make those plans greedy in the long run,
                    ##      - This does avoid re-refining plans at the next level, that achieve sub-goal stages that descend from those in the left blend of the
                    ##        next problem at the current level, but also has a significant impact on the hierarchical progression and problem division process.
                    if avoid_refining_sgoals_marked_for_blending: ## TODO or level == bottom_level:
                        if (right_point := division_points.right) is not None:
                            problem_size: int = dividing_scenario.get_subgoals_indices_range(problems[level], ignore_blend=True).problem_size
                            ## The index when left point could actually be index 1 (the first one which means that the next problem is actually initial!),
                            ## modified to 0 to get index of previous sub-goal stage before the blend, which is actually indicating the initial state!
                            ## This means that the whole plan is going to be revised, and there is nothing to be refined!
                            end_step = min(end_step, self.__sgoals_achieved_at[level].get(right_point.index_when_left_point(problem_size) - 1, 0))
                            self.__logger.debug(f"Avoiding refining at level {level - 1} produced sgoals in range [{end_step + 1}-{monolevel_plan.end_step}] marked to be revised by problem blending at level {level}.")
                    
                    ## Otherwise, re-refine any sub-goal stages revised by blending;
                    ##      - Start from the step that achieved the sub-goal stage immediately preceeding that of the left-blend index of the current monolevel problem,
                    ##      - Ensure that we still divide over any sub-goal stages that are yet to be achieved, i.e. parts of the
                    ##        plan at this level that have not been refined yet (this can happen in complete-first or hybrid mode).
                    else:
                        start_step = min(start_step, self.__sgoals_achieved_at[level].get(first_sgoals - 1, 0))
                        self.__logger.debug(f"Value of start step of abstract plan to divide adjusted for left blend of current problem: {start_step=}")
                
                ## If the entire monolevel plan that was just generated is not to be divided;
                ##      - Discard the sub-goals produced at this level that were produced from abstract actions
                ##        that achieved ascending sub-goal stages from the current monolevel refinement problem;
                ##          - That were inside the right-blend of the current problem,
                ##          - If avoiding refining sub-goal stages marked for blending;
                ##              - Will be revised by the left-blend of the next problem,
                ##      - This is necessary to ensure the valid planning levels are determined correctly.
                if end_step is not None:
                    states: dict[int, list[Action]] = {}
                    actions: dict[int, list[Fluent]] = {}
                    produced_sgoals: dict[int, list[SubGoal]] = {}
                    sgoals_achieved_at: dict[int, int] = {}
                    for step in self.__states[level]:
                        if step <= end_step:
                            states[step] = self.__states[level][step]
                            if step > 0:
                                actions[step] = self.__actions[level][step]
                                if level != 1:
                                    produced_sgoals[step] = self.__sgoals[level][step]
                    for index in self.__sgoals_achieved_at[level]:
                        if index <= right_point.index:
                            sgoals_achieved_at[index] = self.__sgoals_achieved_at[level][index]
                    self.__states[level] = states
                    self.__actions[level] = actions
                    self.__sgoals[level] = produced_sgoals
                    self.__sgoals_achieved_at[level] = sgoals_achieved_at
                
                ## Add a division scenario for the (partial) refinement planning problem of the monolevel plan that was just generated
                if (conformance and level != 1
                    and division_strategy is not None):
                    self.__logger.log(self.__log_level(Verbosity.Verbose),
                                      "Preparing to invoke division strategy...")
                    
                    ## Proactively divide the abstract plan at the current level iff either;
                    ##      - Ground first planning is enabled (since the next level requires a new scenario since this one
                    ##        was the lowest unsolved problem so it is not possible for a scenario that divides it to exist),
                    ##      - Complete first planning is enabled and the level is complete (since we only move to the next level once the current level is complete,
                    ##        and we only ever generate one scenario per level, so this is the first and only time we generate a scenario).
                    if ((online_method == OnlineMethod.GroundFirst
                         or (online_method == OnlineMethod.Hybrid
                             and increments == 1))
                        or ((online_method == OnlineMethod.CompleteFirst
                             or (online_method == OnlineMethod.Hybrid
                                 and increments > 1))
                            and self.__complete_plan.get(level, False))):
                        
                        self.__logger.log(self.__log_level(Verbosity.Verbose),
                                          f"Online method {online_method} chose to invoke the division strategy before a downwards level change:"
                                          f"Current increment = {increments}, Current planning level is complete = {self.__complete_plan.get(level, False)}")
                        
                        ## Extract the abstract plan to be divided
                        abstract_plan: MonolevelPlan = self.get_monolevel_plan(level, start_step=start_step, end_step=end_step)
                        self.__logger.log(self.__log_level(Verbosity.Verbose), f"Dividing abstract plan:\n{abstract_plan}")
                        
                        ## Generate the division scenario that divides this plan
                        generated_scenario: DivisionScenario = division_strategy.proact(abstract_plan, previously_solved_problems)
                        self.__division_scenarios.setdefault(level, []).append(generated_scenario)
                        self.__logger.log(self.__log_level(Verbosity.Verbose), f"Division scenario generated:\n{generated_scenario}")
                    
                    else: self.__logger.log(self.__log_level(Verbosity.Verbose),
                                            f"Online method {online_method} chose not to invoke the division strategy:"
                                            f"Current increment = {increments}, Current planning level is complete = {self.__complete_plan.get(level, False)}")
                
                ## The current monolevel planning problem at the current planning level has been solved
                self.__logger.log(self.__verbosity.value.log, f"Monolevel problem {problems[level]} at level {level} solved.")
                planning_level_tracking_progress_bar.update(1)
                
                ## The planning level is about to change down a level iff;
                ##      - The current planning level is the bottom of the hierarchical level range on the current online planning increment.
                if pause_on_level_change and level != min(current_level_range):
                    planning_level_tracking_progress_bar.clear()
                    planning_increment_bar.clear()
                    input("Planning paused before downwards level change"
                          f" :: Progression > current level = {level}, next level = {level - 1}, current range = [{max(current_level_range)}-{min(current_level_range)}], current increment = {increments}")
                    sys.stdout.write("\033[A")
                    sys.stdout.flush()
                    planning_increment_bar.refresh()
                    planning_level_tracking_progress_bar.refresh()
            
            ## The current online planning increment has finished
            self.__logger.log(self.__verbosity.value.log, f"Online planning increment {increments} finished.")
            planning_level_tracking_progress_bar.close()
            planning_increment_bar.update(1)
            if pause_on_increment_change and online:
                planning_increment_bar.clear()
                input("Planning paused before online planning increment change"
                      f" :: Progression > finished increments = {increments}, predicted total = {total_increments_prediction}")
                sys.stdout.write("\033[A")
                sys.stdout.flush()
                planning_increment_bar.refresh()
            
        else:
            planning_increment_bar.close()
            if division_strategy is not None:
                division_strategy.reset()
        
        ## Extract the resulting hierarchical plan
        hierarchical_plan = self.get_hierarchical_plan(min(level_range), max(level_range))
        
        ## Log the result of hierarchical planning
        self.__logger.log(self.__log_level(Verbosity.Simple),
                          "Hierarchical plan generated successfully :: Ground Plan Quality >> "
                          f"Length = {hierarchical_plan[min(level_range)].plan_length}, Actions = {hierarchical_plan[min(level_range)].total_actions}")
        self.__logger.log(self.__log_level(Verbosity.Standard),
                          "Hierarchical planning summary: "
                          f"(Execution latency = {hierarchical_plan.execution_latency_time}, "
                          f"Average ground wait time = {hierarchical_plan.get_average_wait_time(hierarchical_plan.bottom_level)}, "
                          f"Absolution time = {hierarchical_plan.absolution_time})\n"
                          + center_text("\n".join(str(hierarchical_plan[level]) for level in reversed(level_range)),
                                        prefix_blank_line=True, vbar_left= "| ", vbar_right=" |",
                                        framing_width=200, centering_width=210, terminal_width=220))
        
        return hierarchical_plan
    
    def load_schema(self, schema: RefinementSchema, init_problem: bool = True, purge_solutions: bool = True) -> None:
        self.__logger.log(self.__log_level(Verbosity.Simple),
                          f"Loading refinement schema:\n{schema!s}")
        
        if purge_solutions:
            self.purge_solutions()
        
        ## Overwrite the constraining sub-goal stages and division scenario from the schema
        self.__sgoals[schema.level] = schema.constraining_sgoals
        self.__division_scenarios[schema.level] = [DivisionScenario.from_points(schema.problem_divisions[1:-2])]
        
        try:
            ## Ensure that the problem is initialised as requested
            if not self.problem_initialised:
                if init_problem:
                    self.initialise_problem()
                else:
                    self.__logger.log(self.__log_level(Verbosity.Standard, level=logging.WARNING),
                                      "Refinement schema loaded with non-initialised problem.")
        except ASH_NoSolutionError as error:
            raise log_and_raise(ASH_InternalError, "Refinement schema failed to load.",
                                from_exception=error, logger=self._planner_logger)
        
        self.__logger.log(self.__log_level(Verbosity.Verbose),
                          "Refinement schema loaded successfully.")
    
    ################################################################################
    #### Auxillary functions
    ################################################################################
    
    def __search(self,
                 solve_signal: ASP.SolveSignal,
                 level: int,
                 start_step: int = 0,
                 first_sgoals: int = 1,
                 last_sgoals: int = 1,
                 finalise: bool = True,
                 sequential_yield: bool = False,
                 /,
                 
                 detect_interleaving: bool = False,
                 generate_search_space: bool = False,
                 
                 make_observable: bool = False,
                 division_strategy: Optional[DivisionStrategy] = None) -> Solution:
        """
        Search for a plan by running an incremental solve call with a given solve signal.
        
        Parameters
        ----------
        `solve_signal: ASP.SolveSignal` - An ASP solve signal to be used to handle the search.
        
        `level: int` - The abstraction level of the problem.
        
        `start_step: int = 0` - The start step of the problem.
        
        `first_sgoals: int` - The requested first in sequence sub-goal stage index value (inclusive) of a conformance refinement planning problem.
        
        `last_sgoals: int`- The requested last in sequence sub-goal stage index value (inclusive) of a conformance refinement planning problem.
        
        `finalise: bool = True` - Whether the problem is final, i.e. it achieves the final-goal at the given level.
        
        `sequential_yield: bool` - Whether the conformance refinement planning problem should be solved in sequential yield mode.
        
        `detect_interleaving: bool = False` - Deprecated, to be removed.
        
        `generate_search_space: bool = False` - Deprecated, to be removed.
        
        `make_observable: bool = False` - Deprecated, to be removed.
        
        `division_strategy: Optional[DivisionStrategy] = None` - The reactive division strategy being used to divide the planning problem.
        """
        self.__logger.debug("Starting search:\n\t" + "\n\t".join(map(str, locals().items())))
        
        if (sequential_yield
            and generate_search_space):
            raise ASH_InternalError("Problem spaces cannot be generated in sequential yield mode. "
                                    "To generate the problem spaces for a reactively divided plan, "
                                    "generate a refinement schema in sequential yield mode, "
                                    "and generate the problem spaces over that schema in standard refinement mode.")
        
        ## If sequential yielding is disabled;
        if not sequential_yield:
            ## Make a sanity check that the solve incrementor has the correct stop condition,
            if solve_signal.logic_program.incrementor.stop_condition != ASP.SolveResult.Satisfiable:
                raise ASH_InternalError("Reached standard search without a stop condition on the solve incrementor.")
            
            ## Run until the solve incrementor reaches a stop condition.
            answer: ASP.Answer = solve_signal.run_for()
            return Solution(answer, last_sgoals)
        
        if generate_search_space:
            ## Ensure that the search space is generated at the step following the initial;
            ##      - This is the first step upon which actions can be generated.
            solve_signal.queue_assign_external(f"gen_search_space_at({start_step + 1})", True)
            
            def generate_search_space(feedback: ASP.Feedback) -> bool:
                "Function that generates the search space whilst the plan is incomplete."
                
                ## Generate the search space at the current search step only;
                ##      - The external atom `gen_search_step_at(step)` defines the search space generation step,
                ##      - It is assigned true at the next step (which will become current on the next solve increment),
                ##      - The previous step in released to false to allow maximisation of achieved goals up to the current step.
                solve_signal.queue_assign_external(f"gen_search_space_at({feedback.end_step + 1})", True)
                solve_signal.release_external(f"gen_search_space_at({feedback.end_step})")
                
                ## Stop only when the plan is complete
                return bool(list(solve_signal.get_answer().inc_models.values())[-1].model.get_atoms("incomplete_plan", 1, param_constrs={0 : feedback.end_step}))
            
            answer: ASP.Answer = solve_signal.run_while(generate_search_space)
            return Solution(answer, last_sgoals)
        
        ## Variable for storing the overhead time;
        ##      - The overhead time is primarily needed to record the online grounding time for the program extension that handles the reactive division,
        ##        since this is not recorded in the usual statistics objects returned by ASP Parser.
        overhead_time: float = 0.0
        
        ## Variable for keeping track of the current last in seuqnece sub-goal stage (that which is to be achieved next).
        current_last_sgoals: int = first_sgoals
        
        ## Variables for interleaving detection;
        ##      - The actions for each progression through the sub-goal stage sequence (only used when detecting interleaving).
        current_actions: Optional[dict[int, list[Action]]] = None
        previous_actions: dict[int, list[Action]] = {}
        current_conformance_mapping: ConformanceMapping
        previous_conformance_mapping: ConformanceMapping
        interleaving_quantity: int = 0
        interleaving_score: int = 0
        
        ## Variables for reactive division;
        ##      - Record all committed reactive divisions,
        ##      - Record the step upon which each sub-goal stage was minimally (greedily) achieved,
        ##      - Record the incremental times for use in time reactive time calculations.
        reactive_divisions: list[DivisionPoint] = []
        sequential_yield_steps: ReversableDict[int, int] = ReversableDict()
        increment_times: list[float] = []
        
        try:
            ## Progress bar for observing progression through the goal sequence;
            ##      - If the problem is final then;
            ##          - The goal-sequence contains the final-goal,
            ##          - Otherwise it ends at the last sub-goal stage.
            postfix: dict[str, str] = {"Exp" : "(f=0.0, d=0.0)",
                                       "Div" : "(t= 0, l= 0| 0)"}
            sgoals_progress_bar = tqdm(desc="Goals", unit="stages",
                                       disable=self.__verbosity == Verbosity.Disable,
                                       postfix=postfix,
                                       initial=(first_sgoals - 1), total=last_sgoals,
                                       leave=False, ncols=180, miniters=1, colour="magenta")
            
            ## Assign only the first in sequence sub-goal stage to be achieved initially
            solve_signal.queue_assign_external(f"current_last_sgoals({current_last_sgoals}, {start_step + 1})", True)
            
            ## If the problem is final and there is only one sub-goal stage then;
            ##      - Also immediately require the final-goal to be achieved.
            if finalise and current_last_sgoals == last_sgoals:
                solve_signal.queue_assign_external(f"seq_achieve_fgoals({start_step})", True, inc_range=ASP.IncRange())
            
            ## Insert and achieve each sub-goal stages sequentially.
            feedback: ASP.Feedback
            for feedback in solve_signal.yield_run():
                self.__logger.debug(f"Increment feedback: {feedback}")
                increment_times.append(feedback.increment_statistics.total_time)
                
                ## If the program is satisfiable;
                if feedback.solve_result == ASP.SolveResult.Satisfiable:
                    
                    ## Record local start time for calculating the overhead time
                    local_start_time: float = time.perf_counter()
                    
                    ## Mark the current sub-goal stage index as achieved and record the yield step
                    sequential_yield_steps[current_last_sgoals] = feedback.end_step
                    self.__logger.debug(f"Current sequential yield steps:\n{sequential_yield_steps!s}")
                    
                    ## Move to the next in sequence sub-goal stage index
                    current_last_sgoals += 1
                    sgoals_progress_bar.update(1)
                    total: int = (last_sgoals - first_sgoals) + 1
                    achieved: int = (current_last_sgoals - first_sgoals) + 1
                    self.__logger.debug(f"Goal at sequence index {current_last_sgoals} achieved :: Progression >> "
                                        f"total requested sgoals = {total}, current total achieved sgoals = {achieved} ({(achieved / total) * 100.0:6.2f}% solved)")
                    
                    ########
                    ## Dealing with interleaving detection and planning progression observability
                    
                    if detect_interleaving or make_observable:
                        ## Obtain the currently planned actions that minimally achieve the previous sub-goal stage index
                        current_actions = solve_signal.get_answer().fmodel.query(Action, param_constrs={'L' : level}, group_by='S')
                        
                        ## Obtain the range of sub-goal stages that were actually achieved
                        constraining_sgoals: dict[int, list[SubGoal]] = {index : sgoals for index, sgoals in self.__sgoals[level + 1].items()
                                                                         if index in range(first_sgoals, current_last_sgoals)}
                        
                        ## Find the current conformance mapping up to the minimal achievement of the previous sub-goal stage index
                        current_conformance_mapping = ConformanceMapping.from_answer(constraining_sgoals, solve_signal.get_answer(), sequential_yield_steps)
                        self.__logger.debug("Current conformance mapping:\n"
                                            + "\n".join(f"Index {index}: Sub-plan > length = {len(steps)}, steps = {steps!s}"
                                                        for index, steps in current_conformance_mapping.current_sgoals.reversed_items()))
                        
                        ## Record the currently observable planning progression
                        postfix["Exp"] = f"(f={current_conformance_mapping.length_expansion_factor:3.1f}, d={current_conformance_mapping.length_expansion_deviation:3.1f})"
                        self.__logger.debug(f"Observable partial plan that minimally uniquely achieves goal at index {current_last_sgoals - 1}:\n"
                                            + "\n".join(map(str, current_actions.items())))
                    
                    else: postfix["Exp"] = f"(f={(feedback.end_step - start_step) / (current_last_sgoals - first_sgoals):3.1f}, d=N/A)"
                    
                    ## Look for interleaving on the actions of the old plan that minimally uniquely achieved the previous sub-goal stage
                    if detect_interleaving:
                        ## If more than the first sub-goal stage has been achieved;
                        ##      - Look for differences between the previous partial plan, and the part of the current partial plan at overlaps the current, in order to detect for interleaving.
                        if (current_last_sgoals - 1) > first_sgoals:
                            Interleaving = NamedTuple("Interleaving", [("removed", list[Action]), ("added", list[Action])])
                            modified_actions: dict[int, Interleaving] = {}
                            
                            for step in current_actions:
                                action_sets: Interleaving = modified_actions.setdefault(step, Interleaving([], current_actions[step]))
                                for action in previous_actions.get(step, []):
                                    if action not in action_sets.added:
                                        action_sets.removed.append(action)
                            
                            ## Find the interleaved actions
                            for step in modified_actions:
                                
                                ## Check if any sub-goal stage was achieved at this step (it is a matching child step)...
                                if step in current_conformance_mapping.sgoals_achieved_at.values():
                                    
                                    ## If so, check if it has been delayed to allow interleaving and state preperation...
                                    index: int = current_conformance_mapping.sgoals_achieved_at.reversed_get(step, [-1])[0]
                                    old_step: Optional[int] = previous_conformance_mapping.sgoals_achieved_at.get(index, None)
                                    
                                    if (old_step is not None and step > old_step):
                                        
                                        ## Update interleaving quantity and score;
                                        ##      - The quantity is the number of interleaved sub-plans,
                                        ##      - The score is the total length difference of early sub-plans between their minimal (greedy) achievement and delayed interleaved (non-greedy) achievement.
                                        interleaving_quantity += 1
                                        interleaving_score += (step - old_step)
                                        
                                        self.__logger.debug(f"Sub-goal stage at index {index} achievement delayed by interleaving:\n"
                                                            f"old achievement step = {old_step}, new achievement step = {step}, "
                                                            f"interleaving quantity = {interleaving_quantity}, interleaving score = {interleaving_score}")
                                
                                current_index: int = current_conformance_mapping.current_sgoals.get(step, None)
                                achieved_index: int = current_conformance_mapping.sgoals_achieved_at.reversed_get(step, [None])[0]
                                self.__logger.debug(f"Interleaving at [step = {step}, current index = {current_index}, achieved index = {achieved_index}]:\n"
                                                    f"Removed actions: {modified_actions[step].removed}\n"
                                                    f"Added actions: {modified_actions[step].added}")
                            
                            postfix["Int"] = f"(q={(interleaving_quantity / (current_last_sgoals - first_sgoals)) * 100.0:4.1f}, s={(interleaving_score / (feedback.end_step - start_step)) * 100.0:4.1f}"
                        
                        ## Record the current plan as that which minimally achieved the previous sub-goal stage
                        previous_actions = current_actions
                        previous_conformance_mapping = current_conformance_mapping
                        
                        ## Finally, ensure that the plan that minimally achieves the next sub-goal stage includes as many of the same actions form the current plan as possible;
                        ##      - This ensures that different actions are only chosen iff interleaving can reduce the overall plan length for the combined problem,
                        ##      - The priority of the maximisation statement is increased with the achievement of each sub-goal stage to make sure that the correct plan is being preferred.
                        encoded_actions: str = "\n".join(f"prefer({action})." for step in previous_actions for action in previous_actions[step])
                        encoded_actions += f":~ not occurs(L, R, A, S), prefer(occurs(L, R, A, S)), pl(L). [1 @ {10 + (current_last_sgoals - 1)}, L, R, A, S]"
                        solve_signal.online_extend_program(ASP.BasePart("base"), encoded_actions)
                    
                    ########
                    ## Dealing with changing the goal sequence index
                    
                    ## The current planning problem is solved if the current sub-goal stage index is greater than the requested problem's last in sequence index.
                    if current_last_sgoals > last_sgoals:
                        self.__logger.debug(f"Terminating solving because last in problem sequence goal at index {last_sgoals} was achieved successfully.")
                        break
                    
                    ## Assign the next sub-goal stage to be achieved (checked correct 15/09/2021);
                    ##      - This is assigned at the previous search step (the step the previous sub-goal index was achieved)
                    ##        this so that the current sub-goal index becomes current at the previous step so that it is ready to be achieved at the current.
                    solve_signal.queue_assign_external(f"current_last_sgoals({current_last_sgoals}, {feedback.end_step})", True)
                    self.__logger.debug(f"Setting current sequential sub-goal stage [index = {current_last_sgoals}, step = {feedback.end_step}]:\n"
                                        + "\n".join(str(subgoal) for subgoal in self.__sgoals[level + 1][current_last_sgoals]))
                    
                    ## If the goal sequence index is the final-goal index (the problem is final and the index is the last sub-goal stage index);
                    ##      - Require the final-goal to be achieved over all following steps
                    if finalise and current_last_sgoals == last_sgoals:
                        solve_signal.queue_assign_external(f"seq_achieve_fgoals({feedback.end_step})", True, inc_range=ASP.IncRange())
                        self.__logger.debug("Enforcing achievement of final-goal:\n" + "\n".join(map(str, self.__final_goals[level])))
                    
                    ## Add the local overhead time to the total
                    overhead_time += (time.perf_counter() - local_start_time)
                
                ## Obtain the last effective division index
                last_division_index: int = first_sgoals
                if reactive_divisions:
                    last_division_index = reactive_divisions[-1].index
                
                ## Call the division strategy for reactive division iff both;
                ##      - The current sub-goal index is not;
                ##          - The first in the problem's goal sequence,
                ##          - The same as the last division index.
                ##      - And either;
                ##          - The problem is not final,
                ##          - Or the current sub-goal stage index is not the last in the problem's goal sequence.
                if (division_strategy is not None
                    and last_division_index != current_last_sgoals
                    and (not finalise or last_sgoals != current_last_sgoals)):
                    
                    ## Record local start time for calculating the overhead time
                    local_start_time: float = time.perf_counter()
                    
                    ## Call the division strategy to decide if a reacitve division should be made
                    reaction: Reaction = division_strategy.react(problem_level=level,
                                                                 problem_total_sgoals_range=SubGoalRange(first_sgoals, last_sgoals),
                                                                 problem_start_step=start_step,
                                                                 current_search_length=feedback.end_step,
                                                                 current_subgoal_index=current_last_sgoals - 1,
                                                                 matching_child=feedback.solve_result == ASP.SolveResult.Satisfiable,
                                                                 incremental_times=increment_times,
                                                                 observable_plan=current_actions)
                    self.__logger.debug(f"Reaction at search length {feedback.end_step}:\n{reaction}.")
                    
                    ## If a reactive division was made then apply it as requested
                    if reaction.divide:
                        
                        ## If the division was interrupting then cancel the search immediately;
                        ##      - Record the division point for insertion into the division scenario after search has finished as it affects the planning progression.
                        if reaction.interrupt:
                            reactive_divisions.append(DivisionPoint(current_last_sgoals - 1, reactive=feedback.end_step,
                                                                    interrupting=True, preemptive=feedback.end_step - sequential_yield_steps[current_last_sgoals - 1]))
                            self.__logger.debug(f"Interrupting planning due to reactive division:\n{reactive_divisions[-1]!s}")
                            break
                        
                        ## Calculate the backwards horizon for continuous division
                        backwards_horizon: int = 0
                        if isinstance(reaction.backwards_horizon, float):
                            backwards_horizon = round(reaction.backwards_horizon * (current_last_sgoals - last_division_index))
                        else: backwards_horizon = min(reaction.backwards_horizon, current_last_sgoals - last_division_index)
                        
                        ## Determine the sub-goal stage index and search step to fix the plan up until
                        fix_until_index: int = max(first_sgoals, current_last_sgoals - backwards_horizon - 1)
                        fix_until_step: int = feedback.end_step
                        if backwards_horizon != 0:
                            fix_until_step = sequential_yield_steps.get(current_last_sgoals - backwards_horizon - 1, start_step) ## TODO use the conformance mapping.
                        
                        ## Record the continuous division for the purpose of studying its affect on planning
                        reactive_divisions.append(DivisionPoint(fix_until_index, reactive=fix_until_step,
                                                                preemptive=feedback.end_step - sequential_yield_steps[current_last_sgoals - 1],
                                                                committed_index=current_last_sgoals - 1, committed_step=feedback.end_step))
                        self.__logger.debug(f"Reactively dividing problem:\n{reactive_divisions[-1]!s}")
                        postfix["Div"] = f"(t={len(reactive_divisions)}, l=[{reactive_divisions[-1].index}|{reactive_divisions[-1].committed_index}])"
                        
                        ## Fix the plan up to the achievement of the most recently achieved sub-goal stage minus the backwards horizon;
                        ##      - Although we are still technically solving a combined problem, these actions can no longer be changed, making this act like a problem division and simplifying overall problem complexity.
                        actions: list[Action] = solve_signal.get_answer().fmodel.query(Action, param_constrs={'L' : level, 'S' : lambda symbol: int(str(symbol)) <= fix_until_step})
                        self.__logger.debug(f"Fixing actions and accepting as partial solution up to; step = {fix_until_step}, index = {fix_until_index}:\n"
                                            + "\n".join(str(action) for action in actions))
                        self.__fix_plan_to_grounding(solve_signal, actions)
                    
                    ## Add the local overhead time to the total
                    overhead_time += (time.perf_counter() - local_start_time)
                
                ## Update the CLI progress bar
                if postfix: sgoals_progress_bar.set_postfix(postfix)
            
        finally:
            sgoals_progress_bar.close()
        
        return Solution(solve_signal.get_answer(), current_last_sgoals - 1, overhead_time, sequential_yield_steps, reactive_divisions)
    
    def __fix_plan_to_grounding(self, solve_signal: ASP.SolveSignal, actions: Iterable[Action], fluents: Iterable[Fluent] = []) -> None:
        """
        Fix a plan given by a sequence of actions and optionally fluents to the grounding of a logic program being controlled by a given solve signal.
        
        In continuous reactive division, only the actions are fixed, since this saves time parsing the answer sets each time.
        In proactive or interrupting reactive divisions (a shifting division) on a saved grounding, both actions and fluents are fixed.
        This applies a stricter constraint requiring less inferance from the planner and allowing more ground rules to be simplified away (i.e. those relating to state representation).
        
        Internal use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `solve_signal : ASP.SolveSignal` - The solve signal currently controlling the interactive incremental solve of the given logic program.
        
        `actions: Iterable[Action]` - An iterable of action literals to fix to the grounding (see Planner.Action).
        
        `fluents: Iterable[Fluent] = []` - An iterable of fluent literals to fix to the grounding (see Planner.Fluent).
        """
        ## Encode and insert the actions and fluents into the logic program;
        ##      - They cannot be inserted directly as occurs/holds statements as this would cause an atom redefinition error,
        ##      - So we enclosed them in a special predicate and add a constraint that ensures that the fixed actions must be planned.
        encoded_actions: str = "\n".join(f"fix_action({action})." for action in actions)
        encoded_actions += "\n:- not occurs(L, R, A, S), fix_action(occurs(L, R, A, S)), pl(L)."
        solve_signal.online_extend_program(ASP.BasePart("base"), encoded_actions)
        if fluents:
            encoded_fluents: str = "\n".join(f"fix_fluent({fluent})." for fluent in fluents)
            encoded_fluents += "\n:- not holds(L, F, V, S), fix_fluent(holds(L, F, V, S)), pl(L)."
            solve_signal.online_extend_program(ASP.BasePart("base"), encoded_fluents)
    
    def __handle_groundings(self, solve_signal: ASP.SolveSignal, program: ASP.LogicProgram, level: int, save_grounding: bool, finalise: bool) -> None:
        """
        Handle the groundings of a given logic program used for planning at a given abstraction level via its solve signal.
        The grounding is saved iff; requested and the problem is not final, otherwise the grounding is deleted.
        
        Internal use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `solve_signal : ASP.SolveSignal` - The solve signal currently controlling the interactive incremental solve of the given logic program.
        
        `program : ASP.LogicProgram` - The logic program being used for planning.
        
        `level : int` - An positive non-zero integer defining the current planning level.
        
        `save_grounding : bool` - A Boolean, True to save the logic program's grounding to be used for the next partial problem at the given level.
        
        `finalise : bool` - A Boolean, True if the current partial problem is final, if it is then the grounding is not saved regardless.
        """
        if save_grounding and not finalise:
            self.__logger.debug(f"Saving program grounding at level {level}.")
            solve_signal.holding = True
            self.__saved_groundings[level] = program
        
        elif level in self.__saved_groundings:
            self.__logger.debug(f"Deleting program grounding at level {level}.")
            solve_signal.holding = False
            del self.__saved_groundings[level]
