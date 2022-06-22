import enum
import logging
import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Iterator, NamedTuple, Optional, Type, Union, final
import statistics

import _collections_abc

import core.Planner as Planner

## Main module logger
_Strategies_logger: logging.Logger = logging.getLogger(__name__)
_Strategies_logger.setLevel(logging.DEBUG)

Number = Union[int, float]
Bound = Union[Optional[Number], dict[int, Optional[Number]]]
Horizon = Bound
Bounds = dict[str, Bound]

@dataclass(frozen=True)
class Blend:
    left: Number = 0
    right: Number = 0
    
    def __post_init__(self) -> None:
        for field in fields(self):
            value: Number = getattr(self, field.name)
            
            ## Ensure the quantities are numbers
            if not isinstance(value, (float, int)):
                raise TypeError(f"Blend quantities must be floats or integers. Got; {value} of type {type(value)}.")
            
            ## Ensure that both quantities are zero or greater
            if value < 0:
                if isinstance(value, float):
                    object.__setattr__(self, field.name, 0.0)
                else: object.__setattr__(self, field.name, 0)
            
            ## Ensure that percentage quantities are one or less
            if isinstance(value, float) and value > 1.0:
                object.__setattr__(self, field.name, 1.0)
    
    def __str__(self) -> str:
        return f"(Left = {self.left}, Right = {self.right})"
    
    def get_left(self, previous_problem_size: int) -> int:
        if isinstance(self.left, float):
            return int(self.left * previous_problem_size)
        return self.left
    
    def get_right(self, next_problem_size: int) -> int:
        if isinstance(self.right, float):
            return int(self.right * next_problem_size)
        return self.right

@dataclass(frozen=True, order=True)
class DivisionPoint:
    """
    Represents a division point by its sub-goal stage index.
    
    Represents a reactive division by its sub-goal stage index and the time step it was committed on.
    There can be at most one division per index, as multiple would have no affect.
    
    A proactive division is applied on a matching child step, since they are applied on the achievement of sub-goal stages.
    A reactive division can be committed on any child step of a sub-goal stage, but only applied the the previous sub-goal stage unless it is a matching child, and the solver can still continue planning from that step.
    This is called a preemptive division, and is an admissible technique because the plan cannot possibly get shorter than the current search length, or a solution would have already been found.
    
    Fields
    ------
    `index : int` - The index of this division point.
    The earlier and later partial problems are inclusive and exclusive of this sub-goal stage index respectively.
    If this division point was commited reactively, this is the sub-goal stage that was most recently achieved when it was commited.
    
    `blend : Blend = Blend()` - The blend quantities of this division point.
    
    `inherited : bool = False` - Whether this division point was inherited from the previous abstraction level due to the online planning method.
    
    `reactive : {int | None} = None` - Either None to indicate the division point was commited proactively, or an integer defining the time step the division index was achieved on.
    This is equivalent to the plan length that was fixed by the division.
    
    `interrupting : bool = False` - A Boolean, True if the division is interrupting, False if it was continuous.
    
    `preemptive : int = 0` - An integer, a positive non-zero value if this division point was committed preemptively defining how many search steps have been passed since the last sub-goal stage was achieved, zero otherwise.
    
    `committed_index : int = -1` - The sub-goal stage index that was current when this reactive division was committed.
    
    `committed_step : int = -1` - An integer defining the time step this reactive division was commited on.
    """
    index: int
    blend: Blend = field(compare=False, default_factory=Blend)
    inherited: bool = False
    reactive: Optional[int] = None
    interrupting: bool = False
    preemptive: int = 0
    committed_index: int = -1
    committed_step: int = -1
    
    def __post_init__(self) -> None:
        if (self.reactive is None
            and (self.interrupting or self.preemptive)):
            raise ValueError(f"Interrupting or preemptive divisions must be reactive. Got; {self!r}.")
        if self.reactive and self.committed_index < 0:
            object.__setattr__(self, "committed_index", self.index)
            object.__setattr__(self, "committed_step", self.reactive)
    
    def __str__(self) -> str:
        return f"(Index = {self.index}, Blend = {self.blend}, Inherited = {self.inherited}, Type = {'proactive' if self.proactive else 'reactive'}" + (f", Interrupting = {self.interrupting}, Preemptive = {self.preemptive}" if self.reactive else "") + ")"
    
    @property
    def proactive(self) -> bool:
        "Whether this division point was assigned proactively."
        return self.reactive is None
    
    @property
    def shifting(self) -> bool:
        "Whether this division point is monolevel problem shifting."
        return self.proactive or self.interrupting
    
    def index_when_left_point(self,
                              previous_problem_size: int = 0, /, *,
                              ignore_blend: bool = False
                              ) -> int:
        """
        Get the index of this division point, modified to account for its blend quantities, when acting as the left point of a partial problem.
        
        Parameters
        ----------
        `previous_problem_size : int = 0` - The normal size of the previous in sequence partial problem template from the division scenario this point belongs to.
        Where the normal size is the size ignoring all blend quantities, and the size is equal to; the index of this point, minus the index of the previous point.
        
        `ignore_blend : bool = False` - Whether to ignore the blend quantities of this point.
        If True, then this method returns the index of this division point unmodified.
        
        Returns
        -------
        `int` - The modified index of this division point.
        """
        if ignore_blend:
            return self.index + 1
        return (self.index + 1) - self.blend.get_left(previous_problem_size)
    
    def index_when_right_point(self, next_problem_size: int = 0, /, *, ignore_blend: bool = False) -> int:
        """
        Get the index of this division point, modified to account for its blend quantities, when acting as the right point of a partial problem.
        
        Parameters
        ----------
        `next_problem_size : int = 0` - The normal size of the next in sequence partial problem template from the division scenario this point belongs to.
        Where the normal size is the size ignoring all blend quantities, and the size is equal to; the index the next point, minus the index of this point.
        
        `ignore_blend : bool = False` - Whether to ignore the blend quantities of this point.
        If True, then this method returns the index of this division point unmodified.
        
        Returns
        -------
        `int` - The modified index of this division point.
        """
        if ignore_blend:
            return self.index
        return self.index + self.blend.get_right(next_problem_size)

class DivisionPointPair(NamedTuple):
    left: Optional[DivisionPoint]
    right: Optional[DivisionPoint]

@dataclass(frozen=True, order=True)
class SubGoalRange(_collections_abc.Collection):
    first_index: int
    last_index: int
    
    def __len__(self) -> int:
        return (self.last_index - self.first_index) + 1
    
    def __iter__(self) -> Iterator[int]:
        for index in range(self.first_index, self.last_index + 1):
            yield index
    
    def __contains__(self, index: int) -> bool:
        return self.first_index <= index and index <= self.last_index
    
    @property
    def problem_size(self) -> int:
        return len(self)

class DivisionScenario(_collections_abc.Sequence):
    """
    A division scenario is a proactively generated sequence of division points, over a contiguous sub-sequence of sub-goal stages generated from an abstract level plan.
    It forms a template-like structure, showing how a combined problem is going to be divided into a sequence of partial refinement planning problems for the original model level.
    
    A scenario is proactively generated every time the planner changes down a level in the abstract hierarchy.
    The generated scenario divides the immediately previously generated plan at the previous level is divided by this scenario.
    
    It is done by containing a sequence of sub-goal stage sub-sequences, each of which is included in its own partial-problem.
    To generate a partial-problem, all that is needed is to obtain the satisfying state of the previous sub-goal stage, which then becomes the starting state of the partial problem.
    
    Properties
    ----------
    `divided_abstract_plan : MonolevelPlan` - The abstract monolevel plan that is divided by this scenario.
    
    `previously_solved_problems : int` - The number of partial problems that were solved before this scenario became applicable.
    
    `total_problems : int` - The number of partial problems templated by this division scenario, equivalent to one less than the number of divisions of the abstract plan made by the scenario.
    
    `problem_range : range` - The contiguous sub-sequence of partial problem numbers defined by this division scenario.
    """
    
    __slots__ = ("__abstract_plan",
                 "__division_points",
                 "__previously_solved_problems")
    
    def __init__(self,
                 abstract_plan: Optional["Planner.MonolevelPlan"],
                 division_points: Iterable[DivisionPoint],
                 previously_solved_problems: int = 0
                 ) -> None:
        """
        Create a divisio scenario from a sequence of division points,
        which blueprints a sequence of partial conformance refinement planning problems, over an abstract plan.
        
        Parameters
        ----------
        `abstract_plan: {Planner.MonolevelPlan | None}` - The abstract plan that the scenario divides the refinement problem of.
        This can be None, but it is recommended that it is always given.
        
        `division_points: Iterable[DivisionPoint]` - A sequence of division points that define the scenario.
        
        `previously_solved_problems: int = 0` - The number of partial refinement problems that have already been solved
        (or have already been blueprinted by an existing division scenario) at the original level.
        """
        if abstract_plan.plan_length <= 0:
            raise ValueError
        self.__abstract_plan: Planner.MonolevelPlan = abstract_plan
        ## The maximum number of division points that can be placed over an abstract plan is equal to one less than the plan length.
        if (len(division_points) + 1) > abstract_plan.plan_length:
            raise ValueError(f"Too many division points. Got; {len(division_points)} division points, over abstract plan of length {abstract_plan.plan_length}.")
        self.__division_points: list[DivisionPoint] = list(division_points)
        self.__division_points.sort()
        if previously_solved_problems < 0:
            raise ValueError(f"Number of previously solved problems must be greater than or equal to zero. Got; {previously_solved_problems}.")
        self.__previously_solved_problems: int = previously_solved_problems
    
    def __contains__(self, item: object) -> bool:
        if isinstance(item, int):
            for point in self:
                if point.index == item:
                    return True
        return super().__contains__(item)
    
    def __iter__(self) -> Iterator[DivisionPoint]:
        yield from self.__division_points
    
    def __getitem__(self, value: int) -> DivisionPoint:
        return self.__division_points[value]
    
    def __len__(self) -> int:
        return len(self.__division_points)
    
    def __str__(self) -> str:
        return (f"Total Templated Partial Problems = {self.total_problems}, Problem Range = [{min(self.problem_range)}-{max(self.problem_range)}], "
                f"Size = {self.size}, Sub-goal Stage Range = [{min(self.__abstract_plan.action_step_range)}-{max(self.__abstract_plan.action_step_range)}]\n"
                f"Division points [total={len(self)}] : [{', '.join(map(str, self))}]")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__abstract_plan!r}, {self.__division_points!r})"
    
    ####################
    ## Scenario properties
    
    @property
    def divided_abstract_plan(self) -> "Planner.MonolevelPlan":
        return self.__abstract_plan
    
    @property
    def previously_solved_problems(self) -> int:
        "The number of problems at the next level that were solve prior to generating this division scenario."
        return self.__previously_solved_problems
    
    @property
    def total_problems(self) -> int:
        "The total number of partial planning problems templated by this scenario."
        return self.get_total_divisions() + 1
    
    @property
    def problem_range(self) -> range:
        "The partial problem number range templated by this scenario."
        return range(self.__previously_solved_problems + 1,
                     self.__previously_solved_problems + self.total_problems + 1)
    
    @property
    def size(self) -> int:
        "The total number of sub-goal stages in the (sub-)sequence of sub-goal stages from the (partial-)plan divided by this scenario."
        return (self.last_index - self.first_index) + 1
    
    @property
    def first_index(self) -> int:
        "The first sub-goal stage index of the (sub-)sequence of sub-goal stages from the (partial-)plan divided by this scenario."
        return self.__abstract_plan.state_start_step + 1
    
    @property
    def last_index(self) -> int:
        "The last sub-goal stage index of the (sub-)sequence of sub-goal stages from the (partial-)plan divided by this scenario."
        return self.__abstract_plan.end_step
    
    ####################
    ## Problem divisions
    
    def get_total_divisions(self, shifting_only: bool = True) -> int:
        """
        The total number of problem division points contained in this division scenario.
        
        Parameters
        ----------
        `shifting_only : bool` - A Boolean defining, True to count only shifting divisions, False to also count continuous divisions.
        A shifting division is one that defines a seperate (partial) monolevel planning problem, when the last index of which is achieved yields its partial plan as safe to execute.
        A continuous division is one commited inside a monolevel planning problem but which does not interrupt it, and thus does not change its size.
        
        Returns
        -------
        `int` - An integer defining the total number of division points.
        """
        return sum(1 for point in self if not shifting_only or point.shifting)
    
    def get_division_points(self,
                            shifting_only: bool = True,
                            fabricate_inherited: bool = False
                            ) -> list[DivisionPoint]:
        """
        Get all the division points in this division scenario as an ordered list.
        
        Parameters
        ----------
        `shifting_only : bool = True` - True to include only problem shifting (proactive and interrupting reactive) divisions, False to also include non-shifting continuous reactive divisions.
        
        `fabricate_inherited : bool = False` - True to fabricate division points for the first and last boundary sub-goal stage indices of this scenario's sub-goal stage range.
        """
        division_points: list[DivisionPoint] = []
        
        if shifting_only:
            for point in self.__division_points:
                if point.shifting:
                    division_points.append(point)
        else: division_points = self.__division_points.copy()
        
        if not fabricate_inherited:
            return division_points
        
        return [DivisionPoint(self.first_index - 1, inherited=True)] + division_points + [DivisionPoint(self.last_index, inherited=True)]
    
    def get_division_point_pair(self,
                                problem_number: int,
                                fabricate_inherited: bool = False
                                ) -> DivisionPointPair:
        
        problem_range: range = self.problem_range
        
        ## Check the problem number is valid
        if problem_number not in problem_range:
            raise ValueError("The given problem number is not in the range of this division scenario. "
                             f"Got; problem number = {problem_number}, problem range = {self.problem_range}")
        
        left_point: Optional[DivisionPoint] = None
        right_point: Optional[DivisionPoint] = None
        
        if fabricate_inherited:
            left_point = DivisionPoint(self.first_index - 1, inherited=True)
            right_point = DivisionPoint(self.last_index, inherited=True)
        
        division_points: list[DivisionPoint] = self.get_division_points()
        if problem_number > min(problem_range):
            left_point = division_points[problem_number - min(problem_range) - 1]
        if problem_number < max(problem_range):
            right_point = division_points[problem_number - min(problem_range)]
        
        return DivisionPointPair(left_point, right_point)
    
    def get_subgoals_indices_range(self,
                                   problem_number: int,
                                   ignore_blend: bool = False
                                   ) -> SubGoalRange:
        
        problem_range: range = self.problem_range
        
        ## Check the problem number is valid
        if problem_number not in problem_range:
            raise ValueError("The given problem number is not in the range of this division scenario. "
                             f"Got; problem number = {problem_number}, problem range = {self.problem_range}")
        
        ## Find the sub-goals stage index range
        ##      - If the problem is the first;
        ##          - The first sgoal is the starting one of the divided abstract plan,
        ##          - Otherwise, the first sgoal is the left blend of the previous division point (exclusive).
        ##      - If the problem is the last;
        ##          - The last sgoal is the ending one of the divided abstract plan,
        ##          - Otherwise, the last sgoal is the right blend of the next division point (inclusive).
        first_index: int = self.first_index
        last_index: int = self.last_index
        
        division_points: list[DivisionPoint] = self.get_division_points()
        
        if problem_number > min(problem_range):
            left_point: DivisionPoint = division_points[problem_number - min(problem_range) - 1]
            if ignore_blend:
                first_index = left_point.index_when_left_point(ignore_blend=True)
            else: first_index = left_point.index_when_left_point(self.get_subgoals_indices_range(problem_number - 1, ignore_blend=True).problem_size)
        
        if problem_number < max(problem_range):
            right_point: DivisionPoint = division_points[problem_number - min(problem_range)]
            if ignore_blend:
                last_index = right_point.index_when_right_point(ignore_blend=True)
            else: last_index = right_point.index_when_right_point(self.get_subgoals_indices_range(problem_number + 1, ignore_blend=True).problem_size)
        
        return SubGoalRange(first_index, last_index)
    
    def update_reactively(self, new_point: DivisionPoint) -> None:
        """
        Update this division scenario with a reactive division point.
        The blend quantities of the scenario are updated such that the left blend of the left point does not cross the reactive division point.
        If the reactive division was interrupting and the point is inside the right blend of the right point of the requested problem, then the reactive division becomes the left point of the next problem and that next problem no longer has a blend (or the blend is updated and shifted onto the reactive division??).
        """
        if new_point.reactive is None:
            raise ValueError(f"Division point must be reactive. Got; {new_point!s}.")
        
        if new_point.blend != Blend():
            raise ValueError(f"Cannot insert reactive divisions with non-zero blend quantities. Got; {new_point!s}.")
        
        if new_point.index in (point.index for point in self.__division_points):
            raise ValueError(f"Duplicate division point inserted. Got; {new_point!s} which matches with {self.__division_points[self.__division_points.index(new_point)]!s}.")
        
        ## Iterate through the current list of division points until one is reached whose true division index is greater than the index of the new reactive division point;
        ##      - Variable `sequence_index` is the sequence index value of the division point in the list of points itself and is integral to the ordering of those points.
        for sequence_index, point in enumerate(self.get_division_points(shifting_only=False, fabricate_inherited=True)):
            
            if new_point.index < point.index:
                self.__division_points.insert(sequence_index, new_point)
                return
        
        raise ValueError(f"Division point {new_point} is not in the range of scenario: {self}.")

@dataclass(frozen=True)
class Reaction:
    divide: bool = False
    interrupt: bool = False
    backwards_horizon: Number = 0
    rationale: Optional[str] = None
    
    def __str__(self) -> str:
        return f"(Divide = {self.divide}, Interrupt = {self.interrupt}, Backwards Horizon = {self.backwards_horizon}, Rationale = {self.rationale})"

class DivisionStrategy(metaclass=ABCMeta):
    """
    Base class for division strategies.
    The class can be instantiated directly, but will only generate division scenarios with no divisions, i.e. it will not actually divide problems.
    The designer should use one of its standard sub-classes, as stored in the enum GetStrategy, or make their own custom strategy by sub-classing and overriding the abstract methods of this class.
    
    Proactive strategies can selectively choose to ignore the blend quantity if it deems the chance a division being inadmissible is sufficiently low.
    
    A reactive or adaptive strategy uses its react method to decide where to put divisions during solving:
        - reactive makes no division proactively and decides entirely based on the feedback function,
            - reactive also allows dynamic modification of the blend quantity based on the measured complexity of the combined sub-problems,
        - adaptive makes a set of divisions pro-actively and reactively adds additional divisions reactively if necessary.
    
    Public Fields
    -------------
    `knowledge : Any` - Any object or primitive data the designer wishes to attach to the strategy, containing knowledge used to inform and make decisions.
    
    Properties
    ----------
    `bounds : dict[str, {int | float}] (assignable)` - The validated bounds given to this strategy.
    Stored as a dictionary, whose keys are strings, and whose values are either integers of floating point numbers.
    
    `blend_quantity : int (assignable)` - An integer defining the blend quantity.
    This is the number of sub-goal stages that each partial problem overlaps with the next.
    Each partial problem increases in size by double the blend quantity, except the first and last partial problems.
    The first only has a right blend, whereas the last only has a left blend.
    This is because the first and the last problems either sit against inherited division points, or are initial or final.
    
    Notes
    -----
    This class declares its fields via slots.
    
    The abstract version of the proact method deals with
    """
    
    __slots__ = ("knowledge",           # Any
                 "__bounds",            # dict[str, {Number | dict[int, Number]}]
                 "__blend_quantities",  # {Blend | dict[int, Blend]}
                 "__top_level",         # int
                 "__inde_tasks",        # bool
                 "__order_tasks")       # bool
    
    def __init__(self,
                 top_level: int,
                 bounds: Bounds,
                 blend: Union[Number, Blend, dict[int, Union[Number, Blend]]] = {},
                 knowledge: Any = None,
                 independent_tasks: bool = False,
                 order_tasks: bool = False
                 ) -> None:
        """
        Abstract super constructor for division strategies.
        
        Parameters
        ----------
        
        """
        
        self.__top_level = top_level
        self.__inde_tasks = independent_tasks
        self.__order_tasks = order_tasks
        
        ## The strategy's bounds;
        ##      - Maps = bound_name: str x ?(level: int) -> bound_value: {int | float}
        self.__bounds: Bounds = {}
        self.bounds = bounds
        
        ## The blend quantities;
        ##      - Maps = ?(level: int) -> blend_quantity: Blend
        self.__blend_quantities: Union[Blend, dict[int, Blend]] = {}
        self.blend_quantities = blend
        
        ## An arbitrary object used to store any desired knowledge the designer wishes to attach to this strategy
        self.knowledge: Any = knowledge
    
    ####################
    ## Properties
    
    @property
    def bounds(self) -> dict[str, dict[int, Number]]:
        return self.__bounds
    
    @bounds.setter
    def bounds(self, bounds: Bounds) -> None:
        _bounds = self.validate_bounds(bounds)
        if _bounds is not None:
            self.__bounds = _bounds
        else: self.__bounds = bounds
    
    def get_bound(self, bound: str, level: int = 0, default: Optional[Number] = None) -> Number:
        if isinstance(_bound := self.__bounds.get(bound, {}), dict):
            if (_bound := _bound.get(level, default)) is not None:
                return _bound
            elif default is not None:
                return default
            else: raise ValueError(f"The bound {bound} does not exist at level {level} and no default was given.")
        else: return _bound
    
    def set_bound(self, bound: str, level: int, value: Number) -> None:
        self.__bounds.setdefault(bound, {})[level] = value
        self.bounds = self.__bounds
    
    @property
    def blend_quantities(self) -> Union[Blend, dict[int, Blend]]:
        return self.__blend_quantities
    
    @blend_quantities.setter
    def blend_quantities(self, blend_quantities: Union[Number, Blend, dict[int, Union[Number, Blend]]]) -> None:
        "Safely sets the blend quantities of this divisions strategy."
        
        def as_blend(value: Union[Number, Blend]) -> Blend:
            if isinstance(value, (int, float)):
                return Blend(value, value)
            elif isinstance(value, Blend):
                return value
            else: raise ValueError(f"Blend quantities must be either; a float, integer, or pre-constructed Blend object. Got; {value} of type {type(value)}.")
        
        if isinstance(blend_quantities, dict):
            for level in blend_quantities:
                self.__blend_quantities[level] = as_blend(blend_quantities[level])
        else: self.__blend_quantities = as_blend(blend_quantities)
    
    def get_blend(self, level: int = 0) -> Blend:
        if isinstance(self.__blend_quantities, dict):
            return self.__blend_quantities.get(level, Blend())
        else: return self.__blend_quantities
    
    def set_blend(self, level: int, left: Number, right: Number) -> Blend:
        self.__blend_quantities[level] = Blend(left, right)
    
    ####################
    ## Bounds
    
    @classmethod
    def validate_bounds(cls, bounds: Bounds) -> Optional[Bounds]:
        return None
    
    @classmethod
    def default_bounds(cls) -> Bounds:
        return {}
    
    ####################
    ## Decision making
    
    @abstractmethod
    def proact(self, abstract_plan: "Planner.MonolevelPlan", previously_solved_problems: int = 0) -> DivisionScenario:
        """
        Abstract base version of proactive division, must be overridden by inheriting classes.
        
        If the level is the specified top level
        """
        division_indices: list[int] = []
        divisions: list[DivisionPoint] = []
        
        ## If the plan's level is the top-level of the domain's hierarchy and was generated in the tasking model...
        if (abstract_plan.level == self.__top_level
            and abstract_plan.model_type == Planner.DomainModelType.Tasking):
            
            ## Treat tasks as independent by making every possible division;
            ##      - Takes presidence over dividing based on the fgoal achievement order,
            ##      - This is regardless of whether the fgoals where actually ordered or not.
            if self.__inde_tasks:
                division_indices = list(range(2, abstract_plan.plan_length))
            
            ## Divides based on the actual achievement steps, regardless of what the preference order actually was, or if there was one at all.
            elif self.__order_tasks:
                for step in abstract_plan.fgoals_achieved_at:
                    if 1 < step < abstract_plan.end_step:
                        division_indices.append(step)
            
            divisions: list[DivisionPoint] = self.make_heterogenous_divisions(division_indices,
                                                                              abstract_plan.plan_length,
                                                                              abstract_plan.state_start_step,
                                                                              self.get_blend(abstract_plan.level))
        
        return DivisionScenario(abstract_plan, divisions, previously_solved_problems)
    
    def react(self,
              problem_level: int,
              problem_total_sgoals_range: SubGoalRange,
              problem_start_step: int,
              current_search_length: int,
              current_subgoal_index: int,
              matching_child: bool,
              incremental_times: list[float],
              observable_plan: Optional[dict[int, list["Planner.Action"]]]
              ) -> Reaction:
        return Reaction()
    
    def reset(self) -> None:
        pass
    
    def total_increments_prediction(self, planning_domain: "Planner.PlanningDomain", online_method: "Planner.OnlineMethod") -> Optional[int]:
        return None
    
    @staticmethod
    def make_homogenous_divisions(partial_problems: int,
                                  plan_length: int,
                                  start_step: int = 0,
                                  blend: Blend = Blend()
                                  ) -> list[DivisionPoint]:
        """
        Make a sequence of homogenous division points, to define (at most) a given number of partial problems, for refining an abstract plan of a given length.
        
        For a plan of length `l`, `d` division points are generated, dividing the plan refinement problem into a sequence of exactly `p = d + 1` partial refinement problems.
        The sequence contains exactly `p - (l % p)` 'small' partial problems of size `(l // p)`, and `(l % p)`` 'large' partial problems of size `(l // p) + 1`.
        The sequence is biased such that, all small problems occur first in the seqeunce, and the large follow.
        
        Parameters
        ----------
        `partial_problems : int` - The number of partial problems to divide the given plan length into.
        
        `plan_length : int` - The length of the abstract plan being divided.
        
        `start_step : int = 0` - The step that the abstract plan being divided starts upon, this is used to calculate the absolute division point indices.
        If not given or zero, and the plan being divided is assumed to be initial (starts from step zero), the division points indices are relative to the initial step.
        
        Returns
        -------
        `list[DivisionPoint]` - A list of division points defining a homogenous division sequence over the given abstract plan length.
        
        Raises
        ------
        `ValueError` - If either;
            - The number of partial problems is less than one, or greater than the plan length,
            - The plan length is less than one,
            - The start step is less than zero.
        
        Example Usage
        -------------
        Divide a figurative abstract plan of length 10 into 4 homogenous parts without blending.
        
        ```
        >>> from ASH.core.Strategies import Proactive, DivisionScenario
        ## Make 4 partial problems
        >>> division_points: list[DivisionPoint] = Proactive.make_homogenous_divisions(partial_problems=4, plan_length=10)
        ## Giving exactly d = 3 = p - 1 = 4 - 1 division points
        >>> division_points
        [(index=2), (index=4), (index=7)]
        
        ## The indices of the partial problems are;
        >>> scenario = DivisionScenario.test_scenario(division_points, plan_length=10)
        >>> for problem in range (1, 4 + 1):
        >>>     print(f"Problem [{problem}]: {scenario.get_subgoals_indices_range(problem)!s}")
        >>> Problem [1]: (Indices = [1, 2], Size = 2)
        >>> Problem [2]: (Indices = [3, 4], Size = 2)
        >>> Problem [3]: (Indices = [5, 7], Size = 3)
        >>> Problem [4]: (Indices = [8, 10], Size = 3)
        ```
        
        The division scenario diagram looks like this;
        
        TODO
        
        Divide a figurative abstract plan of length 10 into 4 homogenous parts, this time with a 50% balanced blend.
        
        TODO
        """
        _Strategies_logger.debug(f"Making homogenous divisions: {partial_problems=}, {plan_length=}, {start_step=}, {blend=}")
        
        if plan_length < 1:
            raise ValueError(f"Invalid plan length {plan_length}, must be greater than or equal to 1.")
        
        if start_step < 0:
            raise ValueError(f"Invalid start step {start_step}, must be greater than or equal to 0.")
        
        if partial_problems < 1 or partial_problems > plan_length:
            raise ValueError(f"Invalid number of partial problems {partial_problems}, must be greater than or equal to 1 and less than or equal to plan length ({plan_length}).")
        
        ## Determine the number of small and large problems
        number_small_problems: int = partial_problems - (plan_length % partial_problems)
        number_large_problems: int = (plan_length % partial_problems)
        
        ## Determine the size of each problem
        small_group_size: int = plan_length//partial_problems
        large_group_size: int = small_group_size + 1
        
        _Strategies_logger.debug(f"Decided: {number_small_problems=}, {number_large_problems=}, {small_group_size=}, {large_group_size=}")
        
        def get_size(division_number: int) -> int:
            "Inner function for getting the size of individual partial problems."
            if division_number == 0:
                return 0
            elif division_number <= number_small_problems:
                return small_group_size
            else: return large_group_size
        
        ## Loop variables
        division_points: list[DivisionPoint] = []
        current_index: int = start_step
        
        ## There are a number of divisions equal to one less than the number of partial problems
        for division_number in range(1, partial_problems):
            ## Determine the sgoal range limits either side of the current division
            prev_index: int = current_index
            current_index += get_size(division_number)
            next_index: int = current_index + get_size(division_number + 1)
            
            ## Determine the blend quantities either side of the current division
            left_blend: int = current_index - max(current_index - blend.get_left(current_index - prev_index), prev_index)
            right_blend: int = min(current_index + blend.get_right(next_index - current_index), next_index) - current_index
            
            ## Add the point division to the list
            division_points.append(DivisionPoint(current_index, Blend(left_blend, right_blend)))
        
        _Strategies_logger.debug(f"Division points generated:\n{division_points}")
        
        return division_points
    
    @staticmethod
    def make_heterogenous_divisions(division_points: Iterable[int],
                                    plan_length: int,
                                    start_step: int = 0,
                                    blend: Blend = Blend()
                                    ) -> list[DivisionPoint]:
        
        if plan_length < 1:
            raise ValueError(f"Plan length must be an integer great than zero. Got; {plan_length=}.")
        
        if start_step < 0:
            raise ValueError(f"Start step must be an integer great than or equal to zero. Got; {start_step=}.")
        
        if any(plan_length <= division_point <= (start_step + 1) for division_point in division_points):
            raise ValueError(f"A division point index is invalid, must be in the range [{start_step+1}-{plan_length}]")
        
        _Strategies_logger.debug(f"Making heterogenous divisions: {division_points=}, {plan_length=}, {start_step=}, {blend=}")
        
        def get_size(division_number: int) -> int:
            "Inner function for getting the size of individual partial problems."
            if division_number == 0:
                return 0
            previous: int = start_step
            if division_number > 1:
                previous = division_points[division_number - 2]
            return previous - division_points[division_number - 1]
        
        ## Loop variables
        division_points: list[DivisionPoint] = []
        current_index: int = start_step
        
        ## There are a number of divisions equal to one less than the number of partial problems
        for division_number in range(1, len(division_points)):
            ## Determine the sgoal range limits either side of the current division
            prev_index: int = current_index
            current_index += get_size(division_number)
            next_index: int = current_index + get_size(division_number + 1)
            
            ## Determine the blend quantities either side of the current division
            left_blend: int = current_index - max(current_index - blend.get_left(current_index - prev_index), prev_index)
            right_blend: int = min(current_index + blend.get_right(next_index - current_index), next_index) - current_index
            
            ## Add the point division to the list
            division_points.append(DivisionPoint(current_index, Blend(left_blend, right_blend)))
        
        _Strategies_logger.debug(f"Division points generated:\n{division_points}")
        
        return division_points

@final
class Basic(DivisionStrategy):
    """
    The most basic strategy for dividing planning problems.
    It divides a planning problem into a given number of partial planning problems.
    The minimum of either; the given number of partial problems and the number of sub-goal stages being divided.
    This strategy divides a sequenece of abstract sub-goal stages into a sequence of sub-sequences.
    """
    def __init__(self,
                 top_level: int,
                 problems: Optional[Union[int, dict[int, int]]],
                 blend: Union[Number, Blend, dict[int, Union[Number, Blend]]] = {},
                 independent_tasks: bool = False,
                 order_tasks: bool = False
                 ) -> None:
        
        if (_problems := problems) is None:
            _problems = self.__class__.default_bounds()["problems"]
        super().__init__(top_level=top_level,
                         bounds={"problems" : _problems},
                         blend=blend,
                         independent_tasks=independent_tasks,
                         order_tasks=order_tasks)
    
    def proact(self,
               abstract_plan: "Planner.MonolevelPlan",
               previously_solved_problems: int
               ) -> DivisionScenario:
        
        ## Use any available tasking based division scenario
        if (scenario := super().proact(abstract_plan)):
            return scenario
        
        ## Obtain valid number of partial problems to generate;
        ##      - Number of divisions is one greater than the number of problems.
        problems: int = self.get_bound("problems", abstract_plan.level, default=1)
        if problems > abstract_plan.plan_length:
            problems = abstract_plan.plan_length
        
        ## Make a sequence of homogenous divisions to template the given number of equal size partial problems
        divisions: list[DivisionPoint] = self.make_homogenous_divisions(problems,
                                                                        abstract_plan.plan_length,
                                                                        abstract_plan.state_start_step,
                                                                        self.get_blend(abstract_plan.level))
        return DivisionScenario(abstract_plan, divisions, previously_solved_problems)
    
    @classmethod
    def validate_bounds(cls, bounds: Bounds) -> Optional[Bounds]:
        return None ## TODO
    
    @classmethod
    def default_bounds(cls) -> Bounds:
        return {"problems" : 2}
    
    def total_increments_prediction(self,
                                    planning_domain: "Planner.PlanningDomain",
                                    online_method: "Planner.OnlineMethod"
                                    ) -> Optional[int]:
        level_range: range = planning_domain.constrained_level_range(bottom_level=2)
        
        if online_method == Planner.OnlineMethod.GroundFirst:
            return int(math.prod(self.get_bound("problems", level, 1) for level in level_range))
        
        elif online_method == Planner.OnlineMethod.CompleteFirst:
            return int(sum(self.get_bound("problems", level, 1) for level in level_range))
        
        elif online_method == Planner.OnlineMethod.Hybrid:
            return int(sum(self.get_bound("problems", level, 1) for level in level_range)) - (max(level_range) - 1)
        
        return None

class NaiveProactive(DivisionStrategy):
    
    def __init__(self,
                 top_level: int,
                 size_bound: Optional[Bound] = None,
                 blend: dict[int, Union[Number, Blend]] = {},
                 independent_tasks: bool = False,
                 order_tasks: bool = False
                 ) -> None:
        
        if (_size_bound := size_bound) is None:
            _size_bound = self.__class__.default_bounds()["size_bound"]
        super().__init__(top_level=top_level,
                         bounds={"size_bound" : _size_bound},
                         blend=blend,
                         independent_tasks=independent_tasks,
                         order_tasks=order_tasks)
    
    @final
    def react(self, problem_level: int, problem_total_sgoals_range: SubGoalRange, problem_start_step: int, current_search_length: int, current_subgoal_index: int, matching_child: bool, incremental_times: list[float], observable_plan: Optional[dict[int, list["Planner.Action"]]]) -> Reaction:
        return super().react(problem_level, problem_total_sgoals_range, problem_start_step, current_search_length, current_subgoal_index, matching_child, incremental_times, observable_plan)
    
    def total_increments_prediction(self, planning_domain: "Planner.PlanningDomain", online_method: "Planner.OnlineMethod") -> Optional[int]:
        if online_method == Planner.OnlineMethod.GroundFirst:
            return int(math.prod(1/self.get_bound("size_bound", level, 1.0) for level in reversed(range(2, planning_domain.top_level + 1)) if isinstance(self.get_bound("size_bound", level, 1.0), float)))
        
        elif online_method == Planner.OnlineMethod.CompleteFirst:
            return int(sum(1/self.get_bound("size_bound", level, 1.0) for level in range(2, planning_domain.top_level + 1) if isinstance(self.get_bound("size_bound", level, 1.0), float)))
        
        elif online_method == Planner.OnlineMethod.Hybrid:
            return None ## TODO
        
        return None
    
    @final
    def get_true_size_bound(self, level: int, plan_length: int) -> int:
        """
        Get the true size bound of this strategy, given the abstraction level and length of an abstract plan being divided.
        
        The true size bound is normalised against the given plan length, such that is it equal to the maximum of;
        one, and the minimum of; the strategy's size bound (multiplied by the plan length if it is a float), and the plan length.
        
        An abstract plan divided by the true size bound will always produce a valid division scenario that defines at least one partial problem.
        
        Parameters
        ----------
        `level : int` - The abstraction level of the plan that is being divided by the strategy as an integer.
        
        `plan_length : int` - The length of the plan that is being divided by the strategy as an integer.
        
        Returns
        -------
        `int` - The true size bound as an integer (greater than zero), whose value is normalised against the given plan length.
        
        Raises
        ------
        `TypeError` - If the plan length is not an integer.
        
        `ValueError` - If the plan length is less than or equal to zero.
        """
        if not isinstance(plan_length, int):
            raise TypeError(f"Plan length must be an integer. Got {plan_length} of type {type(plan_length)}.")
        if plan_length <= 0:
            raise ValueError(f"Plan length must be greater than zero. Got {plan_length}.")
        
        ## Find the size bound for the given level
        size_bound: Number = self.get_bound("size_bound", level, plan_length)
        
        ## Normalise the size bound on the plan length;
        ##      - The bound cannot be less than 1,
        ##      - The bound cannot be more than the plan length.
        true_size_bound: int
        if isinstance(size_bound, float):
            true_size_bound = round(plan_length * size_bound)
        else: true_size_bound = size_bound
        true_size_bound = min(plan_length, max(1, true_size_bound))
        
        return true_size_bound
    
    @classmethod
    def validate_bounds(cls, bounds: Bounds) -> Optional[Bounds]:
        size_bounds = bounds["size_bound"]
        if isinstance(size_bounds, dict):
            size_bounds = list(size_bounds.values())
        else: size_bounds = [size_bounds]
        for size_bound in size_bounds:
            if size_bound is None:
                continue
            if not isinstance(size_bound, (int, float)):
                raise TypeError(f"Naive proactive division strategies must have a integer or floating point size bound value. Got {size_bound} of type {type(size_bound)}.")
            if isinstance(size_bound, int) and size_bound <= 0:
                raise ValueError(f"An integer size bound value for naive proactive strategies must be greater than zero. Got {size_bound}.")
            if isinstance(size_bound, float) and not (0.0 < size_bound <= 1.0):
                raise ValueError(f"A floating point size bound value for naive proactive strategies must be in the range (0.0-1.0] Got {size_bound}.")
        return None

class Hasty(NaiveProactive):
    """
    The hasty naive proactive homogenous division strategy.
    Hasty favours planning speed, by keeping the size of partial problems below a maximum size bound.
    
    Notes
    -----
    The default bound size for this strategy is `0.25`, this attempts to divide each abstract plan into four equally sized parts, every time the planner descends a level in the abstract hierarchy.
    """
    
    def proact(self, abstract_plan: "Planner.MonolevelPlan", previously_solved_problems: int = 0) -> DivisionScenario:
        _Strategies_logger.debug(f"Proactively dividing plan: {abstract_plan}.")
        
        ## Use any available tasking based division scenario
        if (scenario := super().proact(abstract_plan)):
            return scenario
        
        ## Get the true size bound
        plan_length: int = abstract_plan.plan_length
        true_size_bound: int = self.get_true_size_bound(abstract_plan.level, plan_length)
        
        ## Determine number of partial problems;
        ##      - round up the number of partial problems,
        ##      - ensuring each stays smaller the maximum size bound.
        partial_problems: int = math.ceil(plan_length/true_size_bound)
        
        _Strategies_logger.debug(f"{plan_length=}, {true_size_bound=}, {partial_problems=}")
        
        return DivisionScenario(abstract_plan,
                                self.make_homogenous_divisions(partial_problems,
                                                               plan_length,
                                                               abstract_plan.state_start_step,
                                                               self.get_blend(abstract_plan.level)),
                                previously_solved_problems)
    
    @classmethod
    def default_bounds(cls) -> dict[str, Number]:
        return {"size_bound" : 0.25}

class Steady(NaiveProactive):
    """
    The steady naive proactive homogenous division strategy.
    Steady favours plan quality, by keeping the size of partial problems above a minimum size bound.
    
    Notes
    -----
    The default bound size for this strategy is `0.45`, this attempts to divide each abstract plan in half, every time the planner descends a level in the abstract hierarchy.
    """
    
    def proact(self,
               abstract_plan: "Planner.MonolevelPlan",
               previously_solved_problems: int = 0
               ) -> DivisionScenario:
        
        _Strategies_logger.debug(f"Proactively dividing plan: {abstract_plan}.")
        
        ## Use any available tasking based division scenario
        if (scenario := super().proact(abstract_plan)):
            return scenario
        
        ## Get the size bound
        plan_length: int = len(abstract_plan)
        true_size_bound: int = self.get_true_size_bound(abstract_plan.level, plan_length)
        
        ## Determine number of partial problems;
        ##      - round down the number of partial problems,
        ##      - ensuring each stays bigger than the minimum size bound.
        partial_problems: int = math.floor(plan_length/true_size_bound)
        
        _Strategies_logger.debug(f"{plan_length=}, {true_size_bound=}, {partial_problems=}")
        
        return DivisionScenario(abstract_plan,
                                self.make_homogenous_divisions(partial_problems,
                                                               plan_length,
                                                               abstract_plan.state_start_step,
                                                               self.get_blend(abstract_plan.level)),
                                previously_solved_problems)
    
    @classmethod
    def default_bounds(cls) -> dict[str, Number]:
        return {"size_bound" : 0.45}

class Jumpy(NaiveProactive):
    """
    The jumpy naive proactive heterogenous division strategy.
    Jumpy favours low execution latency and plan quality, by making one division only such that the size of the first partial problem is kept below a minimum size bound.
    
    Notes
    -----
    The default bound size for this strategy is `0.25`, this attempts to divide each abstract plan such that first partial problem is one quarter of the size of the complete problem, and the second is three-quarters of the size.
    """
    
    def proact(self,
               abstract_plan: "Planner.MonolevelPlan",
               previously_solved_problems: int = 0
               ) -> DivisionScenario:
        
        _Strategies_logger.debug(f"Proactively dividing plan: {abstract_plan}.")
        
        ## Use any available tasking based division scenario
        if (scenario := super().proact(abstract_plan)):
            return scenario
        
        ## Get the size bound
        plan_length: int = len(abstract_plan)
        if plan_length == 1: return DivisionScenario(abstract_plan, [], previously_solved_problems)
        true_size_bound: int = self.get_true_size_bound(abstract_plan.level, plan_length)
        return DivisionScenario(abstract_plan, [DivisionPoint(true_size_bound + abstract_plan.state_start_step)], previously_solved_problems)
    
    def total_increments_prediction(self,
                                    planning_domain: "Planner.PlanningDomain",
                                    online_method: "Planner.OnlineMethod"
                                    ) -> Optional[int]:
        return math.prod(2 for level in range(2, planning_domain.top_level + 1) if self.get_bound("size_bound", level, -1) != 1)
    
    @classmethod
    def default_bounds(cls) -> dict[str, Number]:
        return {"size_bound" : 0.25}
    
    @classmethod
    def validate_bounds(cls, bounds: Bounds) -> Optional[Bounds]:
        size_bounds = bounds["size_bound"]
        if isinstance(size_bounds, dict):
            size_bounds = list(size_bounds.values())
        else: size_bounds = [size_bounds]
        for size_bound in size_bounds:
            if size_bound is None:
                continue
            if size_bound > 0.5:
                raise ValueError("Jumpy requires a size bound of 0.5 or less.")
        return super().validate_bounds(bounds)

class ReactiveBoundType(enum.Enum):
    """
    `SearchLength = "search_length"` - The current search length.
    
    `Incremental = "incremental_time_bound"` - The total incremental planning time (seconds/step), averaged over the moving range.
    
    `Differential = "differential_time_bound"` - The rate of change (increase) in the total incremental planning time (seconds/step/step), averaged over the moving range.
    
    `Integral = "integral_time_bound"` - The sum of the total incremental planning times (seconds) over the moving range.
    
    `Cumulative = "cumulative_time_bound"` - The sum of all total incremental planning times (seconds) since the last reactive division.
    """
    SearchLength = "search_length_bound"
    Incremental = "incremental_time_bound"
    Differential = "differential_time_bound"
    Integral = "integral_time_bound"
    Cumulative = "cumulative_time_bound"

class Reactive(DivisionStrategy):
    """
    Base class for reactive division strategies.
    The class contains built-in support for calcuating times for the four standard type of reactive division strategy bounds.
    Sub-classes must override the react(...) method to make reactive divisions.
    """
    
    __slots__ = ("__last_division_index",   # dict[int, int]   || maps: level -> division point
                 "__last_division_step",    # dict[int, int]   || maps: level -> division step
                 "__backwards_horizon",     # dict[int, Number]
                 "__moving_average",        # int
                 "__preemptive",            # bool
                 "__interrupting")          # bool
    
    def __init__(self,
                 top_level: int,
                 search_length_bound: Bound = None,
                 incremental_time_bound: Bound = None,
                 differential_time_bound: Bound = None, 
                 integral_time_bound: Bound = None,
                 cumulative_time_bound: Bound = None,
                 backwards_horizon: Horizon = 0,
                 moving_average: Optional[int] = None,
                 preemptive: bool = True,
                 interrupting: bool = False,
                 knowledge: Any = None,
                 independent_tasks: bool = False,
                 order_tasks: bool = False
                 ) -> None:
        
        ## Three standard types of reactive division strategy bounds
        bounds: Bounds = {}
        default_bounds: Bounds = self.__class__.default_bounds()
        def assign(bound_type: ReactiveBoundType, bound: Bound):
            if bound is not None:
                bounds[bound_type.value] = bound
            elif default_bound := default_bounds.get(bound_type.value) is not None:
                bounds[bound_type.value] = default_bound
        assign(ReactiveBoundType.SearchLength, search_length_bound)
        assign(ReactiveBoundType.Incremental, incremental_time_bound)
        assign(ReactiveBoundType.Differential, differential_time_bound)
        assign(ReactiveBoundType.Integral, integral_time_bound)
        assign(ReactiveBoundType.Cumulative, cumulative_time_bound)
        
        ## Optional parameters
        self.__backwards_horizon: Union[Number, dict[int, Number]] = {}
        self.backwards_horizon = backwards_horizon
        self.__moving_average: int = 0
        self.moving_average = moving_average
        self.__preemptive: bool = True
        self.preemptive = preemptive
        self.__interrupting: bool = False
        self.interrupting = interrupting
        
        ## Variables for storing when previous reactive division was made:
        ##      - Need these because reactive divisions are made in the monolevel plan algorithm.
        self.__last_division_index: dict[int, int] = {}
        self.__last_division_step: dict[int, int] = {}
        
        super().__init__(top_level=top_level,
                         bounds=bounds,
                         knowledge=knowledge,
                         independent_tasks=independent_tasks,
                         order_tasks=order_tasks)
    
    @property
    def backwards_horizon(self) -> Horizon:
        return self.__backwards_horizon
    
    @backwards_horizon.setter
    def backwards_horizon(self, horizon: Horizon) -> None:
        "Safely sets the backwards horizon of this divisions strategy."
        
        def as_horizon(value: Optional[Number]) -> Number:
            if value is None:
                return 0
            if not isinstance(value, (int, float)):
                raise TypeError(f"Backwards horizon must be an integer or floating point number. Got; {value} of type {type(value)}.")
            if isinstance(value, int) and value < 0:
                raise ValueError(f"An integer backwards horizon must be greater than or equal to zero. Got; {value}.")
            if isinstance(value, float) and not (0.0 <= value <= 1.0):
                raise ValueError(f"A floating point backwards horizon value must be in the range [0.0-1.0]. Got; {value}.")
            return value
        
        if isinstance(horizon, dict):
            for level in horizon:
                self.__backwards_horizon[level] = as_horizon(horizon[level])
        else: self.__backwards_horizon = as_horizon(horizon)
    
    def get_horizon(self, level: int = 0) -> Number:
        if isinstance(self.__backwards_horizon, dict):
            return self.__backwards_horizon.get(level, 0)
        else: return self.__backwards_horizon
    
    @property
    def moving_average(self) -> Optional[int]:
        return self.__moving_average
    
    @moving_average.setter
    def moving_average(self, value: Optional[int]) -> None:
        if value is not None:
            if not isinstance(value, int):
                raise TypeError(f"Moving average must be an integer greater than zero. Got; {value} of type {type(value)}.")
            if value < 1:
                raise ValueError(f"Moving average must be an integer greater than zero. Got; {value} of type {type(value)}.")
        self.__moving_average: Optional[int] = value
    
    def calculate(self,
                  problem_level: int,
                  start_step: int,
                  search_length: int,
                  bound_type: ReactiveBoundType,
                  incremental_times: list[float],
                  valid_only: bool = True
                  ) -> float:
        """
        Calculate the current time value over a list of incremental times for any of the standard reactive bound types.
        
        Parameters
        ----------
        `problem_level : int` - The abstraction level of the current planning problem.
        
        `start_step : int` - The start step of the current planning problem.
        
        `search_length : int` - 
        
        `bound_type : ReactiveBoundType` - The reactive bound type to calculate the time for (see ReactiveBoundType).
        
        `incremental_times : list[float]` - A list of total incremental planning times, starting from the first.
        
        `valid_onle : bool` - A Boolean, True to use only incremental times since the last division step in the calculation (the valid times), otherwise False to use all times greater than or equal to the start step.
        
        Returns
        -------
        `float` - The calcuated time.
        
        Raises
        ------
        `TypeError` - If the bound type is not of the correct type.
        """
        if not isinstance(bound_type, ReactiveBoundType):
            raise TypeError(f"Invalid bound type. Got; {bound_type} of type {type(bound_type)}.")
        
        true_last_division_step: int = max(self.get_last_division_step(problem_level), start_step)
        
        if bound_type == ReactiveBoundType.SearchLength:
            return search_length - true_last_division_step
        
        ## The valid times are all the total incremental planning times since the last division step
        valid_times: list[float] = incremental_times
        if valid_only: valid_times = incremental_times[true_last_division_step - start_step:]
        
        if valid_times:
            ## The usable times are either;
            ##      - Those in the moving range if the range is given,
            ##      - Otherwise all the valid times.
            usable_times: list[float]
            if self.moving_average is not None:
                range_ = range(min(self.moving_average, len(valid_times)))
                usable_times = [valid_times[index] for index in range_]
            usable_times = valid_times
            
            _Strategies_logger.debug(f"Incremental times: {incremental_times}")
            _Strategies_logger.debug(f"Valid times: {valid_times}")
            _Strategies_logger.debug(f"Usable times: {usable_times}")
            
            ## The incremental is simply the moving average over a sub-sequence of the incremental search step times
            if bound_type == ReactiveBoundType.Incremental:
                return statistics.mean(usable_times)
            
            ## The integral is simply the sum of the incremental search times to determine the cumulative search time since the last division
            if bound_type == ReactiveBoundType.Integral:
                return sum(usable_times)
            
            if bound_type == ReactiveBoundType.Cumulative:
                return sum(valid_times)
            
            ## At least two search steps are needed to calculate the differential
            if len(usable_times) < 2: return 0.0
            
            ## Calculate the gradient between each consecutive search step and the average rate of change between them
            gradients: list[float] = [usable_times[-(index + 1)] - usable_times[-(index + 2)] for index in range(len(usable_times) - 1)]
            
            ## The differential is the average gradient over the moving range
            if bound_type == ReactiveBoundType.Differential:
                return statistics.mean(gradients)
        
        return 0.0
    
    @property
    def preemptive(self) -> bool:
        "Whether pre-emptive division is enabled for this strategy."
        return self.__preemptive
    
    @preemptive.setter
    def preemptive(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError(f"Preemptive must be a Boolean. Got; {value} of type {type(value)}.")
        self.__preemptive: bool = value
    
    @property
    def interrupting(self) -> bool:
        "Whether interrupting division is enabled for this strategy."
        return self.__interrupting
    
    @interrupting.setter
    def interrupting(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError(f"Interrupting must be a Boolean. Got; {value} of type {type(value)}.")
        self.__interrupting: bool = value
    
    def get_last_division_index(self, level: int) -> int:
        """
        The sub-goal stage index of the last reactive division made by this strategy at a given abstraction level.
        This is the sub-goal stage index that was current when the division was committed, and hence the plan is fixed up to the minimal achievement of the previous index.
        """
        return self.__last_division_index.get(level, 0)
    
    def get_last_division_step(self, level: int) -> int:
        "The planning step at which the last reactive division made by this strategy at a given abstraction level."
        return self.__last_division_step.get(level, 0)
    
    def reset(self) -> None:
        "Reset this strategies' internal variables for tracking planning progression, should be called after the planner returns an absolute hierarchical plan."
        self.__last_division_index: dict[int, int] = {}
        self.__last_division_step: dict[int, int] = {}
    
    def _update(self, problem_level: int, subgoal_index: int, search_length: int) -> None:
        "Update this strategy's last division point and time."
        
        last_division_index: int = self.get_last_division_index(problem_level)
        if subgoal_index <= 0 or subgoal_index <= last_division_index:
            raise ValueError("Sub-goal index must be an integer greater than zero and the last division index. "
                             f"Got; {subgoal_index=}, {last_division_index=}")
        
        last_division_step: int = self.get_last_division_step(problem_level)
        if search_length < 0 or search_length <= last_division_step:
            raise ValueError("Search length must be an integer greater than or equal to zero and the last division step. "
                             f"Got; {search_length=}, {last_division_step=}")
        
        self.__last_division_index[problem_level] = subgoal_index
        self.__last_division_step[problem_level] = search_length
    
    def proact(self,
               abstract_plan: "Planner.MonolevelPlan",
               previously_solved_problems: int
               ) -> DivisionScenario:
        return super().proact(abstract_plan, previously_solved_problems=previously_solved_problems)
    
    @classmethod
    def validate_bounds(cls, bounds: Bounds) -> Optional[Bounds]:
        for bound_type in ReactiveBoundType:
            if (_bounds := bounds.get(bound_type.value)) is None:
                continue
            if isinstance(_bounds, dict):
                _bounds = list(_bounds.values())
            else: _bounds = [_bounds]
            for bound in _bounds:
                if bound is not None:
                    if not isinstance(bound, (int, float)):
                        raise TypeError(f"Reactive division strategies must have integer or floating point bound values. Got; {bound} of type {type(bound)} for bound type {bound_type.value}.")
                    if bound <= 0.0:
                        raise ValueError(f"Bound values for reactive strategies must be greater than zero. Got; {bound} for bound type {bound_type.value}.")
        return None

class Relentless(Reactive):
    """
    Backwards horizon is only applicable for continuous reactive division.
    If interrupting division is enabled, the backwards horizon will thus be ignored.
    
    Uses either a cumulative time, incremental time, or search length bound to make either continuous or interrupting reactive problem divisions.
    
    Why can't we divide proactively based on the search length?
    Because we don't know what the length of the refined plan will be.
    We can predict with the reckless strategy, but it may simply be more effective to use reactive division of search length is the desired type of bound, the downside is that we have to use sequential yield planning so we can't use the minimum search length bound.
    
    We fix the actions, such that we are accepting the current plan as the plan that is going to be executed to reach up to that particular sub-goal stage.
    The downside of this is that it can't revise this part of the plan anymore if it turns out that those decisions were poor in the long run.
    The upside is that it makes the problem much easier to solve, because it no longer has to look back to consider whether revising that earlier part of the plan might result in a better overall plan.
    There are obviously some tradeoff here, because if the plan generated is much worse as a result of making the division, it might increase the search length by quite a lot.
    This is unlikely the make planning slower the not making the division (i.e. complete combined planning) but it might make the trade-off between plan quality and planning time not worth it.
    So there is somewhat of a dilemma in choosing how often you are going to make these divisions, and this has to be decided (currently by the user) before planning.
    The planner can't decide these intelligently, as this would require a lot of knowledge, probably machine learning to generalise for similar previously seen problem instances.
    The reason for this is that the planner doesn't actually know how good the plan is respective of the global optimum, that is it doesn't have a metric to use to gain some kind of knowledge of the global plan quality, so I can't adapt the bound.
    So basically, i'll be testing lots of different bounds on lots of different planning problems, and given that there are loads of different ways to setup these strategies, I think we've already got a huge amount to talk about really.
    """
    
    __slots__ = ("__bound_type")
    
    def __init__(self,
                 top_level: int,
                 time_bound: Bound,
                 bound_type: ReactiveBoundType = ReactiveBoundType.Incremental,
                 backwards_horizon: Horizon = 0,
                 moving_average: Optional[int] = None,
                 preemptive: bool = True,
                 interrupting: bool = False,
                 independent_tasks: bool = False,
                 order_tasks: bool = False
                 ) -> None:
        
        self.__bound_type: ReactiveBoundType = bound_type
        super().__init__(top_level=top_level,
                         **{bound_type.value : time_bound},
                         backwards_horizon=backwards_horizon if not interrupting else 0,
                         moving_average=moving_average,
                         preemptive=preemptive,
                         interrupting=interrupting,
                         independent_tasks=independent_tasks,
                         order_tasks=order_tasks)
    
    def react(self,
              problem_level: int,
              problem_total_sgoals_range: SubGoalRange,
              problem_start_step: int,
              current_search_length: int,
              current_subgoal_index: int,
              matching_child: bool,
              incremental_times: list[float],
              observable_plan: Optional[dict[int, list["Planner.Action"]]]
              ) -> Reaction:
        
        backwards_horizon: Number = self.get_horizon(problem_level)
        
        ## The problem can be divided reactively iff;
        ##      - The strategy is preemptive or this is a matching child,
        ##      - The strategy is uses continuous reactive divion (and thus cannot use backwards horizon) or,
        ##          - The backwards horizon is percentage based or,
        ##              - The backwards horizon is size based and,
        ##              - The number of achieved sub-goals stages since the last division is greater than the backwards horizon.
        can_divide: bool = ((self.preemptive or matching_child)
                            and (self.interrupting
                                 or isinstance(backwards_horizon, float)
                                 or (current_subgoal_index - self.get_last_division_index(problem_level)) > backwards_horizon))
        
        ## Calculate the current reactive value and obtain the bound (if one exists)
        value: Number = self.calculate(problem_level, problem_start_step, current_search_length, self.__bound_type, incremental_times)
        bound: Number = self.get_bound(self.__bound_type.value, problem_level, -1)
        
        ## If the bound exists and has been reached it is said to be triggered
        triggered: bool = bound != -1 and value >= bound and can_divide
        _Strategies_logger.debug(f"Reactive calculation [step = {current_search_length}, index = {current_subgoal_index}, matching = {matching_child}]: "
                                 f"bound type = {self.__bound_type}, bound = {bound}, value = {value}, backwards horizon = {backwards_horizon}, preemptive = {self.preemptive}, triggered = {triggered}")
        
        ## Update the strategy if a division was made
        if triggered: self._update(problem_level, current_subgoal_index, current_search_length)
        
        return Reaction(divide=triggered,
                        interrupt=self.interrupting,
                        backwards_horizon=backwards_horizon,
                        rationale=f"{self.__bound_type.name!s} search time bound since last division {'reached' if triggered else 'not reached'}: bound = {bound}, value = {value}")

class Impetuous(Reactive):
    
    __slots__ = ("__continuous_bound_type")
    
    def __init__(self,
                 top_level: int,
                 cumulative_time_bound: Bound,
                 continuous_time_bound: Bound,
                 continuous_bound_type: ReactiveBoundType = ReactiveBoundType.Incremental,
                 backwards_horizon: Horizon = 0,
                 moving_average: int = 5,
                 preemptive: bool = True,
                 independent_tasks: bool = False,
                 order_tasks: bool = False
                 ) -> None:
        
        if continuous_bound_type == ReactiveBoundType.Cumulative:
            raise ValueError("Cannot use a cumulative time bound as the continuous bound type.")
        self.__continuous_bound_type: ReactiveBoundType = continuous_bound_type
        super().__init__(top_level=top_level,
                         cumulative_time_bound=cumulative_time_bound,
                         **{continuous_bound_type.value : continuous_time_bound},
                         backwards_horizon=backwards_horizon,
                         moving_average=moving_average,
                         preemptive=preemptive,
                         interrupting=True,
                         independent_tasks=independent_tasks,
                         order_tasks=order_tasks)
    
    def react(self,
              problem_level: int,
              problem_total_sgoals_range: SubGoalRange,
              problem_start_step: int,
              current_search_length: int,
              current_subgoal_index: int,
              matching_child: bool,
              incremental_times: list[float],
              observable_plan: Optional[dict[int, list["Planner.Action"]]]
              ) -> Reaction:
        
        backwards_horizon: Number = self.get_horizon(problem_level)
        
        ## The problem can be divided reactively iff;
        ##      - The strategy is preemptive or this is a matching child,
        ##      - The strategy is uses continuous reactive divion (and thus cannot use backwards horizon) or,
        ##          - The backwards horizon is percentage based or,
        ##              - The backwards horizon is size based and,
        ##              - The number of achieved sub-goals stages since the last division is greater than the backwards horizon.
        can_divide: bool = ((self.preemptive or matching_child)
                            and (self.interrupting
                                 or isinstance(backwards_horizon, float)
                                 or (current_subgoal_index - self.get_last_division_index(problem_level)) > backwards_horizon))
        
        value: Number
        bound: Number
        interrupting_triggered: bool = False
        continuous_triggered: bool = False
        bound_type: Optional[ReactiveBoundType] = None
        rationale: Optional[str] = None
        
        ## If the either the interrupting or continuous bounds exist and have been reached they are said to be triggered;
        ##      - Interrupting divisions take precedence over continuous divisions.
        if (bound := self.get_bound(ReactiveBoundType.Cumulative.value, problem_level, -1)) != -1:
            
            value = self.calculate(problem_level, problem_start_step, current_search_length, ReactiveBoundType.Cumulative, incremental_times, valid_only=False)
            interrupting_triggered = value >= bound
            bound_type = ReactiveBoundType.Cumulative
            rationale = f"Cumulative interrupting bound since last division {'reached' if interrupting_triggered else 'not reached'}: bound = {bound}, value = {value}"
            _Strategies_logger.debug("Interrupting reactive time calculation:\n"
                                     f"bound type = {bound_type.name!s}, bound = {bound}, value = {value}, tiggered = {interrupting_triggered}")
        
        if (not interrupting_triggered
            and (bound := self.get_bound(self.__continuous_bound_type.value, problem_level, -1)) != -1):
            
            value = self.calculate(problem_level, problem_start_step, current_search_length, self.__continuous_bound_type, incremental_times)
            continuous_triggered = value >= bound
            bound_type = self.__continuous_bound_type
            rationale = f"{self.__continuous_bound_type.name!s} continuous time bound since last division {'reached' if continuous_triggered else 'not reached'}: bound = {bound}, value = {value}"
            _Strategies_logger.debug("Continuous reactive time calculation:\n"
                                     f"bound type = {bound_type.name!s}, bound = {bound}, value = {value}, tiggered = {continuous_triggered}")
        
        triggered: bool = (can_divide
                           and (interrupting_triggered
                                or continuous_triggered))
        _Strategies_logger.debug(f"Final reactive calculation [step = {current_search_length}, index = {current_subgoal_index}, matching = {matching_child}]: "
                                 f"bound type = {bound_type}, bound = {bound}, value = {value}, backwards horizon = {backwards_horizon}, preemptive = {self.preemptive}, triggered = {triggered}")
        
        ## Update the strategy if a division was made
        if triggered: self._update(problem_level, current_subgoal_index, current_search_length)
        
        return Reaction(divide=triggered,
                        interrupt=interrupting_triggered,
                        backwards_horizon=backwards_horizon if not interrupting_triggered else 0,
                        rationale=rationale)

@final
class Rapid(Relentless):
    """
    Adaptive strategies that commit divisions proactively, and want to use blending to avoid dependencies, can only commit continuous reactive divisions, otherwise it might break how the blends work.
    """
    __slots__ = ("__proactive_basis")   # NaiveProactive
    
    def __init__(self,
                 top_level: int,
                 proactive_basis: Type[NaiveProactive],
                 size_bound: Optional[Bound],
                 reactive_time_bound: Bound,
                 proactive_blend: dict[int, Union[Number, Blend]] = {},
                 reactive_bound_type: ReactiveBoundType = ReactiveBoundType.Incremental,
                 backwards_horizon: Horizon = 0,
                 moving_average: int = 1,
                 independent_tasks: bool = False,
                 order_tasks: bool = False) -> None:
        
        self.__proactive_basis: NaiveProactive = proactive_basis(size_bound, proactive_blend)
        super().__init__(top_level=top_level,
                         time_bound=reactive_time_bound,
                         bound_type=reactive_bound_type,
                         backwards_horizon=backwards_horizon,
                         moving_average=moving_average,
                         preemptive=True,
                         interrupting=False,
                         independent_tasks=independent_tasks,
                         order_tasks=order_tasks)
    
    def proact(self, abstract_plan: "Planner.MonolevelPlan", previously_solved_problems: int) -> DivisionScenario:
        return self.__proactive_basis.proact(abstract_plan, previously_solved_problems)

@enum.unique
class GetStrategy(enum.Enum):
    ## Basic
    basic = Basic
    
    ## Naive proactive
    hasty = Hasty
    steady = Steady
    jumpy = Jumpy
    
    ## Naive Reactive
    relentless = Relentless
    impetuous = Impetuous
    
    ## Naive adaptive
    rapid = Rapid