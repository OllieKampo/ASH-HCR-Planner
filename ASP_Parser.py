###########################################################################
###########################################################################
## Python script for running ASP programs with Clingo and parsing models ##
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

import contextlib
import enum
import functools
import logging
import os
import re
import time
from abc import abstractclassmethod
from collections import UserDict
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cached_property
from operator import itemgetter
from typing import (Any, Callable, Generator, Hashable, Iterable, Iterator,
                    Mapping, NamedTuple, Optional, Pattern, Sequence, Tuple,
                    Type, TypeVar, Union)

import _collections_abc
import clingo
import clingo.ast
import psutil
from clingo.solving import SolveHandle
from tqdm import tqdm

## ASP Parser module logger
_ASP_logger: logging.Logger = logging.getLogger(__name__)
_ASP_logger.setLevel(logging.DEBUG)

class SubscriptableDataClass(_collections_abc.Sequence):
    "Makes a dataclass subscriptable by allowing fields to be accessed by index."
    
    def __init__(self) -> None:
        super().__init__()
        if not is_dataclass(self):
            raise TypeError(f"Classes inheriting from {self.__class__} must be dataclasses.")
    
    def __getitem__(self, index: Union[int, slice]) -> Union[Any, list[Any]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return getattr(self, fields(self)[index].name)
    
    def __len__(self) -> int:
        return len(fields(self))

#############################################################################################################################################
#############################################################################################################################################
#########################   █████  ███    ██ ███████ ██     ██ ███████ ██████      ███████ ███████ ████████ ███████ #########################
#########################  ██   ██ ████   ██ ██      ██     ██ ██      ██   ██     ██      ██         ██    ██      #########################
#########################  ███████ ██ ██  ██ ███████ ██  █  ██ █████   ██████      ███████ █████      ██    ███████ #########################
#########################  ██   ██ ██  ██ ██      ██ ██ ███ ██ ██      ██   ██          ██ ██         ██         ██ #########################
#########################  ██   ██ ██   ████ ███████  ███ ███  ███████ ██   ██     ███████ ███████    ██    ███████ #########################
#############################################################################################################################################
#############################################################################################################################################

## ASP symbol type
ASP_Symbol = Union[str, clingo.Symbol]

def to_clingo_form(symbol: ASP_Symbol) -> clingo.Symbol:
    """
    Takes an ASP symbol given as a string or clingo Symbol, and retruns a clingo Symbol.
    If a string is given as argument, it is converted to clingo form via the gringo term parser and returned
    (gringo errors messages are printed to this solve signal's underlying logic program's logger as warnings).
    Otherwise, if a clingo Symbol is given as argument, the argument itself is returned.
    
    Parameters
    ----------
    `symbol : {str | clingo.Symbol}` - An ASP symbol, given as a string or a pre-constructed clingo Symbol.
    
    Returns
    -------
    `clingo.Symbol` - The argument converted to a clingo Symbol if it was a string, otherwise the argument itself.

    Raises
    ------
    `RuntimeError` - If the symbol is given as a string, and has invalid syntax according to gringo's term parser.
    """
    if isinstance(symbol, str):
        try:
            return clingo.parse_term(symbol, logger=lambda code, message: _ASP_logger.warning(f"Clingo warning: {code} {message}"))
        except RuntimeError as exception:
            _ASP_logger.error(f"Error converting string {symbol} to a clingo symbol.", exc_info=True)
            raise exception
    elif isinstance(symbol, clingo.Symbol):
        return symbol
    else: raise ValueError(f"Symbol must be a string or clingo symbol. Got; {symbol} of type {type(symbol)}.")

class Atom(UserDict):
    """
    A dictionary class used for representing ASP atoms extracted from ASP models by ASP_Parser.Model.query(...).
    Atoms are plain dictionaries, except when converted to a string, they are formatted to represent clingo ASP symbols of the form
    `name(arg_1, arg_i, ... arg_n)` where the name is the predicate name of this atom, and the args are defined by the values of the atom's dictionary.
    The class requires one additional abstract class method `predicate_name(cls) -> str` to be overridden to define the predicate name of the atom.
    
    Parameters with key 'NAME' and 'TRUTH' are ignored when inserting arguments upon converting an atom to a string.
    The value of 'TRUTH' is used to determine the truth of the atom, if it is not present the atom is considered to be true.
    
    If the class method `default_parameters(cls) -> Optional[tuple[str]]` is overridden by a sub-class and does not return None,
    then the class type can be passed to ASP_Parser.Model.query(...) as argument for name `atom_name: {str | Atom}`.
    
    Example Usage
    -------------
    A minimal example might include representing people from a database.
    
    ```
    import ASP_Parser as ASP
    class Person(ASP.Atom):
        "Represents a 'person' by the name 'N' and age 'A'."
        
        @classmethod
        def default_params(cls) -> tuple[str]:
            return ('N', 'A')
        
        @classmethod
        def predicate_name(cls) -> str:
            return "person"
    
    program = ASP.LogicProgram(\"""
                               current_date(1, 1, 2021).
                               person(N, A) :- declare_person(M, D).
                               \""")
    answer: ASP.Answer = program.solve()
    people: list[Person] = answer.fmodel.query(Person)
    print(*people.items(), sep="\\n")
    ```
    
    """
    def __str__(self) -> str:
        return "{0}{1}({2})".format('' if str(self.get('TRUTH', 'true')).lower() == 'true' else '-',
                                    self.predicate_name(),
                                    ', '.join(str(self[param]) for param in self if param not in ['NAME', 'TRUTH']))
    
    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__,
                               super().__repr__())
    
    @property
    def symbol(self) -> clingo.Symbol:
        "The clingo symbol object form of this atom."
        return to_clingo_form(str(self))
    
    @property
    def encode(self) -> str:
        "An encoded string format of this atom that represents a fact statement."
        return str(self) + '.'
    
    @property
    def _dict(self) -> dict[str, Any]:
        "A raw dictionary copy of this atom."
        return {key : value for key, value in self.items()}
    
    @abstractclassmethod
    def predicate_name(cls) -> str:
        """
        Get the name of the predicate that encode literals of this type.
        
        It is required to override this abstract class method to declare an atom class type.
        
        Returns
        -------
        `str` - A string defining the predicate name.
        """
        raise NotImplementedError
    
    @classmethod
    def default_params(cls) -> Optional[tuple[str]]:
        "The default parameters of this atom."
        return None
    
    @classmethod
    def default_cast(cls) -> Optional[dict[str, Callable[[clingo.Symbol], Any]]]:
        """
        The default types to cast parameters of this atom to, given as a mapping between parameter names and callables
        that take the respective argument for that parameter as its only argument and returns some casted version of it to any desired type.
        """
        return None
    
    @classmethod
    def default_sort(cls) -> Optional[Union[str, tuple[str]]]:
        "The default parameters of this atom to use for sorting, parameters that occur first are higher priority sort keys."
        return None
    
    def select(self, *parameters, ignore_missing: bool = False) -> dict[str, Any]:
        "Select a sub-set of the parameters of this atom and return the parameter-argument mapping as a dictionary."
        selection: dict[str, Any] = {}
        for parameter in parameters:
            if parameter in self:
                selection[parameter] = self[parameter]
            elif not ignore_missing:
                raise ValueError(f"Paramter {parameter} is not in the atom {self}.")
        return selection
    
    @staticmethod
    def nest_grouped_atoms(grouped_atoms: dict[tuple, list[Union["Atom", dict[str, Any]]]], as_standard_dict: bool = False) -> dict:
        """
        Nest the keys of a sequence of grouped atoms extracted from a model such that the keys tuple is converted to a nested sequence of dictionaries each with singleton keys.
        """
        nested_atoms: dict = {}
        if grouped_atoms and not isinstance(next(iter(grouped_atoms.keys())), tuple):
            raise ValueError("Only atoms grouped with tuple keys can be nested.")
        if grouped_atoms and not isinstance(next(iter(grouped_atoms.values())), list):
            raise ValueError("Only grouped lists of atoms can be nested.")
        for group in grouped_atoms:
            level = nested_atoms
            for key in group[:-1]:
                level = level.setdefault(key, {})
            level[group[-1]] = [dict(atom) for atom in grouped_atoms[group]] if as_standard_dict else grouped_atoms[group]
        return nested_atoms

@dataclass(frozen=True)
class Result:
    """
    Encapsulates satisfiability results for logic program solve calls.
    
    Fields
    ------
    `satisfiable : bool` - A Boolean, True if at least one satisfiable model was found during solving, False otherwise.
    
    `exhausted: bool` - A Boolean, True if the entire search space was exhausted during solving, False otherwise.
    
    Properties
    ----------
    `unsatisfiable : bool` - The negation of the `satisfiable` field (if the program was not satisfiable, it is unsatisfiable, unless search was cancelled).
    
    `interrupted : bool` - The negation of the `exhausted` field (if the search space was not exhausted, search must have been interrupted).
    """
    satisfiable: bool
    exhausted: bool
    
    def __str__(self) -> str:
        return (f"{self.__class__.__name__} :: "
                + " : ".join(["SATISFIABLE" if self.satisfiable else "UNSATISFIABLE",
                              "SEARCH EXHAUSTED" if self.exhausted else "SEARCH INTERRUPTED"]))
    
    @property
    def unsatisfiable(self) -> bool:
        return not self.satisfiable
    
    @property
    def interrupted(self) -> bool:
        return not self.exhausted

@dataclass(frozen=True, order=True)
class Memory:
    """
    Represents the memory used by a logic program whilst solving.
    
    Elements
    --------
    `rss : float` - The Resident Set Size, the non-swapped physical memory a process has used.
    
    `vms : float` - The Virtual Memory Size, the total amount of virtual memory used by the process.
    """
    rss: float
    vms: float
    
    def __str__(self) -> str:
        return f"(RSS = {self.rss:.6f}Mb, VMS = {self.vms:.6f}Mb)"

@dataclass(frozen=True)
class Statistics:
    """
    Encapsulates timing statistics for individual logic program solve calls.
    
    Fields
    ------
    `grounding_time : float` - A float defining the 'wall clock' time spent in seconds
    grounding the logic program, as reported by Python's highest resolution performance timer.
    
    `solving_time : float` - A float defining the 'wall clock' time spent in seconds
    solving the logic program, as reported by Python's highest resolution performance timer.
    
    `total_time : float` - A float defining the total 'wall clock' time spent in seconds
    from initiating and returning from the solve call to the logic program. This may be slightly
    greater than the sum of the solving and ground time, as it accounts for all additional processing.
    
    `memory : Memory` - A memory object representing the maximum amount of memory needed to solve the program.
    
    `step_range : {range | None}` - The contiguous range of steps solved by the solving increment that this statistics object represents.
    If this a cumulative statistics object, then the step range is the entire range over all increments.
    If this is a base statistics object, then the step range is None.
    
    `clingo_stats : {dict[str, Any]}` - A dictionary whose keys are strings and whose values are either
    floats or nested dictionaries. The dictionary contains statistics returned by the internal Clingo solver.
    It is populated only if the solver was called with option ASP_Parser.Options.statistics() or '--stats' enabled.
    """
    grounding_time: float
    solving_time: float
    total_time: float = -1.0
    memory: Memory = Memory(0.0, 0.0)
    step_range: Optional[range] = None
    clingo_stats: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.total_time < 0.0:
            object.__setattr__(self, "total_time", self.grounding_time + self.solving_time)
    
    def __str__(self):
        return (f"{self.__class__.__name__} :: "
                + ", ".join([f"Grounding = {self.grounding_time:.6f}s",
                             f"Solving = {self.solving_time:.6f}s",
                             f"Total = {self.total_time:.6f}s",
                             f"Memory = {self.memory!s}"]
                            + ([f"Step range = [{min(self.step_range)}-{max(self.step_range)}]"]
                               if self.step_range is not None else [])))

@dataclass(frozen=True)
class IncrementalStatistics(Statistics):
    """
    Encapsulates timing statistics for a series of incremental solve calls.
    
    Fields
    ------
    `cumulative : Statistics` - A statistics object containing cumulative timing statistics
    summed over all incremental solve calls. The clingo stats are empty for this object.
    
    `incremental : dict[int, Statistics]` - A dictionary whose keys are integers defining ordinal
    incremental solve call numbers and whose values are statistics objects for the respective call.
    
    Properties
    ----------
    `calls : int` - The number of solver calls made to find a solution to the logic program.
    This is simply 1 if a standard one-shot solve was made, the number of incremental solve calls made.
    
    `incremental_stats_str : str` - A formatted multiple line string of the incremental statistics.
    """
    cumulative: Statistics = Statistics(0.0, 0.0)
    incremental: dict[int, Statistics] = field(default_factory=dict)
    
    def __str__(self):
        return (f"{self.__class__.__name__} :: "
                + ", ".join([f"Cumulative = ({self.cumulative!s})",
                             f"Calls = {self.calls}"]))
    
    @property
    def calls(self) -> int:
        "The number of solver calls made to find a solution to the logic program."
        return len(self.incremental)
    
    @property
    def grand_totals(self) -> Statistics:
        "The grand total times and maximum memory usage, the sum of the base and all incremental times."
        return Statistics(self.grounding_time + self.cumulative.grounding_time,
                          self.solving_time + self.cumulative.solving_time,
                          memory=Memory(max(self.memory.rss, self.cumulative.memory.rss),
                                        max(self.memory.vms, self.cumulative.memory.vms)))
    
    @property
    def incremental_stats_str(self) -> str:
        "Format the incremental statistics into a multiple line string."
        return "\n".join([f"{step} : {stats}" for step, stats in self.incremental.items()])
    
    def combine_with(self, other: "IncrementalStatistics", shift_increments: int = 0) -> "IncrementalStatistics":
        "Combine this incremental statistics object with another. Incremental statistics are combined on a increment-wise basis."
        
        ## Form a merged sequence of incremental statistics
        incremental: dict[int, Statistics] = self.incremental
        
        for increment, current_stat in other.incremental.items():
            _increment: int = increment + shift_increments
            
            if (existing_stat := incremental.get(increment, None)) is not None:
                incremental[_increment] = Statistics(existing_stat.grounding_time + current_stat.grounding_time,
                                                     existing_stat.solving_time + current_stat.solving_time,
                                                     memory=max(existing_stat.memory, current_stat.memory),
                                                     step_range=range(min(existing_stat.step_range.start, current_stat.step_range.start),
                                                                      max(existing_stat.step_range.stop, current_stat.step_range.stop)))
            else: incremental[_increment] = current_stat
        
        ## Calculate cumulative totals
        cumulative_grounding: float = self.cumulative.grounding_time + other.cumulative.grounding_time
        cumulative_solving: float = self.cumulative.solving_time + other.cumulative.solving_time
        cumulative_memory = Memory(max(self.cumulative.memory.rss, other.cumulative.memory.rss),
                                   max(self.cumulative.memory.vms, other.cumulative.memory.vms))
        cumulative_step_range = range(min(self.cumulative.step_range.start, other.cumulative.step_range.start),
                                      max(self.cumulative.step_range.stop, other.cumulative.step_range.stop))
        cumulative = Statistics(cumulative_grounding, cumulative_solving,
                                memory=cumulative_memory, step_range=cumulative_step_range)
        
        return IncrementalStatistics(self.grounding_time,
                                     self.solving_time,
                                     self.total_time,
                                     self.memory,
                                     self.step_range,
                                     cumulative=cumulative,
                                     incremental=incremental)

ParameterConstraint = Union[str, int, tuple[Pattern, "Model.ParseMode"], Callable[[clingo.Symbol], bool]]

@dataclass(frozen=True)
class Model(_collections_abc.Set):
    """
    Encapsulates a model returned by logic program solve calls.
    A model is an answer set, a finite set of atoms produced by a logic program.
    Models have frozen set semantics; they are immutable, unordered and contain no duplicates.
    Models support all the usual set operations, and also a variety of additional methods for querying the model for the existence and truth of atoms.
    Unlike raw Clingo models, models of this type do not expire upon termination of the Clingo program that produced them, they are safe to store and use later.
    
    Fields
    ------
    `symbols : frozenset[clingo.Symbol]` - A frozen set of Clingo ASP symbols (atoms) encapsulated by this model.
    The set is unordered and contains no duplicates.
    
    `cost : tuple[int]` - A tuple of integers defining the optimisation cost of the model.
    The tuple is ordered from the highest to the lowest priority of the optimisation statements in the ASP logic program that created this model.
    
    `optimality_proven : bool` - A Boolean, True if the model is optimal respective of the logic program that generated it, False otherwise.
    
    `number : int` - An ordinal natural number representing the order this model was computed model.
    This is respective of its index in the sequence of all computed models from the solver call from which this model was generated.
    
    `thread_id : int` - A natural number defining the numerical ID of the thread that found this model.
    
    `model_type : {clingo.ModelType, None}` - Clingo model type for this model, None if the program was unsatisfiable.
    """
    symbols: frozenset[clingo.Symbol]
    cost: tuple[int] = field(default_factory=tuple)
    optimality_proven: bool = False
    number: int = -1
    thread_id: int = -1
    model_type: Optional[clingo.ModelType] = None
    
    def __post_init__(self) -> None:
        if not isinstance(self.symbols, frozenset):
            object.__setattr__(self, "symbols", frozenset(self.symbols))
        if not isinstance(self.cost, tuple):
            object.__setattr__(self, "cost", tuple(self.cost))
    
    def __str__(self) -> str:
        return (f"{self.__class__.__name__} :: "
                + ", ".join([f"Total atoms = {len(self.symbols)}",
                             f"Cost = {self.cost}",
                             f"Optimality proven = {self.optimality_proven}",
                             f"Number = {self.number}",
                             f"Thread ID = {self.thread_id}",
                             f"Model type = {self.model_type}"]))
    
    def __len__(self) -> int:
        return len(self.symbols)
    
    def __iter__(self) -> Generator[str, None, None]:
        for symbol in self.symbols:
            yield str(symbol)
    
    def __bool__(self) -> bool:
        return len(self.symbols) != 0
    
    def __contains__(self, atom: Union[str, clingo.Symbol]) -> bool:
        if isinstance(atom, str):
            return clingo.parse_term(atom) in self.symbols
        elif isinstance(atom, clingo.Symbol):
            return atom in self.symbols
        return False
    
    def __and__(self, other: "Model") -> "Model":
        return Model(self.symbols & other.symbols)
    
    def __or__(self, other: "Model") -> "Model":
        return Model(self.symbols | other.symbols)
    
    def __xor__(self, other: "Model") -> "Model":
        return Model(self.symbols ^ other.symbols)
    
    def union(self, other: "Model") -> "Model":
        """
        Find the union of this and another model, returning a new model.
        Where the union of two sets is a set containing the elements that occur in either of the originals.
        
        Parameters
        ----------
        `other : Model` - The other model to unite with this model.
        
        Returns
        -------
        `Model` - A new model representing the union of this and the other model.
        """
        return Model(self.symbols.union(other.symbols))
    
    def intersection(self, other: "Model") -> "Model":
        """
        Find the intersection of this and another model, returning a new model.
        Where the intersection of two sets is a set containing only the elements that occur in both of the originals.
        
        Parameters
        ----------
        `other : Model` - The other model to intersect with this model.
        
        Returns
        -------
        `Model` - A new model representing the intersection of this and the other model.
        """
        return Model(self.symbols.intersection(other.symbols))
    
    def difference(self, other: "Model") -> "Model":
        """
        Find the difference between this and another model, returning a new model.
        Where the difference between two sets is a set containing only the elements
        that occur in the first set (this model) and not in the second (other model).
        
        Parameters
        ----------
        `other : Model` - The other model to find the difference between this model.
        
        Returns
        -------
        `Model` - A new model representing the difference between this and the other model.
        """
        return Model(self.symbols.difference(other.symbols))
    
    def symmetric_difference(self, other: "Model") -> "Model":
        """
        Find the symmetric difference between this and another model, returning a new model.
        Where the symmetric difference between two sets is a set containing only the elements that occur in either of the originals but not both.
        
        Parameters
        ----------
        `other : Model` - The other model to find the symmetric difference between this model.
        
        Returns
        -------
        `Model` - A new model representing the symmetric difference between this and the other model.
        """
        return Model(self.symbols.symmetric_difference(other.symbols))
    
    def is_true(self, atom: str) -> bool:
        """
        Determine whether a given atom is in the model and has a positive truth value.
        
        Parameters
        ----------
        `atom : str` - An alphanumeric string which may contain underscores whose leadering character is a lowercase letter representing an ASP atom.
        Usually a function symbol of the form 'name(arg_1, arg_2, ... arg_n)'.
        If the string is incorrectly formatted, or has invalid syntax according to Gringo's term parser, this function will return False.
        
        Returns
        -------
        `bool` - A Boolean, True if the given atom is in the model and has a positive truth value, False otherwise.
        """
        return any((str(symbol).lstrip("-") == atom and symbol.positive) for symbol in self.symbols)
    
    def is_false(self, atom: str) -> bool:
        """
        Determine whether a given atom is in the model and has a negative truth value.
        
        Parameters
        ----------
        `atom : str` - An alphanumeric string which may contain underscores whose leadering character is a lowercase letter representing an ASP atom.
        Usually a function symbol of the form 'name(arg_1, arg_2, ... arg_n)'.
        If the string is incorrectly formatted, or has invalid syntax according to Gringo's term parser, this function will return False.
        
        Returns
        -------
        `bool` - A Boolean, True if the given atom is in the model and has a negative truth value, False otherwise.
        """
        return any((str(symbol).lstrip("-") == atom and symbol.negative) for symbol in self.symbols)
    
    def is_in(self, atom: Union[str, clingo.Symbol]) -> bool:
        """
        Determine whether a given atom is in the model and has any truth value.
        
        Parameters
        ----------
        `atom : str` - An alphanumeric string which may contain underscores whose leadering character is a lowercase letter representing an ASP atom.
        Usually a function symbol of the form 'name(arg_1, arg_2, ... arg_n)'.
        If the string is incorrectly formatted, or has invalid syntax according to Gringo's term parser, this function will return False.
        
        Returns
        -------
        `bool` - A Boolean, True if the given atom is in the model and has either a negative or positive truth value, False otherwise.
        """
        return any((str(symbol).lstrip("-") == atom) for symbol in self.symbols)
    
    @enum.unique
    class ParseMode(enum.Enum):
        """
        An enumeration defining the three possible regular expression matching modes.
        These are used when parsing a model with:
            - `Model.get_atoms(...)`,
            - `Model.query(...)`,
            - `Model.regex_parse(...)`.
        
        The enumerated values are functions, stored in singleton tuples to conform with python naming conventions.
        
        Items
        -----
        `Match = (lambda regex, value: regex.match(value),)` - Match the regular expression from the start of the string.
        Functionality is defined by re.match(patten: re.Pattern, string: AnyStr).
        
        `FullMatch = (lambda regex, value: regex.fullmatch(value),)` - Match the regular expression to the entire string.
        Functionality is defined by re.fullmatch(patten: re.Pattern, string: AnyStr).
        
        `Search = (lambda regex, value: regex.search(value),)` - Match the regular expression anywhere in the string.
        Functionality is defined by re.search(patten: re.Pattern, string: AnyStr).
        """
        Match = (lambda regex, value: regex.match(value),)
        FullMatch = (lambda regex, value: regex.fullmatch(value),)
        Search = (lambda regex, value: regex.search(value),)
    
    @staticmethod
    def __satisfies_constraint(value: clingo.Symbol, constr: ParameterConstraint) -> bool:
        "Check whether a symbol's argument satisfies a given constraint."
        return ((isinstance(constr, tuple)
                 and constr[1].value[0](constr[0], str(value)) is not None)
                or (isinstance(constr, Callable)
                    and constr(value))
                or str(value) == str(constr))
    
    @staticmethod
    def __group_atoms(sequence: Sequence[clingo.Symbol], key: Callable[[clingo.Symbol], Union[clingo.Symbol, tuple[clingo.Symbol]]]) -> tuple[clingo.Symbol, list[clingo.Symbol]]:
        "Group a sequence of symbols using a given grouping key function."
        groups: dict[Union[clingo.Symbol, tuple[clingo.Symbol]], list[clingo.Symbol]] = {}
        for item in sequence:
            groups.setdefault(key(item), []).append(item)
        return tuple(groups.items())
    
    def get_atoms(
                  self: "Model",
                  atom_name: Union[str, Tuple[re.Pattern, "Model.ParseMode"]],
                  arity: int,
                  /,
                  
                  truth: Optional[bool] = None,
                  param_constrs: Mapping[int, Union[str, int, Tuple[re.Pattern, "Model.ParseMode"], Sequence[Union[str, int, Tuple[re.Pattern, "Model.ParseMode"]]]]] = {},
                  
                  *,
                  sort_by: Optional[Union[int, Sequence[int]]] = None,
                  group_by: Optional[Union[int, Sequence[int]]] = None,
                  convert_keys: Optional[Union[Callable[[clingo.Symbol], Hashable], Mapping[int, Callable[[clingo.Symbol], Hashable]]]] = None,
                  as_strings: bool = False
                  
                  ) -> Union[list[ASP_Symbol], dict[Union[Hashable, tuple[Hashable]], list[ASP_Symbol]]]:
        """
        Extract atoms from the model of a given; name, arity, and classical truth value.
        
        The extracted atoms are returned as a list of either Clingo symbols or strings, optionally sorted according to their arguments.
        Paramaters can optionally be constrained such that the returned atoms' arguments must match the constraint for the given parameter.
        
        Parameters
        ----------
        `atom_name : {str | (re.Pattern, ParseMode)}` - The name constraint of the atoms to extract.
        If given as a string, the extracted atoms' names must match exactly.
        If given as a two-tuple, the first item must be a compiled regular expression and second item an entry from the Model.ParseMode enum specifying how to match the regular expression, only atoms whose name's match will be returned.
        
        `arity : int` - The arity of the atoms to extract.
        Only atoms whose number of arguments equals this value will be returned.
        
        `truth : {bool | None}` - The classical truth of the atoms to extract.
        True to extract only positive atoms, False to extract only negative atoms, or None to extract all atoms regardless of their truth.
        
        `param_constrs : Mapping[int, {str | int | (re.Pattern, ParseMode)}] = {}` - A mapping whose keys are integers defining parameter indices in the range [0-arity] and whose values are either strings, integers, or two-tuples.
        If a two-tuple, its first item is a regular expression and its second item is an entry from the `ParseMode` enum defining how to match the regular expression.
        The values define constraints for the arguments of the parameter of that key, only atoms whose arguments match all parameter constraints will be returned.
        
        Keyword Parameters
        ------------------
        `sort_by : Sequence[int] = []` - The sorting priority of arguments for the parameters of the list of returned atoms.
        A sequence of integers defining parameter indices in the range [0-arity] specifying.
        
        `group_by : Sequence[int] = []` - Group the returned atoms according to those which have the same arguments for the given parameters.
        If given and not None, then the atoms will be returned as a parameter index to atom list mapping, where all atoms in a list have the same argument for the given parameter index.
        
        `convert_keys : Mapping[int, Any] = []` - Convert the keys of a grouped mapping of atoms to the given types.
        
        `as_strings : bool = False` - A Boolean, True to return the extracted atoms as strings, otherwise False (the default) to return them as raw clingo symbols.
        
        Returns
        -------
        `{list[str] | list[clingo.Symbol]}` - A list of either strings or Clingo symbols containing unique atoms from this model.
        
        Example Usage
        -------------
        >>> import ASP_Parser as ASP
        >>> import clingo
        >>> import re
        >>> answer: ASP.Answer = ASP.LogicProgram(\"""
                                                  occurs(move(robot, library), 0).
                                                  occurs(grasp(robot, book), 1).
                                                  -occurs(grasp(robot, book), 0).
                                                  -occurs(move(robot, library), 1).
                                                  holds(in(robot, office), 0).
                                                  holds(grasping(robot, book), 2).
                                                  holds(in(robot, library), 1).
                                                  \""").solve()
        
        Extract positive atoms with name 'occurs' and arity 2. Note that the output's order is arbitrary and not necessarily the same order the atoms occur in the program.
        >>> answer.model.get_atoms("occurs", 2, True)
        [occurs(grasp(robot,book),1), occurs(move(robot,library),0)]
        
        Extract positive atoms with name 'holds' and arity 2, sort by the second argument.
        >>> answer.model.get_atoms("holds", 2, True, sort_by=[1])
        [holds(in(robot,office),0), holds(in(robot,library),1), holds(grasping(robot,book),2)]
        
        Extract negative atoms with name 'occurs', arity 2 and whose second argument is 1.
        >>> answer.model.get_atoms("occurs", 2, False, param_constrs={1 : 1})
        [-occurs(move(robot,library),1)]
        
        Extract atoms regards of truth value with name 'occurs', arity 2 and whose first argument starts with the string 'move'.
        >>> answer.model.get_atoms("occurs", 2, None, param_constrs={0 : (re.compile(r"move"), ASP.Model.ParseMode.Match)})
        [-occurs(move(robot,library),1), occurs(move(robot,library),0)]
        
        Extract positive atoms whose name contains the letter 's' and which have arity 2, sort first by the first argument and then by the second, return the atoms as strings.
        >>> answer.model.get_atoms((re.compile(r"s"), ASP.Model.ParseMode.Search), 2, True, as_strings=True, sort_by=[1, 0])
        ['holds(in(robot,office),0)', 'occurs(move(robot,library),0)', 'occurs(grasp(robot,book),1)', 'holds(in(robot,library),1)', 'holds(grasping(robot,book),2)']
        """
        ## Determine which atoms fit the constraints provided
        atoms: Union[list[clingo.Symbol], list[str]] = []
        for symbol in self.symbols:
            if ((truth is None
                 or truth == symbol.positive)
                and len(symbol.arguments) == arity
                and ((isinstance(atom_name, tuple)
                      and atom_name[1].value[0](atom_name[0], symbol.name) is not None)
                     or symbol.name == atom_name)
                and (all((isinstance(param_constrs[index], Iterable) and not isinstance(param_constrs[index], (str, tuple))
                          and any(self.__satisfies_constraint(symbol.arguments[index], constr) for constr in param_constrs[index]))
                         or self.__satisfies_constraint(symbol.arguments[index], param_constrs[index])
                         for index in param_constrs))):
                atoms.append(symbol)
        
        def keys_for(list_):
            """Creates functions that extracts atom arguments for sorting and grouping."""
            return lambda item: (tuple(item.arguments[index] for index in list_ if index in range(0, arity)) if isinstance(list_, Iterable) else item.arguments[list_])
        
        def convert(index, key):
            """Converts individual keys to the desired type."""
            if isinstance(convert_keys, Mapping):
                return key if index not in convert_keys else convert_keys[index](key)
            else: return convert_keys(key)
        
        ## Sort the atoms as requested
        if sort_by is not None:
            atoms = sorted(atoms, key=keys_for(sort_by))
        
        ## Find the valid grouping indices
        _group_by: Optional[Union[int, Sequence[int]]] = None
        if isinstance(group_by, Iterable):
            _group_by = (index for index in group_by if index in range(0, arity))
        elif group_by in range(0, arity):
            _group_by = group_by
        
        ## Group the atoms accordingly
        if _group_by is not None:
            if convert_keys is not None:
                atoms = {tuple(convert(index, key) for index, key in zip(_group_by, keys)) if isinstance(_group_by, list) else convert(_group_by, keys)
                        : list(group) for keys, group in self.__group_atoms(atoms, key=keys_for(_group_by))}
            else: atoms = {keys : list(group) for keys, group in self.__group_atoms(atoms, key=keys_for(_group_by))}
        
        ## Convert to strings as requested
        return atoms if not as_strings else ([str(atom) for atom in atoms] if _group_by is None else {key : [str(atom) for atom in atoms[key]] for key in atoms})
    
    def query(
              self: "Model",
              atom_name: Union[str, tuple[re.Pattern, "Model.ParseMode"], Type[Atom]],
              atom_params: Optional[Sequence[str]] = None,
              /,
              
              truth: bool = True,
              param_constrs: Mapping[str, Union[ParameterConstraint, Iterable[ParameterConstraint]]] = {},
              *,
              
              sort_by: Optional[Union[str, Sequence[str]]] = None,
              group_by: Optional[Union[str, Sequence[str]]] = None,
              cast_to: Optional[Union[Callable[[clingo.Symbol], Any], Sequence[Callable[[clingo.Symbol], Any]], Mapping[str, Union[Callable[[clingo.Symbol], Any], Sequence[Callable[[clingo.Symbol], Any]]]]]] = None,
              add_truth: Union[bool, Type[str], Type[bool]] = False,
              add_name: Union[bool, Type[str]] = False
              
              ) -> Union[list[dict[str, ASP_Symbol]], dict[Union[Any, tuple[Any]], list[dict[str, ASP_Symbol]]]]:
        """
        Extract atoms from the model of a given; name, arity, and classical truth value, and return their parameter-argument mappings.
        
        Arbitrary names for the parameters of the extracted atoms must be given as a sequence of strings.
        The extracted atoms are returned as a list, optionally sorted according to their arguments, of dictionaries.
        The dictionarys' keys are the given parameter names and their values are the atoms' arguments for those parameters.
        Paramaters can optionally be constrained, such that the returned atoms' arguments must match the constraint for the given parameter.
        The constraint can be given either as a string or a regular expression.
        
        Complex queries can choose to:
            - sort the atoms in ascending order according to a sub-sequence of their arguments,
            - group the return list into a dictionary or lists containing only atoms with the same argument for a given sub-set of their parameters,
            - cast the arguments to any desired type.
        
        The returned dictionary can optionally include the name of the atom under key "NAME" and the classical truth of the atom under key "TRUTH".
        
        Parameters
        ----------
        `atom_name : {str | tuple[re.Pattern, ParseMode]}` - The name of the atoms to extract, only atoms whose name's match will be returned.
        Given as either a string or a two-tuple whose first element is a compiled regular expression and second is a parsing mode specifying how to match the expression.
        
        `atom_params : Sequence[str]` - An ordered sequence arbitrary names for the parameters of the extracted atoms given as strings.
        The arity of the extracted atoms will be equal to the length of this sequence.
        
        `truth : {bool | None}` - The classical truth value of the atoms to extract.
        Given as either a Boolean or None, True extracts only positive atoms, False extracts only negative, and None extracts all atoms regardless of truth value.
        
        `param_constrs : Mapping[str, {str, int, Tuple[Pattern, ParseMode], Iterable[{str, int, Tuple[Pattern, ParseMode]}]}] = {}`
            - A mapping whose keys are strings defining parameter names from `atom_params` and whose values are constraints for those parameters.
              The parameters constraints are applied at atoms such that; the atoms' arguments for the parameters given as the mapping keys must satisfy the respective constraint given as the mapping value.
              Constraints must be given as either; strings, integers, two-tuples, or an iterable (which cannot also be a tuple) of any of the latter.
              If a two-tuple, its first item is a regular expression and its second item is an entry from the `Model.ParseMode` enum defining how to match the regular expression.
              If an iterable, the constraints contained are applied disjunctively such that, the argument for the given parameter must satisfy at least one of the constraints.
        
        Keyword Parameters
        ------------------
        `sort_by : Sequence[str] = []` - A sequence of strings defining parameter names from `atom_params` specifying the sorting priority of arguments of those parameters of the list of returned atoms.
        
        `group_by : Sequence[str] = []` - Group the returned atoms according to those which have the same arguments for the given parameters.
        If given and not None, then the atoms will be returned as a parameter name to atom list mapping, where all atoms in a list have the same argument for the given parameter name.
        
        `convert_keys : Mapping[int, Any] = []` - Convert the keys of a grouped mapping of atoms to the given types.
        
        `add_truth : bool = False` - A Boolean, True to include the classical truth of returned atoms under key "TRUTH", otherwise False (the default).
        
        `add_name : bool = False` - A Boolean, True to include the name of returned atoms under key "NAME", otherwise False (the default).
        
        Returns
        -------
        `list[dict[str, {str, clingo.Symbol}]]` - A list of dictionaries containing the parameter-argument mappings of unique atoms from this model.
        The dictionary keys are strings from `atom_params` (insertion order is preserved) and their values are the arguments for those parameters as either strings or Clingo symbols.
        
        Example Usage
        -------------
        >>> import ASP_Parser as ASP
        >>> import clingo
        >>> import re
        >>> answer: ASP.Answer = ASP.LogicProgram(\"""
                                                  occurs(move(robot, library), 0).
                                                  occurs(grasp(robot, book), 1).
                                                  -occurs(grasp(robot, book), 0).
                                                  -occurs(move(robot, library), 1).
                                                  holds(in(robot, office), 0).
                                                  holds(grasping(robot, book), 2).
                                                  holds(in(robot, library), 1).
                                                  \""").solve()
        
        Extract positive atoms with name 'occurs' and arity 2. Note that the output's order is arbitrary and not necessarily the same order the atoms occur in the program.
        >>> answer.model.query("occurs", ["A", "I"], True)
        [{'A': grasp(robot,book), 'I': 1}, {'A': move(robot,library), 'I': 0}]
        
        Extract positive atoms with name 'holds' and arity 2, sort by the second argument.
        >>> answer.model.query("holds", ["A", "I"], True, sort_by=["I"])
        [{'A': in(robot,office), 'I': 0}, {'A': in(robot,library), 'I': 1}, {'A': grasping(robot,book), 'I': 2}]
        
        Extract negative atoms with name 'occurs', arity 2 and whose second argument is 1.
        >>> answer.model.query("occurs", ["A", "I"], False, param_constrs={"I" : 1})
        [{'A': move(robot,library), 'I': 1}]
        
        Extract atoms regards of truth value with name 'occurs', arity 2 and whose first argument starts with the string 'move'.
        >>> answer.model.query("occurs", ["A", "I"], None, param_constrs={"A" : (re.compile(r"move"), ASP.Model.ParseMode.Match)}, add_truth=True)
        [{'TRUTH': false, 'A': move(robot,library), 'I': 1}, {'TRUTH': true, 'A': move(robot,library), 'I': 0}]
        
        Extract positive atoms whose name contains the letter 's' and which have arity 2, sort first by the first argument and then by the second, return the atoms as strings.
        >>> answer.model.query((re.compile(r"s"), ASP.Model.ParseMode.Search), ["X", "Y"], True, sort_by=["X", "Y"], add_name=True, as_strings=True)
        [{'NAME': 'occurs', 'X': 'grasp(robot,book)', 'Y': '1'}, {'NAME': 'holds', 'X': 'grasping(robot,book)', 'Y': '2'}, {'NAME': 'holds', 'X': 'in(robot,library)', 'Y': '1'}, {'NAME': 'holds', 'X': 'in(robot,office)', 'Y': '0'}, {'NAME': 'occurs', 'X': 'move(robot,library)', 'Y': '0'}]
        """
        _atom_name: Union[str, tuple[re.Pattern, "Model.ParseMode"]]
        _atom_type: Type[dict]
        _atom_params: tuple[str]
        if isinstance(atom_name, type) and issubclass(atom_name, Atom):
            _atom_name = atom_name.predicate_name()
            _atom_type = atom_name
            if (atom_params is None
                and (_atom_params := atom_name.default_params()) is None):
                raise ValueError(f"Atom type {atom_name.__class__} given without explicit or default parameters.")
        else:
            _atom_name = atom_name
            _atom_type = dict
            if (_atom_params := atom_params) is None:
                raise ValueError(f"Atom parameters must be given explicitly unless an Atom type with default parameters defined is given.")
        arity: int = len(_atom_params)
        
        def get_name_checker(name_type: type) -> Callable[[clingo.Symbol], bool]:
            "Creates a lambda function for checking an atom's name agaist the constraint given."
            if name_type == str:
                return lambda symbol: symbol.name == _atom_name
            return lambda symbol: _atom_name[1].value[0](_atom_name[0], symbol.name) is not None
        name_checker: Callable[[clingo.Symbol], bool] = get_name_checker(type(_atom_name))
        
        constr_params: list[Tuple[str, int]] = []
        for param in param_constrs:
            if param in _atom_params:
                constr_params.append((param, _atom_params.index(param)))
        
        _cast_to = cast_to
        _sort_by = sort_by
        if issubclass(_atom_type, Atom):
            if _cast_to is None:
                _cast_to = atom_name.default_cast()
            if _sort_by is None:
                _sort_by = atom_name.default_sort()
        
        atoms: list[dict[str, Any]] = []
        atom: dict[str, clingo.Symbol] = _atom_type()
        for symbol in self.symbols:
            if ((truth is None or truth == symbol.positive)
                and len(symbol.arguments) == arity
                and name_checker(symbol)
                and (all((isinstance(param_constrs[param], Iterable) and not isinstance(param_constrs[param], (str, tuple))
                          and any(self.__satisfies_constraint(symbol.arguments[index], constr) for constr in param_constrs[param]))
                         or self.__satisfies_constraint(symbol.arguments[index], param_constrs[param])
                         for param, index in constr_params))):
                atom = _atom_type()
                
                if add_truth:
                    if isinstance(add_truth, type) and issubclass(add_truth, (bool, str)):
                        atom["TRUTH"] = add_truth(symbol.positive)
                    else: atom["TRUTH"] = clingo.Function(str(symbol.positive).lower())
                if add_name:
                    if isinstance(add_name, type) and issubclass(add_name, str):
                        atom["NAME"] = add_name(symbol.name)
                    else: atom["NAME"] = symbol.name
                
                if _cast_to is not None:
                    for index, param in enumerate(_atom_params):
                        if isinstance(_cast_to, Mapping):
                            if param in _cast_to:
                                if isinstance(_cast_to[param], Sequence):
                                    arg: Any = symbol.arguments[index]
                                    for cast in _cast_to[param]:
                                        arg = cast(arg)
                                    atom[param] = arg
                                else: atom[param] = _cast_to[param](symbol.arguments[index])
                            else: atom[param] = symbol.arguments[index]
                        elif isinstance(_cast_to, Sequence):
                            arg: Any = symbol.arguments[index]
                            for cast in _cast_to:
                                arg = cast(arg)
                            atom[param] = arg
                        else: atom[param] = _cast_to(symbol.arguments[index])
                else:
                    for index, param in enumerate(_atom_params):
                        atom[param] = symbol.arguments[index]
                
                atoms.append(atom)
        
        def validate_params(params: Optional[Union[str, Sequence[str]]]) -> Optional[tuple[str]]:
            "Validates sorting and grouping parameters, discarding any that are not in the atom parameter list."
            if isinstance(params, str) and params in _atom_params:
                return (params,)
            elif (isinstance(params, Sequence) and params
                  and (valid_params := tuple(param for param in params if param in _atom_params))):
                return valid_params
            else: return None
        
        _sort_by: Optional[tuple[str]] = validate_params(_sort_by)
        _group_by: Optional[tuple[str]] = validate_params(group_by)
        
        def keys_for(params: tuple[str]) -> itemgetter:
            "Creates an item getter for getting keys for sorting and grouping atoms."
            return itemgetter(*[param for param in params])
        
        if _sort_by is not None:
            atoms = sorted(atoms, key=keys_for(_sort_by))
        
        if _group_by is not None:
            grouped_atoms: dict[Any, list[dict[str, Any]]] = {}
            for key, group in self.__group_atoms(atoms, key=keys_for(_group_by)):
                grouped_atoms[key] = group
            return grouped_atoms
        
        return atoms
    
    def regex_parse(self,
                    regex: Union[re.Pattern, str],
                    parse_mode: Optional[ParseMode] = None,
                    return_as_strings: bool = True
                    ) -> set[ASP_Symbol]:
        """
        Parse over each atom in the model using a regular expression.
        Those atoms who match the expression according to the given parsing mode are returned.
        
        Parameters
        ----------
        `regex: re.Pattern | str` -
        
        `parse_mode: ParseMode | None = None` -
        
        `return_as_strings: bool = True` -
        
        Returns
        -------
        `set[str | clingo.Symbol]` -
        """
        atoms: set[str] = set()
        for atom in self.symbols:
            str_atom: str = str(atom)
            if ((parse_mode is None
                 and re.match(regex, str_atom) is not None)
                or parse_mode.value[0](regex, str_atom) is not None):
                 atoms.add(str_atom if return_as_strings else atom)
        return atoms
    
    def func_parse(self,
                   callback_function: Callable[[ASP_Symbol], bool],
                   callback_as_strings: bool = True,
                   return_as_strings: bool = True
                   ) -> set[ASP_Symbol]:
        """
        Parse over each atom in the model using a callback function.
        The callback function is called on each atom in the model.
        If the callback returns True is atom is returned in a set as their string representation.
        
        Parameters
        ----------
        `callback_function: Callable[[ASP_Symbol], bool]` -
        
        `callback_as_strings: bool = True` -
        
        `return_as_strings: bool = True` -
        
        Returns
        -------
        `set[str | clingo.Symbol]` -
        """
        atoms: set[ASP_Symbol] = set()
        for atom in self.symbols:
            str_atom: str = str(atom)
            if callback_function(str_atom if callback_as_strings else atom):
                atoms.add(str_atom if return_as_strings else atom)
        return atoms
    
    def evaluate(self, rule: str, solver_options: Iterable[str] = [], assumptions: Iterable[clingo.Symbol] = [], context: Iterable[Callable[..., clingo.Symbol]] = []) -> "Answer":
        ## context_type = type("context", (object,), {func.__name__ : func for func in context})
        _ASP_logger.debug(f"Evaluating rule '{rule}' over:\n{self}")
        logic_program = LogicProgram(rule, name="Evaluate", silent=True)
        logic_program.add_rules(self.symbols)
        return logic_program.solve(solver_options=solver_options, assumptions=assumptions, context=context)

class ModelCount(NamedTuple):
    model: Model
    count: int

class Answer(NamedTuple):
    """
    Contains the return from a solve call to a logic program as a three tuple of the form:
            < result, statistics, models >
    Answers are immutable and so are all of their elements.
    
    Fields
    ------
    `result : Result` - The result of a solver call containing information about satisfiability.
    
    `statistics : Statistics` - The statistics from a solver call containing information about computation times.
    
    `base_models : list[Model]` - A list of stable models (answer sets) produce by a solver call containing sets of atoms (Clingo symbols).
    The list is ordered in the same ordinal sequence as the models were yielded by the solver, i.e. the first model is at index [0] and last model is at index [-1].
    If the logic program that returned this answer contained optimisation statements, then the last model is the, or one of, the optimal models.
    
    `inc_models : dict[int, list[Model]]` - 
    
    Example Usage
    -------------
    >>> import ASP_Parser as ASP
    >>> import clingo
    
    >>> logic_program = ASP.LogicProgram("a :- c. b :- not c. {c}.")
    >>> answer: ASP.Answer = logic_program.solve([ASP.SolveOpts.output(False), ASP.SolveOpts.Models(0)])
    >>> answer
    Answer(result=Result(satisfiable=True, exhausted=True),
           statistics=Statistics(grounding_time=0.00017879999998626772, solving_time=0.00012190000001055523, total_time=0.00030069999999682295),
           models=[Model(symbols=frozenset({b}), cost=[], optimality_proven=False, number=1, thread_id=0, model_type=StableModel),
                   Model(symbols=frozenset({c, a}), cost=[], optimality_proven=False, number=2, thread_id=0, model_type=StableModel)])
    
    >>> print(answer)
    Result :: SATISFIABLE : SEARCH SPACE EXHAUSTED : Total models = 2
    Statistics :: Grounding time = 0.000179s, Solving time = 0.000122s, Total time = 0.000301s
    Final Model :: Total atoms = 2, Cost = [], Optimality proven = False, Number = 2, Thread ID = 0, Model type = StableModel
    
    >>> print(*[f"Model {model.number} : {list(map(str, model.symbols))}" for model in answer.models], sep="\\n")
    Model 1 : [b]
    Model 2 : [c, a]
    
    Often only the final model yielded by the solver is required, especially when using optimisation statements.
    However, an unsatisfiable program will contain no models.
    To obtain the final model in a safe manner use the following, which returns an empty model if none exist in the answer;
    >>> answer.fmodel
    Model(symbols=frozenset({c, a}), cost=[], optimality_proven=False, number=2, thread_id=0, model_type=StableModel)
    
    For an unsatisfiable program;
    >>> logic_program = ASP.LogicProgram("a. -a.")
    >>> answer: ASP.Answer = logic_program.solve([ASP.SolveOpts.output(False), ASP.SolveOpts.Models(0)])
    >>> answer.fmodel
    Model(symbols=frozenset(), cost=[], optimality_proven=False, number=-1, thread_id=-1, model_type=None)
    """
    result: Result ## TODO Change to base_result and inc_result
    statistics: Statistics ## TODO Change to base_statistics and inc_statistics (that means we can get rid of "grand totals" in inc_statistics)
    base_models: Union[list[Model], ModelCount]
    inc_models: dict[int, Union[list[Model], ModelCount]]
    
    def __str__(self) -> str:
        return "\n".join([str(self.result) + f" : Total models = {self.total_models}",
                          str(self.statistics),
                          f"Final {self.fmodel}"])
    
    @property
    def models_counted(self) -> bool:
        return not isinstance(self.base_models, list)
    
    @property
    def total_models(self) -> int:
        return (self.base_models.count if self.models_counted else len(self.base_models)
                + sum(models.count if self.models_counted else len(models) for models in self.inc_models.values()))
    
    @property
    def fmodel(self) -> Model:
        "Get the final model in the answer, or an empty model (containing no atoms), if no models exist (i.e. the result was unsatisfiable)."
        fmodel: Optional[Model] = None
        if self.base_models:
            fmodel = self.base_models[-1]
        if self.inc_models:
            last_inc_models: Union[list[Model], ModelCount] = self.inc_models[max(self.inc_models)]
            if isinstance(last_inc_models, list):
                fmodel = last_inc_models[-1]
            else: fmodel = last_inc_models.model
        return fmodel if fmodel else Model([], [], False, 0)
    
    @staticmethod
    def dummy_answer() -> "Answer":
        "Constructs an empty answer. Useful as a default 'dummy' argument."
        return Answer(Result(False, False), Statistics(0.0, 0.0), Model([]))

#############################################################################################################################################
#############################################################################################################################################
################  ██████  ██████   ██████   ██████  ██████   █████  ███    ███     ██████   █████  ██████  ████████ ███████  ################
################  ██   ██ ██   ██ ██    ██ ██       ██   ██ ██   ██ ████  ████     ██   ██ ██   ██ ██   ██    ██    ██       ################
################  ██████  ██████  ██    ██ ██   ███ ██████  ███████ ██ ████ ██     ██████  ███████ ██████     ██    ███████  ################
################  ██      ██   ██ ██    ██ ██    ██ ██   ██ ██   ██ ██  ██  ██     ██      ██   ██ ██   ██    ██         ██  ################
################  ██      ██   ██  ██████   ██████  ██   ██ ██   ██ ██      ██     ██      ██   ██ ██   ██    ██    ███████  ################
#############################################################################################################################################
#############################################################################################################################################

@dataclass(frozen=True)
class IncRange(SubscriptableDataClass):
    """
    Represents the range of steps over which to ground the incremental program part(s) to which this range is assigned.
    A program part is grounded on every step that falls within the given optional start and end bound values, and is divisible by a given step jump value.
    
    Fields
    ------
    `start : {int, None}` - An integer defining the inclusive start bound of the range, or None if no start bound is desired.
    If not None, then the program part is grounded on all steps greater than or equal to the start bound.
    
    `end : {int, None}` - An integer defining the exclusive end of the range, or None if no end bound is desired.
    If not None, then the program part is grounded on all steps less than the end bound.
    
    `step : int` - An integer defining the step jump size.
    The program part is ground on all steps that can be divided by the step jump size a integer number of times.
    
    Example
    -------
    The following accepts all values greater than or equal to 5
    >>> range_ = IncRange(start=5)
    >>> range_.in_range(2) or range_.in_range(-9999)
    False
    >>> range_.in_range(5) and range_.in_range(8)
    True
    
    The following accepts all values less than 11 (less than or equal to 10)
    >>> range_ = IncRange(start=None, end=11)
    >>> range_.in_range(7) and range_.in_range(-9999)
    True
    >>> range_.in_range(11) or range_.in_range(14)
    False
    
    The following accepts all values divisible by 2 (even numbers)
    >>> range_ = IncRange(step=2)
    >>> range_.in_range(2) and range_.in_range(4)
    True
    >>> range_.in_range(3) or range_.in_range(5)
    False
    """
    start: Optional[int] = None
    end: Optional[int] = None
    step: int = 1
    
    def __str__(self) -> str:
        return f"Incremental range :: start = {self.start}, end = {self.end}, step = {self.step}"
    
    def in_range(self, step: int) -> bool:
        """
        Checks if the step value is in this incremental range.
        
        Parameters
        ----------
        `step : int` - The step value to check.
        
        Returns
        -------
        `bool` - True if the step value is in the range, False otherwise.
        """
        return ((self.start is None or step >= self.start)
                and (self.end is None or step < self.end)
                and (step % self.step == 0))

@dataclass(frozen=True, order=True)
class BasePart:
    """
    Represents a base program part as an immutable tuple of; a name and a sequence of arguments.
    
    The program part header is of the form:
        #program name(args[0], args[i], ... args[n]).
    
    Arguments are converted automatically to Clingo symbols before solving by Gringo's term parser.
    
    Fields
    ------
    `name : str` - A string specifying the name of the program part.
    
    `args : tuple[{str, int}] = ()` - A tuple of strings and integers specifying arguments for the program part's parameters.
    """
    name: str
    args: tuple[Union[str, int]] = field(default_factory=tuple)
    
    def __str__(self) -> str:
        return f"#program {self.name}({', '.join([str(arg) for arg in self.args])})."
    
    @property
    def is_valid(self) -> bool:
        return bool(self.name)
    
    @property
    def _clingo_form(self) -> tuple[str, list[Optional[clingo.Symbol]]]:
        """
        Converts the program part to the form required by Clingo.
        Incremental arguments with value '#inc' are converted to None and must be manually replaced by a Clingo number before grounding.
        """
        return (self.name, [(clingo.parse_term(str(arg)) if arg != "#inc" else None) for arg in self.args])
    
    @classmethod
    def from_string(cls, program_part: str) -> "BasePart":
        """
        Convert a string of the form 'name(args_0, args_i, ... args_n)' to a base part object.
        This does not check the validity of the syntax.
        """
        if '(' not in program_part:
            return BasePart(program_part)
        name, joined_args = program_part.split('(')
        split_args = joined_args.rstrip(')').split(',')
        return BasePart(name, tuple(split_args))

@dataclass(frozen=True, order=True)
class IncPart(BasePart):
    """
    Represents an incremental program part as an immutable tuple of; a name, a sequence of arguments, and an optional step range.
    To specify an argumentt to be replaced by an incremental step value specify it as '#inc'.
    
    The program part header is of the form:
        #program name(args[0], args[i], ... args[n]).
    
    Arguments are converted automatically to Clingo symbols before solving by Gringo's term parser.
    
    Fields
    ------
    `name : str` - A string specifying the name of the program part.
    
    `args : tuple[{str, int}] = ('#inc',)` - A tuple of strings and integers specifying arguments for the program part's parameters.
    Arguments with value '#inc' are replaced with the incremental step value during solving. By default a singleton tuple containing just the incremental step.
    
    `range_ : {IncRange, None} = None` - An optional incremental step range (see ASP_Parser.IncRange) to use this program part for.
    If not given or None, then the program part is used on all steps. By default None.
    """
    args: tuple[Union[str, int]] = ("#inc",)
    range_: Optional[IncRange] = None
    
    def __str__(self) -> str:
        return f"{super().__str__()} %* {str(self.range_)} *%"
    
    @BasePart.is_valid.getter
    def is_valid(self) -> bool:
        return super().is_valid and "#inc" in self.args

@dataclass(frozen=True)
class ProgramParts(SubscriptableDataClass):
    """
    A tuple used to store program parts.
    The program parts are stored in mutable lists that can be modified during solving.
    
    Fields
    ------
    `base_parts : list[BasePart]` - A list of base program parts.
    
    `inc_parts : list[IncPart] = []` - A list of incremental program parts.
    """
    base_parts: list[BasePart]
    inc_parts: list[IncPart] = field(default_factory=list)
    
    def get_clingo_form(self, inc_step: Optional[int] = None) -> list[tuple[str, list[clingo.Symbol]]]:
        clingo_parts: list[tuple[str, list[clingo.Symbol]]] = []
        for part in (self.base_parts if inc_step is None else self.inc_parts):
            if inc_step is None or part.range_ is None or part.range_.in_range(inc_step):
                clingo_part = part._clingo_form
                for index, arg in enumerate(clingo_part[1]):
                    if arg is None:
                        clingo_part[1][index] = clingo.Number(inc_step)
                clingo_parts.append(clingo_part)
        return clingo_parts

class External(NamedTuple):
    symbol: ASP_Symbol
    truth: Optional[bool] = None
    inc_range: Optional[IncRange] = None
    
    def convert_symbol(self, step: Optional[int] = None) -> clingo.Symbol:
        _symbol: ASP_Symbol = self.symbol
        if isinstance(self.symbol, str):
            if "$(step)" in self.symbol and step is None:
                raise ValueError("Cannot convert symbol containing step parameter if no step value is given.")
            _symbol = self.symbol.replace("$(step)", str(step))
        return to_clingo_form(_symbol)

#############################################################################################################################################
#############################################################################################################################################
##############  ██       ██████   ██████  ██  ██████     ██████  ██████   ██████   ██████  ██████   █████  ███    ███ ███████  ##############
##############  ██      ██    ██ ██       ██ ██          ██   ██ ██   ██ ██    ██ ██       ██   ██ ██   ██ ████  ████ ██       ##############
##############  ██      ██    ██ ██   ███ ██ ██          ██████  ██████  ██    ██ ██   ███ ██████  ███████ ██ ████ ██ ███████  ##############
##############  ██      ██    ██ ██    ██ ██ ██          ██      ██   ██ ██    ██ ██    ██ ██   ██ ██   ██ ██  ██  ██      ██  ##############
##############  ███████  ██████   ██████  ██  ██████     ██      ██   ██  ██████   ██████  ██   ██ ██   ██ ██      ██ ███████  ##############
#############################################################################################################################################
#############################################################################################################################################

class SolveSignal:
    """
    Solve signals are used to control incremental solve calls.
    They are returned from logic programs when calling:
        - LogicProgram.start(...)
        - LogicProgram.resume(...)
    """
    
    __slots__ = ("__program",       # {LogicProgram | None}
                 "__control",       # {clingo.Control | None}
                 "__inc_run",       # Callable[[int], Feedback]
                 "__running",       # bool
                 "__holding",       # bool
                 "__halt_reason")   # {HaltReason | None}
    
    def __init__(self, program: "LogicProgram", control: clingo.Control, inc_run: Callable[[int], "Feedback"]):
        self.__program: Optional[LogicProgram] = program
        self.__control: Optional[clingo.Control] = control
        self.__inc_run: Callable[[int], Feedback] = inc_run
        self.__running: bool = False
        self.__holding: bool = False
        self.__halt_reason: Optional[HaltReason] = None
    
    @property
    def logic_program(self) -> Optional["LogicProgram"]:
        """Get the underlying logic program this solve singal is controlling, or None if the signal is expended."""
        return self.__program
    
    @property
    def expended(self) -> bool:
        """Wether the solve signal is expended."""
        return self.__program is None
    
    @property
    def running(self) -> bool:
        """
        Whether the solve signal is currently running.
        Unlike the logic program, which is always running whilst in the solve signal's context,
        the solve signal itself is only running whilst within the loop of 'continue_running(...)'.
        """
        return self.__running
    
    @property
    def holding(self) -> bool:
        """
        If the solve signal is holding, when its context is left, and the signal is expended,
        the logic program will hold a saved grounding, from which a new signal can be spawned to continue the incremental solve.
        """
        return self.__holding
    
    @holding.setter
    def holding(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Holding must be set to a boolean value. Got; {value} of type {type(value)}.")
        if not self.expended:
            self.__holding = value
    
    def free(self, hold: bool = True) -> bool:
        """Free the solve signal from the underlying logic program, expending it."""
        if self.__running:
            raise RuntimeError("Cannot free a solve signal that is running.")
        if not self.expended:
            self.holding = hold
            self.__program = None
            self.__control = None
            return True
        return False
    
    @property
    def halt_reason(self) -> Optional["HaltReason"]:
        """
        Get the halt reason for the most reason solve call, or None if no call has been made.
        Raises an exception if the solve signal is in a running state.
        """
        if not self.__running:
            return self.__halt_reason
        raise RuntimeError("Cannot access halt reason of a running solve signal.")
    
    def get_answer(self, dummy: bool = False) -> Optional[Answer]:
        """
        Get the answer from the most recent solve call.
        This is a proxy for the method with the same name of the underlying logic program of this solve signal.
        
        Parameters
        ----------
        `dummy : bool = False` - A Boolean, by default False.
        If no solver call has yet been made by the underlying logic program since the creation of this solve signal then;
        True returns a 'dummy' answer object, and False returns None.
        Otherise, if at least one solve call has been made, this parameter is ignored, and the answer is always returned.
        
        Returns
        -------
        `{Answer | None}` - The answer, or None if no solver call has been made since the creation of this solve signal and `dummy` is False.
        """
        return self.__program.get_answer(dummy)
    
    def queue_assign_external(self, symbol: ASP_Symbol, truth: Optional[bool] = None, inc_range: Optional[IncRange] = None) -> None:
        """
        external
            A symbol representing the external atom.
        truth
            A Boolean to fix the external to the respective truth value; and None leaves its truth value open.
        inc_range
            The incremental step range over which this external should be assigned.
            If the incremental step range is None, then the external is assigned only once.
            To assign over all steps, set both the start and end steps of the incremental range to None.
        """
        self.__program.queue_assign_external(External(symbol, truth, inc_range))
    
    def assign_external(self, symbol: ASP_Symbol, truth: Optional[bool] = None) -> None:
        self.__control.assign_external(to_clingo_form(symbol), truth)
    
    def release_external(self, symbol: ASP_Symbol) -> None:
        self.__control.release_external(to_clingo_form(symbol))
    
    def online_extend_program(self, program_part: BasePart, program: str, context: Iterable[Callable[..., clingo.Symbol]] = []) -> None:
        """
        This method does not update the logic program's solving statistics directly.
        As such, the time spent grounding the program extension will not be represented by a statistics object extracted from that logic program.
        """
        _ASP_logger.debug(f"Solve signal {self!s} => Extending logic program {self.logic_program!s} to part {program_part!s} with:\n{program}")
        self.__control.add(program_part.name, list(map(str, program_part.args)), program)
        context_type = type("context", (object,), {func.__name__ : func for func in context})
        self.__control.ground([program_part._clingo_form], context_type)
    
    def run_for(self, callback: Callable[["Feedback"], None] = None, increments: Optional[int] = None) -> Answer:
        for feedback in self.yield_run(increments):
            if callback is not None:
                callback(feedback)
        return self.get_answer()
    
    def run_while(self, callback: Callable[["Feedback"], bool], /, increments: Optional[int] = None) -> Answer:
        """
        Continue running incremental solve calls, whilst a callback is true.
        
        Parameters
        ----------
        `callback: (p0: Feedback) -> bool` - A callback function, that takes a feedback object as its only argument, and returns a Boolean.
        After each solve increment, this function is called, if it returns True the solver continues, otherwise it is returns False the solver returns the answer of the most recent solve call.
        
        Returns
        -------
        `Answer` - The answer of the most recent solve call.
        
        Also see
        --------
        `Feedback` - Class containing feedback data for incremental solve calls.
        """
        ## Run the program incrementally,
        for feedback in self.yield_run(increments):
            ## Break when the callback is not True,
            if not callback(feedback): break
        ## Returning the answer.
        return self.get_answer()
    
    def yield_run(self, increments: Optional[int] = None) -> Iterator["Feedback"]:
        """
        Continue running incremental solve calls.
        This is a iterator generator function, yielding feedback objects for each incremental solving call.
        It blocks whilst the solver is running, yields when each increment ends, and raises StopIteration when either;
        a given number of increments have been run, or the solve reaches a halt reason.
        
        Parameters
        ----------
        `increments : {int | None}` - If given and not None, runs at most the given number of increments.
        The solver may run less increments if a halt reason is reached before this number of increments is reached.
        If not given or None, then the solver runs until a halt reason is reached, or solving is cancelled by a break statement.
        
        Yields
        ------
        `Feedback` - A feedback object for each solving increment call.
        
        Raises
        ------
        `RunTimeError` - If either the solve signal is already running, or has been expended.
        
        Also see
        --------
        `Feedback` - Class containing feedback data for incremental solve calls.
        """
        if self.running:
            raise RuntimeError("Cannot run a solve signal that is already running.")
        if self.expended:
            raise RuntimeError("Cannot run a solve signal that is expended.")
        
        self.__running = True
        feedback: Optional[Feedback] = None
        try:
            for feedback in self.__inc_run(increments):
                feedback = feedback
                yield feedback
        except GeneratorExit as exit:
            raise exit
        finally:
            if feedback is not None:
                self.__halt_reason = HaltReason.get_halt_reason(self.__program.incrementor, increments, feedback)
            self.__running = False

@enum.unique
class SolveResult(enum.Enum):
    """
    An enumeration containing solve results.
    These are used as stop conditions for inremental solving.
    The enumeration maps the conditions to their string representation used by Clingo.
    See `clingo.SolveResult` for details.
    
    Elements
    --------
    `Satisfiable = "SAT"` - Stop when the program is satisfiable, i.e. it has a model.
    
    `Unsatisfiable = "UNSAT"` - Stop when the program is unsatisfiable, i.e. it does not have a model.
    
    `Unknown = "UNKNOWN"` - Stop when search is interrupted (usually when the time limit is reached or an error occurs).
    """
    Satisfiable = "SAT"
    Unsatisfiable = "UNSAT"
    Unknown = "UNKNOWN"

@dataclass(frozen=True)
class Feedback(SubscriptableDataClass):
    """
    Represents the feedback from a completed increment of solving.
    Feedback objects are immutable dataclasses, whose fields can be accessed by index, making them act like tuples.
    
    Fields
    ------
    `increment : int` - The absolute ordinal incremental call number of the incremental solve call that produced this feedback object.
    
    `start_step : int` - The first step value reached by the most recent incremental solve call.
    
    `end_step : int` - The last step value reached by the most recent incremental solve call.
    
    `solve_result : SolveResult` - The solve result of the call.
    
    `cumulative_statistics : float` - The total running cumulative time over all incremental solve calls, including all previous calls and up to and including the most recent.
    
    `increment_time : float` - The total computation time spent on the most recent incremental solve call.
    """
    increment: int
    start_step: int
    end_step: int
    solve_result: SolveResult
    cumulative_statistics: Statistics
    increment_statistics: Statistics

@dataclass(frozen=True)
class SolveIncrementor:
    """
    Controls the incremental loop involved in incremental solving.
    It tells the logic program what step value to start at, and how much to increase the step value for each solving increment.
    It also defines when the inremental solving halts, by containing limits of the step value, increment count, solving time, etc.
    As standard, the incrementor increments from zero, in steps of one, and until the program is satisfiable.
    
    Fields
    ------
    `step_start : int = 0` - The initial step to start incrementing from.
    
    `step_increase : int = 1` - The number of steps to increase by on each increment.
    
    `step_increase_initial : int = 1` - An override for the number of steps to increase by on the initial increment.
    
    `step_end_min : {int | None} = None` - The minimum step value that must be reached before returning from an incremental solve call.
    
    `step_end_max : {int | None} = None` - The maximum step value that can not be exceeded during an incremental solve call.
    
    `stop_condition : {SolveResult | None} = SolveResult.Satisfiable` - The stop condition of the solve call, return when this is satisfied and the minimum step bound is reached.
    
    `increment_limit : {int | None} = None` - The maximum number of solver increments.
    
    `increment_time_limit : {int | None} = None` - The maximum time limit in seconds of a single incremental solve call.
    
    `cumulative_time_limit : {int | None} = None` - The time limit in seconds on the total cumulative computation time summed over all incremental solve calls.
    
    `preempt : bool = False` - Whether to check if the time limits have been reached preemptively.
    If True, preemptive checking is enabled, this runs a 'busy waiting' loop during each solving increment.
    This preemptive loop checks if the time limit once per second, and cancels solving if it has been reached.
    Otherwise, if False, preemptive checking is diabled, and the time limit is checked only after each solving increment ends.
    Preemptive checking has a small overhead, but is a more accurate way to check the limits.
    """
    step_start: int = 0
    step_increase: int = 1
    step_increase_initial: int = 1
    step_end_min: Optional[int] = None
    step_end_max: Optional[int] = None
    stop_condition: Optional[SolveResult] = SolveResult.Satisfiable
    increment_limit: Optional[int] = None
    increment_time_limit: Optional[int] = None
    cumulative_time_limit: Optional[int] = None
    preempt: bool = False
    
    def __post_init__(self) -> None:
        ## Check the the stop condition is valid
        if self.stop_condition is not None and self.stop_condition not in SolveResult:
            raise ValueError(f"The stop condition '{self.stop_condition}' is not valid, see 'ASP_Parser.SolveResult'.")
        
        ## Check that the step increments are greater than zero
        if self.step_increase < 1 or self.step_increase_initial < 1:
            raise ValueError("Step increments must greater than zero. "
                             f"Got; step increase = {self.step_increase} and step increase initial = {self.step_increase_initial}.")
        
        ## Check that the step end values are greater than the step start value
        if ((self.step_end_min is not None and self.step_end_min <= self.step_start)
            or (self.step_end_max is not None and self.step_end_max <= self.step_start)):
            raise ValueError("The step end values must be greater than the start step value. "
                             f"Got; start_step = {self.step_start}, step_end_min = {self.step_end_min} and step_end_max = {self.step_end_max}.")
        
        ## Check the solve call has a return condition
        if (self.step_end_max is None
            and self.stop_condition is None
            and self.increment_limit is None
            and self.increment_time_limit is None
            and self.cumulative_time_limit is None):
            _ASP_logger.debug("Incremental solving with max end step, stop condition, increment limit, and time limits as None, may cause the solver to never return.")

class HaltReasonValue(NamedTuple):
    description: str
    is_reached: Callable[[SolveIncrementor, Optional[int], Feedback], bool]

@enum.unique
class HaltReason(enum.Enum):
    """
    An enumeration defining the possible reasons for a logic program incremental solve halting.
    The value of each item is a tuple, containing a description and a function that returns True if the reason was satisfied.
    The reasons are resolved in the order given below.
    If multiple reasons are satisfied, only the first will be returned.
    
    If the time limit is reached before the minimum step limit is reached and the program is satisfiable,
    it will erroneously report "stop condition reached", whereas it should return "cumulative/incremental time limit reached".
    
    Items
    -----
    `StepMaximum = (description="Step end maximum reached")` - If the step end maximum value was reached.
    
    `StopCondition = (description="Stop condition reached")` - If the stop condition was reached, see 'ASP_Parser.SolveResult'.
    
    `IncrementCount = (description="Increment count reached")` - If the increment count limit was reached.
    
    `TimeLimit = (description="Time limit reached")` - If either the incremental or cumulative time limits were reached.
    """
    StepMaximum = HaltReasonValue(description="Step end maximum reached",
                                  is_reached=lambda incrementor, increment_limit, feedback:
                                      incrementor.step_end_max is not None
                                      and incrementor.step_end_max <= feedback.end_step)
    
    StopCondition = HaltReasonValue(description="Stop condition reached",
                                    is_reached=lambda incrementor, increment_limit, feedback:
                                        incrementor.stop_condition is not None
                                        and feedback.solve_result == incrementor.stop_condition)
    
    IncrementCount = HaltReasonValue(description="Increment count reached",
                                     is_reached=lambda incrementor, increment_limit, feedback:
                                         (incrementor.increment_limit is not None
                                          and incrementor.increment_limit <= feedback.increment)
                                         or (increment_limit is not None
                                             and increment_limit <= feedback.increment))
    
    TimeLimit = HaltReasonValue(description="Time limit reached",
                                is_reached=lambda incrementor, increment_limit, feedback:
                                    (incrementor.increment_time_limit is not None
                                     and incrementor.increment_time_limit <= feedback.increment_statistics.total_time)
                                    or (incrementor.cumulative_time_limit is not None
                                        and incrementor.cumulative_time_limit <= feedback.cumulative_statistics.total_time))
    
    def is_reached(self, incrementor: SolveIncrementor, increment_limit: Optional[int], feedback: Feedback) -> bool:
        return self.value.is_reached(incrementor, increment_limit, feedback)
    
    @classmethod
    def get_halt_reason(cls, incrementor: SolveIncrementor, increment_limit: Optional[int], feedback: Feedback) -> Optional["HaltReason"]:
        for reason in cls:
            if reason.is_reached(incrementor, increment_limit, feedback):
                return reason
        return None

@dataclass
class Bounds:
    """Dataclass for storing the step bounds used in incremental solving."""
    increment: int
    previous_step: int
    current_step: int

class LogicProgram:
    """
    An ASP non-monotonic logic program.
    """
    
    __slots__ = (## The program's AST itself
                 "__program",               # list[clingo.ast.AST]
                 
                 ## Variables for logging and CLI outputs
                 "__name",                  # str
                 "__logger",                # logging.logger
                 "__verbosity",             # int
                 "__warnings",              # bool
                 "__message_limit",         # int
                 "__tqdm",                  # bool
                 
                 ## Variables for holding state of incremental solver calls
                 "__control",               # {clingo.Control | None}
                 "__running",               # bool
                 "__bounds",                # Bounds
                 "__queued_externals",      # list[External]
                 
                 ## Variables for holding inputs
                 "__assumptions",           # list[clingo.Symbol]
                 "__context",               # type
                 "__incrementor",           # {SolveIncrementor | None}
                 "__program_parts",         # ProgramParts
                 "__time_limit",            # int
                 "__count_multiple_models", # bool
                 
                 ## Variables for holding outputs
                 "__satisfiable",           # Optional[bool] if None then no solve call has yet been made
                 "__exhausted",             # bool
                 "__base_stats",            # Statistics
                 "__cumulative_stats",      # Statistics
                 "__incremental_stats",     # dict[int, Statistics]
                 "__models",                # list[Model]
                 "__model_count",           # int
                 "__process")               # {psutil.Process | None}
    
    @classmethod
    def __get_instance_number(cls, name: str) -> int:
        if not hasattr(cls, f"__{cls.__name__}_instance_counter"):
            cls.__instance_counter: dict[str, int] = {}
        cls.__instance_counter[name] = cls.__instance_counter.get(name, 0) + 1
        return cls.__instance_counter[name]
    
    def __init__(self,
                 program: str,
                 name: Optional[str] = None,
                 silent: bool = True,
                 warnings: bool = True,
                 message_limit: int = 20,
                 enable_tqdm: bool = False):
        """
        Instantiate an ASP logic program from its string representation.
        
        Parameters
        ----------
        `program : str` - A string defining an ASP program.
        
        `name : {str | None} = None` - A string defining an arbitrary name for this program.
        The string is converted to the form '<name> #n' where 'n' is an ordinal number.
        If not given or None, the program is given a name of the form 'Anonymous #n'.
        
        `silent : bool = True` - A Boolean, True to log output from this logic program at debug level
        (warning messages are logged seperately, see below), otherwise False to log information at info level.
        
        `warnings : bool = True` - A Boolean, True to log warnings at warning level, other False to log then at debug level.
        
        `message_limit : int = 20` - A natural cardinal number defining the quantity of warning
        messages that can be passed from Clingo to the Python side logger of this logic program.
        
        `enable_tqdm : bool = False` - A Boolean, True to enable animated tqdm progress bars
        in the CLI from solver calls made to this logic program, False otherwise.
        
        Raises
        ------
        `TypeError` -
        
        `RuntimeError` - If an error occurs whilst parsing the raw logic program code.
        It is advised to ensure that Clingo warnings are enabled when using a logic program for the first time.
        """
        _ASP_logger.debug("Attempting to instantiate new logic program:\n\t"
                          + "\n\t".join(str(arg) for arg in locals().items() if arg[0] not in ["self", "program"]))
        
        ## Variables for logging and CLI output
        self.__name: str = f"{name if name else 'Anonymous'} #{self.__class__.__get_instance_number(str(name))}"
        self.__logger: logging.Logger = logging.getLogger(f"Logic Program {self.__name}")
        self.__verbosity: int = logging.DEBUG if silent else logging.INFO
        self.__warnings: bool = warnings
        self.__message_limit: int = message_limit
        self.__tqdm: bool = enable_tqdm
        
        ## Create the logic program's Abstract Syntax Tree (AST)
        self.__program: list[clingo.ast.AST] = []
        try:
            self.__logger.debug(f"Parsing raw logic program code:\n{program}")
            clingo.ast.parse_string(program, lambda statement: self.__program.append(statement),
                                    logger=self.__catch_clingo_log, message_limit=self.__message_limit)
        except RuntimeError as error:
            self.__logger.error("Failed to parse logic program code. "
                                "Your program may contain a syntax error. "
                                "Ensure that Clingo warnings are enabled to see details.",
                                exc_info=1)
            raise error
        self.__logger.debug(f"Logic program code parsed successfully.")
        
        ## Variables for holding state of incremental solver calls
        self.__control: Optional[clingo.control.Control] = None
        self.__running: bool = False
        self.__bounds: Optional[Bounds] = None
        self.__queued_externals: list[External] = []
        
        ## Set internal variables for storing inputs and outputs to solve calls
        self.__set_inputs()
        self.__reset_outputs()
        
        self.__logger.debug(f"Logic program instantiated successfully.")
    
    @classmethod
    def from_files(cls, file_paths: Union[str, Iterable[str]], name: Optional[str] = None,
                   silent: bool = False, warnings: bool = True,
                   message_limit: int = 20, enable_tqdm: bool = False) -> "LogicProgram":
        """
        Instantiate an ASP logic program from the rules contained in a list of files.
        
        Parameters
        ----------
        `file_paths : {str | Iterable[str]}` - A string or iterable of strings, defining the file paths to load.
        
        `name : {str | None} = None` - A string defining an arbitrary name for this program.
        The string is converted to the form '<name> #n' where 'n' is an ordinal number.
        If not given or None, the program is given a name of the form 'Anonymous #n'.
        
        `silent : bool = True` - A Boolean, True to log output from this logic program at debug level
        (warning messages are logged seperately, see below), otherwise False to log information at info level.
        
        `warnings : bool = True` - A Boolean, True to log warnings at warning level, other False to log then at debug level.
        
        `message_limit : int = 20` - A natural cardinal number defining the quantity of warning
        messages that can be passed from Clingo to the Python side logger of this logic program.
        
        `enable_tqdm : bool = False` - A Boolean, True to enable animated tqdm progress bars
        in the CLI from solver calls made to this logic program, False otherwise.
        
        Returns
        -------
        `LogicProgram` - A logic program consisting of the rules from the loaded files.
        """
        program: str = ""
        _file_paths: list[str]
        
        if isinstance(file_paths, str):
            _file_paths = [file_paths]
        else: _file_paths = list(file_paths)
        
        for file_path in _file_paths:
            with open(file_path, "r") as file_reader:
                program += file_reader.read()
        
        return cls(program, name, silent, warnings, message_limit, enable_tqdm)
    
    def copy(self, rename: Optional[str] = None) -> "LogicProgram":
        """
        Create an exact copy of this logic program.
        
        Parameters
        ----------
        `rename : {str | None}` - A string to name the copy of the program.
        If not given or None, then the program is named "Copy of '<original name>' #<n>" where n is an ordinal number.
        
        Returns
        -------
        `LogicProgram` - The copy of this logic program.
        """
        self.__logger.debug(f"Creating copy of self: rename = {rename}")
        logic_program = LogicProgram("", name=rename if rename else f"Copy of <{self.name}>",
                                     silent=(self.__verbosity == logging.DEBUG), warnings=self.__warnings,
                                     message_limit=self.__message_limit, enable_tqdm=self.__tqdm)
        logic_program.__program = self.__program.copy()
        return logic_program
    
    
    
    def __str__(self) -> str:
        return f"Logic Program {self.__name}"
    
    def __repr__(self) -> str:
        program_string: str = "\n".join(map(str, self.__program))
        return f"{self.__class__.__name__}({program_string}, {self.__name})"
    
    @property
    def program(self) -> list[clingo.ast.AST]:
        """Get the abstract syntax tree of this logic program."""
        return self.__program
    
    @property
    def name(self) -> str:
        """Get the name of this logic program as a string of the form '<name> #n' where 'n' is an ordinal number."""
        return self.__name
    
    @property
    def incrementor(self) -> Optional[SolveIncrementor]:
        """Get the current solve incrementor being used by this program. Returns None of the program is not running or holding."""
        return self.__incrementor
    
    @property
    def bounds(self) -> Bounds:
        """The current bounds of the held incremental grounding."""
        return self.__bounds
    
    @property
    def program_parts(self) -> ProgramParts:
        return self.__program_parts
    
    @property
    def running(self) -> bool:
        """
        Whether the program is currently running a solve call.
        A program that is running cannot be solved, started, or resumed.
        """
        return self.__running
    
    @property
    def holding(self) -> bool:
        """
        Whether the logic program is currently holding a saved incremental grounding and is not running.
        A program that is holding can be resumed.
        """
        return not self.__running and self.__control is not None
    
    def get_answer(self, dummy: bool = False) -> Optional[Answer]:
        """
        Get the answer from the most recent solve call.
        
        Parameters
        ----------
        `dummy : bool = False` - A Boolean, if no solver call has yet been made by this logic program then;
        True returns a dummy answer object, and False returns None.
        
        Returns
        -------
        `{ASP.Answer | None}` - The answer, or None if no solver call has been made and `dummy` is False.
        """
        if self.__satisfiable is None:
            return Answer.dummy_answer() if dummy else None
        result: Result = Result(self.__satisfiable, self.__exhausted)
        statistics: Statistics = self.__base_stats
        if self.__incremental_stats:
            statistics = IncrementalStatistics(self.__base_stats.grounding_time,
                                               self.__base_stats.solving_time,
                                               self.__base_stats.total_time,
                                               self.__base_stats.memory,
                                               clingo_stats=self.__base_stats.clingo_stats,
                                               cumulative=self.__cumulative_stats,
                                               incremental={index + 1 : stat for index, stat in enumerate(self.__incremental_stats)})
        base_models: Union[list[Model], ModelCount] = ((self.__models[None].copy()
                                                        if not self.__model_count else
                                                        ModelCount(self.__models[None][-1], self.__model_count[None]))
                                                       if self.__models else
                                                       [Model([], [], False, -1, -1, None)])
        inc_models: Union[list[Model], ModelCount] = {inc : (models.copy()
                                                             if not self.__model_count else
                                                             ModelCount(models[-1], self.__model_count[inc]))
                                                      for inc, models in self.__models.items()
                                                      if inc is not None}
        return Answer(result, statistics, base_models, inc_models)
    
    def free(self) -> bool:
        """
        Free the program's held incremental grounding.
        
        Returns
        -------
        `bool` - True if there was a held grounding that was freed, otherwise False.
        """
        if self.__running:
            raise RuntimeError("Cannot free a program that is running.")
        
        if self.holding:
            self.__logger.debug("Freeing held grounding...")
            
            ## Get rid of the held grounding, incremental bounds progress, and queued externals
            self.__control = None
            self.__bounds = None
            self.__queued_externals = []
            
            ## Reset all input storing variables
            self.__set_inputs()
            
            self.__logger.debug("Held grounding freed successfully.")
            return True
        
        return False
    
    def set_incrementor(self, incrementor: SolveIncrementor) -> Optional[SolveIncrementor]:
        """
        Set the solve incrementor of this logic program and return the old one.
        
        Parameters
        ----------
        `incrementor : SolveIncrementor` - The new solve incrementor to assign to the logic program.
        
        Returns
        -------
        `{SolveIncrementor | None}` - The old solve incrementor if the logic program is currently holding a grounding, otherwise None.
        
        Raises
        ------
        `RunTimeError` - If the logic program is currently running.
        """
        if self.__running:
            raise RuntimeError("Cannot set the incrementor of a running logic program.")
        self.__logger.debug(f"Setting incrementor to: {incrementor}")
        old_incrementor: SolveIncrementor = self.__incrementor
        self.__incrementor = incrementor
        return old_incrementor
    
    def modify_program_parts(self, add: Union[IncPart, Iterable[IncPart]], remove: Union[IncPart, Iterable[IncPart]] = []) -> None:
        """
        Modify the program parts of a held grounding.
        """
        if not self.holding:
            raise RuntimeError("Cannot modify the program parts of a logic program that is not holding a saved grounding.")
        
        inc_parts: list[IncPart] = self.__program_parts.inc_parts
        
        if isinstance(add, Iterable):
            inc_parts += list(add)
        else: inc_parts += [add]
        
        for part in remove if isinstance(remove, Iterable) else [remove]:
            if part in inc_parts:
                inc_parts.remove(part)
        
        self.__set_program_parts(self.__program_parts.base_parts, inc_parts)
    
    def queue_assign_external(self, external: External) -> None:
        """
        Queue an external atom to be assigned by this logic program during incremental solving.
        
        External atoms can be queued up prior to starting a solve call, but will be cleared when the held grounding is freed.
        
        Parameters
        ----------
        `external : External` - The external atom to queue for assignment.
        
        Also see
        --------
        `External` - Class for encapsulating external atom assignment.
        
        `SolveSignal.queue_assign_external(...)` - Similar method for queueing externals during incremental solving using a solve signal.
        """
        self.__logger.debug(f"Queueing external: {external}")
        self.__queued_externals.append(external)
    
    def __set_inputs(self, solver_options: list[str] = [], count_multiple_models: bool = False, assumptions: list[clingo.Symbol] = [], context: Iterable[Callable[..., clingo.Symbol]] = [], incrementor: Optional[SolveIncrementor] = None,
                     base_parts: Union[BasePart, Iterable[BasePart]] = BasePart(name="base"), inc_parts: Union[IncPart, Iterable[IncPart]] = IncPart(name="step")) -> None:
        """
        Sets all internal variables used for holding input parameters.
        
        Finds and saves the time limit in the list of solver options.
        
        Internal used only, this method should not be called from outside this class.
        """
        args: list[str] = [f"{key} = {item}" for key, item in locals().items() if key != "self"]
        self.__logger.debug("Setting input storing variables:\n\t" + "\n\t".join(args))
        
        ## Variables for solver options TODO Is the time limit per solve call or cumulative across all solve calls?
        self.__time_limit: Optional[int] = None
        for option in solver_options:
            if option.startswith("--time-limit="):
                self.__time_limit = int(option.split("=")[1])
        
        ## Variables for universal solving parameters
        self.__assumptions: list[clingo.Symbol] = assumptions
        self.__context: type = type("context", (object,), {func.__name__ : func for func in context})
        self.__set_program_parts(base_parts, inc_parts)
        self.__count_multiple_models: bool = count_multiple_models
        
        ## Variables for incremental solving parameters
        self.__incrementor: Optional[SolveIncrementor] = incrementor
        
        self.__logger.debug("Input storing variables set.")
    
    def __set_program_parts(self, base_parts: Union[BasePart, Iterable[BasePart]] = BasePart(name="base"), inc_parts: Union[IncPart, Iterable[IncPart]] = IncPart(name="step")) -> None:
        """
        Set the program parts to be grounded by the current solve call.
        
        Internal used only, this method should not be called from outside this class.
        """
        PT = TypeVar("PT", bound=BasePart)
        def to_list(parts: Union[PT, Iterable[PT]], type_: Type[PT]) -> list[PT]:
            "Function for converting program parts to lists."
            _parts: list[PT]
            if isinstance(parts, Iterable):
                _parts = list(parts)
            else: _parts = [parts]
            for part in _parts:
                if not isinstance(part, type_):
                    raise ValueError(f"Program part must be of type {type_}. Got; {part} of type {type(part)}.")
            return _parts
        
        self.__program_parts = ProgramParts(base_parts=to_list(base_parts, BasePart),
                                            inc_parts=to_list(inc_parts, IncPart))
    
    def __reset_outputs(self) -> None:
        """
        Resets all internal variables used for holding answers.
        
        Internal used only, this method should not be called from outside this class.
        """
        self.__logger.debug("Resetting output storing variables.")
        
        ## Variables for results
        self.__satisfiable: Optional[bool] = None
        self.__exhausted: bool = False
        
        ## Variables for statistics
        self.__base_stats: Optional[Statistics] = None
        self.__cumulative_stats: Optional[Statistics] = None
        self.__incremental_stats: list[Statistics] = []
        self.__process: Optional[psutil.Process] = None
        
        ## Variables for models
        self.__models: dict[Union[int, None], list[Model]] = {}
        self.__model_count: dict[Union[int, None], int] = {}
        
        self.__logger.debug("Output storing variables reset.")
    
    def add_rules(self, rules: Union[ASP_Symbol, Atom, Iterable[Union[ASP_Symbol, Atom]]], program_part: Optional[str] = "base", context: Iterable[Callable[..., clingo.Symbol]] = [], permanent: bool = False) -> int:
        """
        Extend this logic program with the specified rule or iterable of rules given either as; strings, clingo symbols, or Atom objects, and return the number of rules added.
        Rules may be full standard, weak constraint, or heuristic (where the latter must be appended with their cost and term tuples) rule, or just atoms.
        If the rules do not end with a period, and they are not a weak constraint or heuristic, then a period is added automatically.
        This is useful for inserting atoms in this logic program, extracted from other programs or solve calls, via ASP_Parser.Model.get_atoms(...).
        
        Parameters
        ----------
        `rules : {str | clingo.Symbol | Atom | Iterable[{str | clingo.Symbol | Atom}]}` - Either a string, a Clingo symbol, an Atom, or an iterable of any, defining the rule(s) to add to the program.
        
        `program_part : {str | None} = "base"` - Either None or an non-empty alphanumeric string which may contain underscores whose leading character must be a lowercase letter, defining the name of the program part to insert the rules into.
        The program part does not have to already be in the program. If None then the rules are simply appended to the logic program and will be contained in the last program part added to the program, this is not recommended.
        When extending offline, that program part arguments act as parameter declarations.
        When extending online the program parts arguments act as normally as arguments they would when making a solve call.
        
        Returns
        -------
        `int` - An integer defining the number of rules that were added to the program.
        
        Raises
        ------
        `ValueError` - If either:
            - The program part is not None and, is either empty or has invalid syntax,
            - A rule is not a string, clingo symbol or Atom,
            - A rule has invalid syntax according to Gringo's term parser.
        """
        if self.__running:
            raise RuntimeError("Cannot add rules to a logic program that is running.")
        
        ## Inner variables.
        program_extension: str = ""
        quantity: int = 0
        
        ## Check that the program part;
        ##      - contains only letters and,
        ##      - starts with a lowercase letter.
        if program_part is not None and (re.match("[a-zA-Z0-9_]*", program_part) is None or not program_part[0].islower()):
            raise ValueError(f"The progarm part name {program_part} is not valid, "
                             "it must be a non-empty string containing only letters and its leading letter must be lowercase.")
        
        def add_rule(rule: Union[ASP_Symbol, Atom]) -> None:
            "Function for processing individual rules."
            nonlocal program_extension
            if isinstance(rule, (clingo.Symbol, Atom)):
                program_extension += f"\n{rule}."
            elif isinstance(rule, str):
                ## Add the rule directly if it ends with a period or close bracket;
                ##      - A rule with a period is a standard rule,
                ##      - A rule with a close bracket is either a weak constraint or heuristic.
                if rule.endswith(('.', ']')):
                    program_extension += "\n" + rule
                ## Otherwise, add the rules with a period appended.
                else: program_extension += f"\n{rule}."
            else: raise ValueError(f"The rule '{rule}' of type {type(rule)} is invalid, must be of type str or clingo.Symbol.")
            nonlocal quantity
            quantity += 1
        
        ## If the input was not a string but is an iterable;
        ##      - then add each element to the program,
        ##      - otherwise add the given rule to the program.
        if not isinstance(rules, (str, Atom)) and isinstance(rules, Iterable):
            self.__logger.debug(f"Extending program part {program_part} with rules:\n"
                                + "\n".join(repr(rule) for rule in rules))
            for rule in rules:
                add_rule(rule)
        else:
            self.__logger.debug(f"Extending program part {program_part} with rule:\n{rules!r}")
            add_rule(rules)
        
        ## If no new rules were added then;
        ##      - warn the user,
        ##      - otherwise add the rules to the program.
        if not program_extension:
            self.__logger.log(logging.WARNING if self.__warnings else logging.DEBUG,
                              "The logic program was not extended by a call to: 'LogicProgram.add_rules(rules, program_part)'.")
        else:
            try:
                if not self.holding or permanent:
                    self.__logger.debug("Extending program AST permenantly.")
                    clingo.ast.parse_string(f"#program {program_part}.{program_extension}" if program_part is not None else f"{program_extension}",
                                            lambda statement: self.__program.append(statement), logger=self.__catch_clingo_log, message_limit=self.__message_limit)
                else:
                    self.__logger.debug(f"Extending program online, this change will {'not' if not permanent else ''} persist across solve calls.")
                    base_part: BasePart = BasePart.from_string(program_part)
                    self.__control.add(base_part.name, list(map(str, base_part.args)), program_extension)
                    context_type = type("context", (object,), {func.__name__ : func for func in context})
                    self.__control.ground([base_part._clingo_form], context_type)
            except RuntimeError as error:
                self.__logger.error("Failed to parse rules, they may contain syntax errors. "
                                    "Ensure that Clingo warnings are enabled to see details.",
                                    exc_info=1)
                raise error
        
        self.__logger.debug(f"The logic program was extended with {quantity} rules into program part {program_part}:\n{program_extension}")
        
        ## Return the quantity of rules added to the program
        return quantity
    
    def __create_control(self, solver_options: Iterable[str]) -> None:
        """
        Create a clingo control instance.
        """
        try:
            self.__control = clingo.control.Control(solver_options,
                                                    logger=self.__catch_clingo_log,
                                                    message_limit=self.__message_limit)
        except RuntimeError as error:
            self.__logger.error("An error occurred whilst attempting to make a clingo session. "
                                "Your solver options may be invalid, consult the clingo documentation.",
                                exc_info=1)
            raise error
    
    def solve(self, solver_options: Iterable[str] = [], count_multiple_models: bool = False, assumptions: Iterable[clingo.Symbol] = [], context: Iterable[Callable[..., clingo.Symbol]] = [], solve_incrementor: Optional[SolveIncrementor] = None,
              base_parts: Union[BasePart, Iterable[BasePart]] = BasePart(name="base"), inc_parts: Union[IncPart, Iterable[IncPart]] = IncPart(name="step")) -> Answer:
        """
        Directly solve this logic program either in one-shot or incremental mode.
        A standard one-shot solve is ran if no solve incrementor is given (the default).
        Otherwise an incremental solve is, an incremental solve is ran, and the first model generated will be the solution to the base program.
        If the base program is unsatisfiable, this function returns without incremental solving.
        
        Parameters
        ----------
        `solver_options : Iterable[str]` - An iterable of strings defining options for clingo.
        
        `assumptions : Iterable[clingo.Symbol] = []` - An iterable of clingo symbols given as assumptions to the solver.
        
        `context : Any = None` - Should contain functions of the form Callable[[Any], clingo.Symbol]
        
        `solve_incrementor : {SolveIncrementor | None} = None` - A solve incrementor, used for running an incremental solve call.
        If not given or None, then a standard one-shot solve call of only the base program parts is run.
        Otherwise, if given, the one-shot solve of the base parts is made, then the incremental parts are solved incrementally according to the solve incrementor.
        
        `base_parts : Iterable[BasePart] = [BasePart(name="base", args=[])]` - An iterable of base parts to solve.
        
        `inc_parts : Iterable[IncPart] = [IncPart(name="step", args=["#inc"], range_=None)]` - An iterable of incremental program parts to solve on every solver increment.
        
        Returns
        -------
        `Answer` - The answer of the solve call.
        """
        if self.running:
            raise RuntimeError("Cannot directly solve a logic program that is running.")
        
        incremental: bool = solve_incrementor is not None
        self.__logger.log(self.__verbosity, f"Entering new direct {'incremental' if incremental else 'one-shot'} solve call.")
        if self.holding: self.free()
        self.__running = True
        
        ## Set general solving parameters and reset the internal output storing variables
        self.__set_inputs(solver_options,
                          count_multiple_models,
                          list(assumptions),
                          context,
                          solve_incrementor,
                          base_parts,
                          inc_parts)
        self.__reset_outputs()
        
        try:
            ## Create a Clingo control object with given solver options
            self.__create_control(solver_options)
            
            ## Solve the base program;
            ##      - Build the logic program's AST,
            ##      - Run the base program parts.
            self.__build()
            self.__base_run()
            
            ## Attempt to solve the logic program incrementally
            if incremental:
                
                ## Return if the base program has no solution
                if not self.__satisfiable:
                    self.__logger.log(self.__verbosity, "Unable to incrementally solve the logic program, the base program is unsatisfiable.")
                    return self.get_answer(dummy=True)
                
                ## Otherwise continue to the incremental solve
                feedback_list: list[Feedback] = []
                for feedback in self.__inc_run():
                    feedback_list.append(feedback)
                self.__logger.debug(f"Incremental feedback:\n"
                                    + "\n".join([f" {feedback.increment:<5} || {', '.join(map(str, feedback[1:]))}" for feedback in feedback_list]))
            
        finally:
            ## Free the control object
            self.__running = False
            self.free()
        
        ## Generate, log and return the answer
        answer = self.get_answer(dummy=True)
        self.__logger.log(self.__verbosity, f"Returning from {'incremental' if incremental else 'standard'} solve call:\n{answer}")
        return answer
    
    @contextlib.contextmanager
    def start(self, solver_options: Iterable[str] = [], count_multiple_models: bool = False, assumptions: Iterable[clingo.Symbol] = [], context: Iterable[Callable[..., clingo.Symbol]] = [], solve_incrementor: SolveIncrementor = SolveIncrementor(),
              base_parts: Union[BasePart, Iterable[BasePart]] = BasePart(name="base"), inc_parts: Union[IncPart, Iterable[IncPart]] = IncPart(name="step")) -> "SolveSignal":
        """
        Start new incremental solving call.
        If one is already contained and partially progressed, then it is restarted.
        They declare multiple context manager factory functions that make incremental solve calls to the encapsulated logic program.
        The factory functions return a solve signal, which can be used to make and obtain feedback to and from incremental solve calls.
        When the context manager is left, the execution of the program is paused and can be restarted.
        
        The base parts get grounded once and only once, so modifying them by add rules to those parts via "LogicProgram.add_rules(...)" will only take affect if the solve call is restarted.
        To add rules online, such that they are inserted and grounded to take affect during the current call, use "SolveSignal.online_extend_program(...)" instead.
        
        If this logic programs is holding a saved grounding, it will be freed.
        
        Parameters
        ----------
        See `LogicProgram.solve` for parameter descriptions.
        
        Yields
        ------
        `SolveSignal` - A solve signal object for controlling the incremental solve.
        
        Raises
        ------
        `RuntimeError` - If either;
            - the program is already running.
            - the base program is unsatisfiable and hence a solve signal cannot be created.
        """
        if self.__running:
            raise RuntimeError("Cannot start a controllable solve on a logic program that is running.")
        
        self.__logger.log(self.__verbosity, "Starting new controllable incremental solve call.")
        if self.holding: self.free()
        self.__running = True
        
        ## Set general solving parameters and reset the internal output storing variables
        self.__set_inputs(solver_options,
                          count_multiple_models,
                          list(assumptions),
                          context,
                          solve_incrementor,
                          base_parts,
                          inc_parts)
        self.__reset_outputs()
        
        try:
            ## Create a Clingo control object with given solver options
            self.__create_control(solver_options)
            
            ## Solve the base program;
            ##      - Build the logic program's AST,
            ##      - Run the base program parts.
            self.__build()
            self.__base_run()
            
            ## Raise an error if the base program has no solution
            if not self.__satisfiable:
                raise RuntimeError("Unable to create solve signal, the base program is unsatisfiable.")
            
            ## Attempt to yield a solve signal
            solve_signal: SolveSignal = self.__create_solve_signal()
            yield solve_signal
            
        finally:
            ## Free the held grounding unless holding is enabled
            self.__running = False
            if not solve_signal.holding or not self.__satisfiable:
                self.__logger.log(self.__verbosity, "Stopping incremental solve call.")
                self.free()
            else: self.__logger.log(self.__verbosity, "Pausing incremental solve call.")
            solve_signal.free(hold=False)
    
    @contextlib.contextmanager
    def resume(self, solve_incrementor: Optional[SolveIncrementor] = None, maintain_step: bool = True) -> "SolveSignal":
        """
        Resume an incremental solve call with this logic program's held grounding.
        Calling this before any has been started is an error.
        
        Parameters
        ----------
        `solve_incrementor: {SolveIncrementor | None}` - If given and not None, specifies a new solve incrementor to be used on the resumed program.
        
        `maintain_step: bool = True` - Whether to continue incrementing from the previously reached step when the program was paused, or to start from the initial step again.
        
        Yields
        ------
        `SolveSignal` - A solve signal object for controlling the incremental solve.
        
        Raises
        ------
        `RuntimeError` - If the program is either already running or is not holding a saved grounding.
        """
        if self.__running:
            raise RuntimeError("Cannot resume a controllable solve on a logic program that is running.")
        if not self.holding:
            raise RuntimeError("Cannot resume a controllable solve on a logic program that is not holding a saved grounding.")
        
        self.__logger.log(self.__verbosity, "Resuming controllable incremental solve call with held grounding.")
        self.__running = True
        
        ## Overwrite the current incrementor if requested
        if solve_incrementor is not None:
            self.__incrementor = solve_incrementor
        
        ## Reset the current step bound as requested
        if not maintain_step:
            self.__bounds = Bounds(self.__bounds.increment,
                                   self.__incrementor.step_start - 1,
                                   self.__incrementor.step_start + self.__incrementor.step_increase_initial - 1)
        
        try:
            ## Attempt to yield the signal
            solve_signal: SolveSignal = self.__create_solve_signal()
            yield solve_signal
            
        finally:
            ## Free the held grounding unless holding is enabled
            self.__running = False
            if not solve_signal.holding:
                self.__logger.log(self.__verbosity, "Stopping incremental solve call.")
                self.free()
            else: self.__logger.log(self.__verbosity, "Pausing incremental solve call.")
            solve_signal.free(hold=False)
    
    def __create_solve_signal(self) -> SolveSignal:
        """
        Create a solve signal object that encapsulates this logic program's currently assigned control object and saved grounding.
        The solve signal can then be used to control incremental solving of this logic program.
        
        Private use only, this method should not be called from outside this class.
        """
        if self.__control is None:
            raise RuntimeError("Cannot create solve signal for logic program that does not have an assigned control object.")
        self.__logger.debug("Creating solve signal...")
        solve_signal = SolveSignal(self, self.__control, self.__inc_run)
        self.__logger.debug(f"Solve signal created:\n{solve_signal!s}")
        return solve_signal
    
    def __build(self) -> None:
        """
        Build this logic program and pass it to the given clingo Control object, preparing it for solving.
        
        Private use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `control : clingo.Control` - The clingo control object to pass the built logic program to.
        """
        self.__logger.debug(f"Program building started :: Processing {len(self.__program)} rules")
        start_time: float = time.perf_counter()
        
        ## Get the process for recording memory usage and make a single measurement to initialise it
        self.__process = psutil.Process(os.getpid())
        self.__process.memory_info()
        
        ## Build the program
        program_builder: clingo.ast.ProgramBuilder
        with clingo.ast.ProgramBuilder(self.__control) as program_builder:
            for rule in self.__program:
                program_builder.add(rule)
        
        building_time: float = time.perf_counter() - start_time
        self.__logger.debug(f"Program building completed in {building_time}s")
    
    def __ground(self, clingo_parts: list[tuple[str, list[clingo.Symbol]]] = []) -> float:
        """
        Ground this logic program with its internal control object.
        
        Private use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `clingo_parts : list[tuple[str, list[clingo.Symbol]]]` - A list of clingo form program parts.
        Each part is a two tuple, whose first element is a string defining the program part's name, and second elements is a list of clingo symbols defining its arguments.
        
        Returns
        -------
        `float` - A float defining the 'wall clock' time taken to ground the given program parts in seconds as reported by python's highest resolution performance counter.
        """
        ## Convert the program parts to the form required by clingo
        if not (_clingo_parts := clingo_parts):
            _clingo_parts = self.__program_parts.get_clingo_form()
        
        self.__logger.debug(f"Grounding program parts:\n{_clingo_parts}")
        start_time: float = time.perf_counter()
        
        ## Ground the program parts with this logic program's internal context object
        self.__control.ground(_clingo_parts, self.__context)
        
        grounding_time: float = time.perf_counter() - start_time
        self.__logger.debug(f"Grounding completed in {grounding_time:.6f}s.")
        
        return grounding_time
    
    def __solve(self, solve_stage: Optional[int] = None) -> tuple[SolveResult, float, Memory, dict[str, Any]]:
        """
        Solve this logic program with its internal control object.
        
        Private use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `solve_stage : {int | None} = None` - The solve stage, either an integer defining the incremental call number, or None to indicate a base solve.
        
        Returns
        -------
        `(SolveResult, float, Memory, dict[str, Any])` - A four-tuple, whose;
            - first element is a solve result,
            - second is a float,
            - third is a memoty object encapsulating the total memory used in the solve call,
            - and fourth is a possibly nested dictionary whose keys are strings and values are floats defining clingo statistics.
        """
        self.__logger.debug("Solving program.")
        start_time = time.perf_counter()
        
        ## Run the solve
        solve_handle: SolveHandle
        clingo_result: clingo.SolveResult
        with self.__control.solve(assumptions=self.__assumptions, async_=True,
                                  on_model=functools.partial(self.__on_model, solve_stage=solve_stage),
                                  on_finish=self.__on_finish) as solve_handle:
            ## Run until a result is obtained
            clingo_result = solve_handle.get()
            
            ## Capture the memory usage before exiting
            memory_info = self.__process.memory_info()
        
        ## Gather outputs
        result = SolveResult(str(clingo_result))
        solving_time: float = time.perf_counter() - start_time
        memory = Memory(memory_info.rss / (1024 ** 2), memory_info.vms / (1024 ** 2))
        clingo_stats: dict[str, Any] = self.__control.statistics
        
        self.__logger.debug(f"Solving completed in {solving_time:.6f}s with result {result.name}.")
        
        return result, solving_time, memory, clingo_stats
    
    def __base_run(self) -> None:
        """
        Run a standard one-shot grounding and solving of the currently assigned base program parts using the given clingo Control object.
        
        Private use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `control : clingo.Control` - The clingo control object to use for grounding and solving.
        """
        self.__logger.log(self.__verbosity, f"Running one-shot ground and solve of program parts:\n{self.__program_parts.base_parts}")
        grounding_time = self.__ground()
        solving_time, memory, clingo_stats = self.__solve()[1:]
        self.__base_stats = Statistics(grounding_time, solving_time, memory=memory, clingo_stats=clingo_stats)
    
    def __inc_run(self, increments: Optional[int] = None) -> Iterator[Feedback]:
        """
        Run an incremental grounding and solving of this logic program with the given clingo Control object.
        This is a generator function, which yields enumerated feedback objects for each incremental solve call.
        
        Protected use only, this method should not be called from outside this module.
        
        Parameters
        ----------
        `control : clingo.Control` - The clingo control object to ground and solve.
        
        `increments: Optional[int] = None` - An optional positive non-zero integer defining the maximum number of incremental solve calls that can be made.
        If not given or None, then incremental solving continues until a halting condition is met.
        
        Yields
        ------
        `Iterator[Feedback]` - A generator iterator yielding a feedback object for each incremental solve call.
        """
        if increments is not None:
            if not isinstance(increments, int):
                raise TypeError(f"Increment cound must be an integer. Got; {increments} of type {type(increments)}.")
            if increments < 1:
                raise ValueError(f"Increment count must be a positive non-zero integer. Got; {increments} of type {type(increments)}.")
        
        self.__logger.log(self.__verbosity, "Running incremental ground and solve of program parts:\n" + '\n'.join(map(str, self.__program_parts.inc_parts)))
        
        ## Looping variables
        if self.__bounds is None:
            increment: int = 1
            previous: int = self.__incrementor.step_start - 1
            current: int = self.__incrementor.step_start + self.__incrementor.step_increase_initial - 1
            self.__bounds = Bounds(increment, previous, current)
        start_increment: int = self.__bounds.increment
        program_parts: list[Tuple[str, list[clingo.Symbol]]] = []
        result: Optional[SolveResult] = None
        last_feedback: Optional[Feedback] = None
        
        ## Statistics variables
        increment_grounding_time: float = 0.0
        increment_solving_time: float = 0.0
        total_increment_time: float = 0.0
        cumulative_grounding_time: float = 0.0
        cumulative_solving_time: float = 0.0
        if self.__cumulative_stats:
            cumulative_grounding_time = self.__cumulative_stats.grounding_time
            cumulative_solving_time = self.__cumulative_stats.solving_time
        total_cumulative_time: float = cumulative_grounding_time + cumulative_solving_time
        total_system_virtual_memory: int = psutil.virtual_memory().total / (1024 ** 2)
        clingo_stats: dict[str, Any] = {}
        
        ## Halt reason variable;
        ##      - This should become not None before the method returns
        halt_reason_description: Optional[str] = None
        
        def get_step(value: Optional[int]) -> str:
            "Function for extracting optional incrementor step bounds."
            return f"{value:>6d}" if value is not None else "  None"
        
        if self.__tqdm:
            ## Declare CLI progress bar
            process = psutil.Process(os.getpid())
            progress_bar = tqdm(desc="Steps", unit="inc",
                                postfix={"st/inc" : str(self.incrementor.step_increase),
                                         "Memory (Mb)" : str(int(process.memory_info().rss / (1024 ** 2))),
                                         "CPU (%)" : str(int(process.cpu_percent()))},
                                initial=(self.__bounds.previous_step + 1),
                                total=self.__incrementor.step_end_max if self.__incrementor.step_end_max is not None else None,
                                leave=False, ncols=180, miniters=1, colour="cyan")
        
        ## Increment whilst;
        ##      - the number of increments has not been reached and,
        ##      - the time limits have not been reached and,
        ##      - the minimum step value has not been reached or,
        ##          - the maximum step value has not been reached and,
        ##          - the stop condition has not been reached.
        try:
            while (((increments is None
                     or (start_increment + increments - 1) >= self.__bounds.increment)
                    and (self.__incrementor.increment_limit is None
                         or self.__incrementor.increment_limit >= self.__bounds.increment))
                   and ((self.__incrementor.increment_time_limit is None
                         or total_increment_time < self.__incrementor.increment_time_limit)
                        and (self.__incrementor.cumulative_time_limit is None
                             or total_cumulative_time < self.__incrementor.cumulative_time_limit))
                   and ((self.__incrementor.step_end_min is not None
                         and self.__bounds.current_step <= self.__incrementor.step_end_min)
                        or ((self.__incrementor.step_end_max is None
                             or self.__bounds.current_step <= self.__incrementor.step_end_max)
                            and (self.__incrementor.stop_condition is None
                                 or self.__incrementor.stop_condition != result)))):
                
                ## Log that a new incremental call is about to be made
                self.__logger.debug(f"Beginning incremental call [{self.__bounds.increment}]:\n"
                                    + "\n".join([f"Running step bounds   | Previous = {self.__bounds.previous_step:>6d} : Current = {self.__bounds.current_step:>6d}",
                                                 f"Incrementor step ends | Minimum  = {get_step(self.__incrementor.step_end_min)} : Maximum = {get_step(self.__incrementor.step_end_max)}"]))
                
                ## Obtain the step range for the current increment
                ##      - Add one to bound because the previous step has already been solved so needs to be exclusive and the current has not so needs to be inclusive
                step_range = range(self.__bounds.previous_step + 1, self.__bounds.current_step + 1)
                
                ## Setup the program parts for the current increment;
                ##      - Reset the list,
                ##      - Add satisfiability checking program part,
                ##      - Add incremental parts for each step between the previous and current bounds.
                program_parts = []
                program_parts.append(("check", [clingo.Number(self.__bounds.current_step)]))
                for step in step_range:
                    program_parts.extend(self.__program_parts.get_clingo_form(inc_step=step))
                
                ## Release the query atom from the previous increment if this is not the first increment
                if self.__bounds.increment > 1:
                    self.__control.release_external(clingo.Function("query", [clingo.Number(self.__bounds.previous_step)]))
                
                ## Ground the program parts
                increment_grounding_time = self.__ground(program_parts)
                
                ## Assign the query atom for the current iteration
                self.__control.assign_external(clingo.Function("query", [clingo.Number(self.__bounds.current_step)]), True)
                
                ## Assign the queued externals
                for external in self.__queued_externals:
                    for step in step_range:
                        if external.inc_range is None or external.inc_range.in_range(step):
                            self.__logger.debug(f"Assigning queued external for step {step}: original = {external}, conversion = {external.convert_symbol(step)}.")
                            self.__control.assign_external(external.convert_symbol(step), external.truth)
                self.__queued_externals[:] = [external for external in self.__queued_externals
                                              if (external.inc_range is not None
                                                  and (external.inc_range.end is None
                                                       or external.inc_range.end >= self.__bounds.current_step))]
                
                ## Solve the program and record the result
                result, increment_solving_time, memory, clingo_stats = self.__solve(self.__bounds.increment)
                
                ## Update computation timing variables
                cumulative_grounding_time += increment_grounding_time
                cumulative_solving_time += increment_solving_time
                total_increment_time = increment_grounding_time + increment_solving_time
                total_cumulative_time = cumulative_grounding_time + cumulative_solving_time
                
                ## Record computation statistics
                if self.__cumulative_stats is not None:
                    max_memory = Memory(max(memory.rss, self.__cumulative_stats.memory.rss), max(memory.vms, self.__cumulative_stats.memory.vms))
                else: max_memory = memory
                self.__cumulative_stats = Statistics(cumulative_grounding_time, cumulative_solving_time, memory=max_memory, step_range=range(self.__incrementor.step_start, step_range.stop))
                self.__incremental_stats.append(Statistics(increment_grounding_time, increment_solving_time, memory=memory, step_range=step_range, clingo_stats=clingo_stats))
                
                ## Create the feedback information about the most recent solving call
                last_feedback = Feedback(self.__bounds.increment, self.__bounds.previous_step, self.__bounds.current_step,
                                         result, self.__cumulative_stats, self.__incremental_stats[self.__bounds.increment - 1])
                
                ## Log that the incremental call has been completed
                self.__logger.debug(f"Completed incremental call [{self.__bounds.increment}]:\n"
                                    + "\n".join([f"Result = {result}, Stop condition = {self.__incrementor.stop_condition}",
                                                 f"Increment time = {total_increment_time}, Incremental time limit = {self.__incrementor.increment_time_limit}",
                                                 f"Running cumulative time = {total_cumulative_time}, Cumulative time limit = {self.__incrementor.cumulative_time_limit}",
                                                 f"Used memory = {memory!s}, Total system virtual memory = {total_system_virtual_memory}Mb"]))
                if result == SolveResult.Satisfiable:
                    self.__logger.debug(f"Incremental solving has found an answer set:\n{self.get_answer()}")
                
                ## Update the CLI progress bar
                if self.__tqdm:
                    progress_bar.update(self.__bounds.current_step - self.__bounds.previous_step)
                    progress_bar.set_postfix({"st/inc" : str(self.incrementor.step_increase),
                                              "Memory (Mb)" : str(int(process.memory_info().rss / (1024 ** 2))),
                                              "CPU (%)" : str(process.cpu_percent())})
                
                ## Cleanup the logic program ready for the next iteration
                self.__control.cleanup()
                
                ## Increment the call number and current step
                self.__bounds.increment += 1
                self.__bounds.previous_step = self.__bounds.current_step
                self.__bounds.current_step += self.__incrementor.step_increase
                
                ## Yield the feedback information
                yield last_feedback
            
        except GeneratorExit as exit:
            self.__logger.debug(f"Exiting incremental run loop from caught generator exit.")
            halt_reason_description = "External generator exit"
            raise exit
        
        finally:
            if self.__tqdm: progress_bar.close()
            
            if isinstance(statistics := self.get_answer(dummy=True).statistics, IncrementalStatistics):
                self.__logger.debug(f"Incremental statistics:\n{statistics.incremental_stats_str}")
            
            ## If we don't have a halt reason yet, and at least one increment ran, find the halt reason as usual,
            if halt_reason_description is None and last_feedback is not None:
                halt_reason = HaltReason.get_halt_reason(self.__incrementor, start_increment + increments - 1 if increments is not None else None, last_feedback)
                if halt_reason is not None:
                    halt_reason_description = halt_reason.value.description
            
            ## Else if we have a halt reason but no increments ran, make a note of this,
            elif halt_reason_description is not None and last_feedback is None:
                halt_reason_description = f"{halt_reason_description} (no increments ran)"
            
            ## Else if we still don't have a halt reason, an error must have occured.
            if halt_reason_description is None:
                ## TODO If no increments ran because a limited was reached instantly, such that last_feedback is None, we still get here with no description of why.
                ## The logic program should store a halt reason its and determine it when returning from the incremental solve call to fix this.
                halt_reason_description = "Unknown (an error occured)"
            
            ## self.__halt_reason = halt_reason ## TODO and add to result?
            ## Incremental result also has stop_condition: Optional[StopCondition] which was the requested stop condition
            self.__logger.log(self.__verbosity, f"Incremental ground and solve completed in {total_cumulative_time:.6f}s due to: {halt_reason_description}.")
    
    def __catch_clingo_log(self, code: clingo.MessageCode, message: str) -> None:
        """
        Intercepts error messages from clingo and sends them to this program's logger.
        
        Private use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `code : clingo.MessageCode` - The message's clingo code.
        
        `message : str` - A string specifying the message itself.
        """
        self.__logger.log(logging.WARNING if self.__warnings else logging.DEBUG,
                          f"Clingo warning {code}: {message}")
    
    def __on_model(self, clingo_model: clingo.Model, solve_stage: Optional[int] = None) -> None:
        """
        Callback method to intercept model objects from Clingo during solving.
        
        Private use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `model : clingo.Model` - The intercepted clingo solver model.
        
        `solve_stage : {int | None} = None` - The solve stage, either an integer defining the incremental call number, or None to indicate a base solve.
        """
        model: Optional[Model] = None
        model_count: int = self.__model_count.get(solve_stage, 0)
        
        if not model_count:
            
            model = Model(clingo_model.symbols(atoms=True),
                          cost=clingo_model.cost,
                          optimality_proven=clingo_model.optimality_proven,
                          number=clingo_model.number,
                          thread_id=clingo_model.thread_id,
                          model_type=clingo_model.type)
            
            self.__models.setdefault(solve_stage, []).append(model)
            
            if self.__count_multiple_models:
                self.__model_count[solve_stage] = 1
        
        else: self.__model_count[solve_stage] = model_count + 1
        
        # self.__logger.log(self.__verbosity,
        #                   ("Model found for solve stage [{}]:".format("base" if solve_stage is None else f"incremental {solve_stage}")
        #                    + (f"\n{model}" if model else f" Model count = {model_count + 1}")))
    
    def __on_finish(self, result: clingo.SolveResult) -> None:
        """
        Callback method to intercept solve result objects from the clingo solver.
        
        Private use only, this method should not be called from outside this class.
        
        Parameters
        ----------
        `result : clingo.SolveResult` - The intercepted clingo solve result.
        """
        self.__satisfiable = self.__satisfiable or result.satisfiable
        self.__exhausted = result.exhausted

#############################################################################################################################################
#############################################################################################################################################
##############  ███████  ██████  ██     ██    ██ ███████ ██████       ██████  ██████  ████████ ██  ██████  ███    ██ ███████  ###############
##############  ██      ██    ██ ██     ██    ██ ██      ██   ██     ██    ██ ██   ██    ██    ██ ██    ██ ████   ██ ██       ###############
##############  ███████ ██    ██ ██     ██    ██ █████   ██████      ██    ██ ██████     ██    ██ ██    ██ ██ ██  ██ ███████  ###############
##############       ██ ██    ██ ██      ██  ██  ██      ██   ██     ██    ██ ██         ██    ██ ██    ██ ██  ██ ██      ██  ###############
##############  ███████  ██████  ███████  ████   ███████ ██   ██      ██████  ██         ██    ██  ██████  ██   ████ ███████  ###############
#############################################################################################################################################
#############################################################################################################################################

class Options:
    @staticmethod
    def models(models: int = 0) -> str:
        return f"--models={models}"
    
    @enum.unique
    class ParallelMode(enum.Enum):
        Compete = "compete"
        Split = "split"
    
    @staticmethod
    def threads(threads: int, mode: Optional[ParallelMode] = None) -> str:
        return f"--parallel-mode={threads},{mode.value if mode is not None else Options.ParallelMode.Compete.value}"
    
    @enum.unique
    class OptimiseMode(enum.Enum):
        FindOptimum = "opt"
        Bounded = "enum"
        EmunerateOptimal = "optN"
        Ignore = "ignore"
    
    @staticmethod
    def optimise(mode: OptimiseMode, init_bounds: Optional[Iterable[int]] = None) -> str:
        return f"--opt-mode={mode.value}{(',' + ','.join(str(bound) for bound in init_bounds) if init_bounds else '')}"
    
    @enum.unique
    class EnumerationMode(enum.Enum):
        Backtrack = "bt"
        Record = "record"
        DomainRecord = "domRec"
        Union = "brave"
        Intersection = "cautious"
        Auto = "auto"
    
    @staticmethod
    def enumeration(mode: EnumerationMode, bounds: Optional[Iterable[int]] = None) -> str:
        return f"--enum-mode={mode.value}{(',' + ','.join(str(bound) for bound in bounds) if bounds else '')}"
    
    @staticmethod
    def time_limit(seconds: int) -> str:
        return f"--time-limit={seconds}"
    
    @staticmethod
    def program_heuristics() -> str:
        return f"--heuristic=Domain"
    
    @enum.unique
    class PrintMode(enum.Enum):
        All = 0
        Last = 1
        No = 2
    
    @staticmethod
    def print(models: PrintMode, costs: PrintMode, calls: PrintMode) -> str:
        """
        Configure printing of clingo models and calls to the CLI.
        
        Parameters
        ----------
        `models : PrintMode` - Whether stable models (answer sets) are printed.
        
        `costs : PrintMode` - Whether the optimisation costs of models are printed.
        
        `calls : PrintMode` - Whether details about each individual solver call are printed.
        """
        return f"--quiet={models.value},{costs.value},{calls.value}"
    
    @enum.unique
    class VerbosityLevel(enum.Enum):
        """
        Verbosity levels recognised by clingo.
        
        Items
        -----
        `Minimal` - No additional output.
        
        `Standard` - Shows additional seperators between models.
        
        `All` - Shows solver variables, constraints, state and limits.
        """
        Minimal = 1
        Standard = 2
        All = 3
    
    @staticmethod
    def verbosity(level: VerbosityLevel) -> str:
        return f"--verbose={level.value}"
    
    @staticmethod
    def warn(warn: bool) -> str:
        return f"--warn={'all' if warn else 'none'}"
    
    @staticmethod
    def statistics() -> str:
        return "--stats"
