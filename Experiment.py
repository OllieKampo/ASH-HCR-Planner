###########################################################################
###########################################################################
## Module for running experiments with ASH.                              ##
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

"""Module for running experiments with ASH."""

import logging
import math
import statistics
import time
from collections import defaultdict
from typing import Any, Callable, Iterator, NamedTuple, Optional, Union

import numpy
import pandas
import tqdm

import core.Planner as Planner
from ASP_Parser import Statistics
from core.Helpers import center_text
from core.Strategies import DivisionPoint, DivisionScenario, SubGoalRange

## Experiment module logger
_EXP_logger: logging.Logger = logging.getLogger(__name__)
_EXP_logger.setLevel(logging.DEBUG)

class Quantiles(NamedTuple):
    """Convenience class for representing quantiles."""
    
    min: float = 0.0
    lower: float = 0.0
    med: float = 0.0
    upper: float = 0.0
    max: float = 0.0

def rmse(actual: list[int], perfect: list[float]) -> float:
    """Calculate root mean squared error."""
    if len(actual) != len(perfect): raise ValueError("Actual and perfect point spread lists must have equal length.")
    return (sum(abs(float(obs) - float(pred)) ** 2 for obs, pred in zip(actual, perfect)) / len(actual)) ** (0.5)

def mae(actual: list[int], perfect: list[float]) -> float:
    """Calculate mean absolute error."""
    if len(actual) != len(perfect): raise ValueError("Actual and perfect point spread lists must have equal length.")
    return (sum(abs(float(obs) - float(pred)) for obs, pred in zip(actual, perfect)) / len(actual))

class Results:
    """Encapsulates the results of experimental trails as a collection of hierarchical plans."""
    
    __slots__ = ("__optimums",
                 "__plans",
                 "__dataframes",
                 "__is_changed",
                 "__successful_runs",
                 "__failed_runs")
    
    def __init__(self, optimums: Optional[dict[int, int]]) -> None:
        """Create a new results object containing no experimental data."""
        self.__optimums: Optional[dict[int, int]] = optimums
        self.__plans: list[Planner.HierarchicalPlan] = []
        self.__dataframes: dict[str, pandas.DataFrame] = {}
        self.__is_changed: bool = False
        self.__successful_runs: int = 0
        self.__failed_runs: int = 0
    
    def __getitem__(self, index: int) -> Planner.HierarchicalPlan:
        """Get the plan at the given index."""
        return self.__plans[index]
    
    def __iter__(self) -> Iterator[Planner.HierarchicalPlan]:
        """Iterate over the plans in the results."""
        yield from self.__plans
    
    def __len__(self) -> int:
        """Get the number of plans in the results."""
        return len(self.__plans)
    
    def add(self, plan: Planner.HierarchicalPlan) -> None:
        """Add a new plan to the results."""
        self.__plans.append(plan)
        self.__is_changed = True
    
    def runs_completed(self, successful_runs: int, failed_runs: int) -> None:
        """Set the number of successful and failed runs."""
        self.__successful_runs = successful_runs
        self.__failed_runs = failed_runs
        self.__is_changed = True
    
    @property
    def globals(self) -> pandas.DataFrame:
        """Get the global statistics of the experiment."""
        return self.process()["GLOBALS"].drop("RU", axis="columns")
    
    @property
    def cat_level_wise_plans(self) -> pandas.DataFrame:
        """Get the concatenated plan level wise grouped statistics of the experiment."""
        return self.process()["CAT"].drop("RU", axis="columns").groupby("AL")
    
    @property
    def par_level_wise_plans(self) -> pandas.DataFrame:
        """Get the partial plan level wise grouped statistics of the experiment."""
        return self.process()["PAR"].drop(["RU", "IT"], axis="columns").groupby("AL")
    
    @property
    def par_problem_wise_plans(self) -> pandas.DataFrame:
        """Get the partial plan level wise and problem wise grouped statistics of the experiment."""
        return self.process()["PAR"].drop(["RU", "IT"], axis="columns").groupby(["AL", "PN"])
    
    @property
    def step_wise(self) -> pandas.DataFrame:
        """Get the step wise statistics of the experiment."""
        return self.process()["STEP_CAT"].drop("RU", axis="columns").groupby(["AL", "SL"])
    
    @property
    def index_wise(self) -> pandas.DataFrame:
        """Get the index wise statistics of the experiment."""
        return self.process()["INDEX_CAT"].drop("RU", axis="columns").groupby(["AL", "INDEX"])
    
    @staticmethod
    def set_index(dataframe: pandas.DataFrame, sort_ascending: bool = False) -> pandas.DataFrame:
        """Sort and reset the index of the dataframe."""
        return dataframe.sort_index(axis="index", ascending=sort_ascending).reset_index()
    
    @property
    def globals_means(self) -> pandas.DataFrame:
        """Get the means of the global statistics of the experiment."""
        return Results.set_index(self.globals.mean())
    
    @property
    def globals_stdev(self) -> pandas.DataFrame:
        """Get the standard deviations of the global statistics of the experiment."""
        return Results.set_index(self.globals.std())
    
    @property
    def globals_quantiles(self) -> pandas.DataFrame:
        """Get the quantiles of the global statistics of the experiment."""
        return Results.set_index(self.globals.quantile([0.0, 0.25, 0.5, 0.75, 1.0]))
    
    @property
    def cat_level_wise_means(self) -> pandas.DataFrame:
        """Get the means of the concatenated plan level wise statistics of the experiment."""
        return Results.set_index(self.cat_level_wise_plans.mean())
    
    @property
    def cat_level_wise_stdev(self) -> pandas.DataFrame:
        """Get the standard deviations of the concatenated plan level wise statistics of the experiment."""
        return Results.set_index(self.cat_level_wise_plans.std())
    
    @property
    def cat_level_wise_quantiles(self) -> pandas.DataFrame:
        """Get the quantiles of the concatenated plan level wise statistics of the experiment."""
        return Results.set_index(self.cat_level_wise_plans.quantile([0.0, 0.25, 0.5, 0.75, 1.0]))
    
    @property
    def par_level_wise_means(self) -> pandas.DataFrame:
        """Get the means of the partial plan level wise statistics of the experiment."""
        return Results.set_index(self.par_level_wise_plans.mean())
    
    @property
    def par_level_wise_stdev(self) -> pandas.DataFrame:
        """Get the standard deviations of the partial plan level wise statistics of the experiment."""
        return Results.set_index(self.par_level_wise_plans.std())
    
    @property
    def par_level_wise_quantiles(self) -> pandas.DataFrame:
        """Get the quantiles of the partial plan level wise statistics of the experiment."""
        return Results.set_index(self.par_level_wise_plans.quantile([0.0, 0.25, 0.5, 0.75, 1.0]))
    
    @property
    def par_problem_wise_means(self) -> pandas.DataFrame:
        """Get the means of the partial plan problem wise statistics of the experiment."""
        return Results.set_index(self.par_problem_wise_plans.mean())
    
    @property
    def par_problem_wise_stdev(self) -> pandas.DataFrame:
        """Get the standard deviations of the partial plan problem wise statistics of the experiment."""
        return Results.set_index(self.par_problem_wise_plans.std())
    
    @property
    def par_problem_wise_quantiles(self) -> pandas.DataFrame:
        """Get the quantiles of the partial plan problem wise statistics of the experiment."""
        return Results.set_index(self.par_problem_wise_plans.quantile([0.0, 0.25, 0.5, 0.75, 1.0]))
    
    @property
    def step_wise_means(self) -> pandas.DataFrame:
        """Get the means of the step wise statistics of the experiment."""
        return Results.set_index(self.step_wise.mean(), sort_ascending=True)
    
    @property
    def step_wise_stdev(self) -> pandas.DataFrame:
        """Get the standard deviations of the step wise statistics of the experiment."""
        return Results.set_index(self.step_wise.std(), sort_ascending=True)
    
    @property
    def index_wise_means(self) -> pandas.DataFrame:
        """Get the means of the index wise statistics of the experiment."""
        return Results.set_index(self.index_wise.mean(), sort_ascending=True)
    
    @property
    def index_wise_stdev(self) -> pandas.DataFrame:
        """Get the standard deviations of the index wise statistics of the experiment."""
        return Results.set_index(self.index_wise.std(), sort_ascending=True)
    
    def best_quality(self) -> Planner.HierarchicalPlan:
        """Get the best quality plan found in the experiment."""
        best_plan: Planner.HierarchicalPlan
        best_quality, best_length, best_actions = 0
        for hierarchical_plan in self.__plans:
            bottom_plan: Planner.MonolevelPlan = hierarchical_plan[hierarchical_plan.bottom_level]
            if (plan_quality := bottom_plan.calculate_plan_quality(best_length, best_actions)) > best_quality:
                best_plan = bottom_plan
                best_quality = plan_quality
                best_length = bottom_plan.plan_length
                best_actions = bottom_plan.total_actions
        return best_plan
    
    def process(self) -> dict[str, pandas.DataFrame]:
        """Process the currently collected data and return them as a pandas dataframe."""
        if (self.__dataframes is not None
            and not self.__is_changed):
            return self.__dataframes
        self.__is_changed = False
        
        if not self.__plans:
            raise RuntimeError("Cannot process an empty set of plans.")
        
        ## Collate the data into a dictionary
        data_dict: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        
        ## Constants for calculating time scores
        acceptable_lag_time: float = 5.0
        acceptable_action_minimum_execution_time: float = 1.0
        max_time: float = 1800.0
        
        ## Constant for calculating ground plan quality
        ground_optimum: int = 0
        if (self.__optimums is not None
            and self.__optimums[min(self.__optimums)] is not None):
            ground_optimum = self.__optimums[min(self.__optimums)]
        else: ground_optimum = min(hierarchical_plan[hierarchical_plan.bottom_level].total_actions
                                   for hierarchical_plan in self.__plans)
        
        for run, hierarchical_plan in enumerate(self.__plans):
            data_dict["GLOBALS"]["RU"].append(run)
            
            data_dict["GLOBALS"]["BL_LE"].append(plan_length := hierarchical_plan[hierarchical_plan.bottom_level].plan_length)
            data_dict["GLOBALS"]["BL_AC"].append(total_actions := hierarchical_plan[hierarchical_plan.bottom_level].total_actions)
            
            data_dict["GLOBALS"]["EX_T"].append(hierarchical_plan.execution_latency_time)
            data_dict["GLOBALS"]["HA_T"].append(hierarchical_plan.absolution_time)
            
            data_dict["GLOBALS"]["AW_T"].append(wait_time := hierarchical_plan.get_average_wait_time(hierarchical_plan.bottom_level))
            data_dict["GLOBALS"]["AW_T_PA"].append(wait_pa_time := hierarchical_plan.get_average_wait_time(hierarchical_plan.bottom_level, per_action=True))
            
            data_dict["GLOBALS"]["AME_T"].append(minimum_execution_time := hierarchical_plan.get_average_minimum_execution_time(hierarchical_plan.bottom_level))
            data_dict["GLOBALS"]["AME_T_PA"].append(minimum_execution_pa_time := hierarchical_plan.get_average_minimum_execution_time(hierarchical_plan.bottom_level, per_action=True))
            
            ## Plan quality score relative to optimal
            data_dict["GLOBALS"]["QL_SCORE"].append(quality_score := ground_optimum / total_actions)
            
            ## Time scores have an inverse logarithmic trend which tends to zero in the limit to 1800 seconds
            latency_score: float = 1.0
            if (latency_time := hierarchical_plan.execution_latency_time) > acceptable_lag_time:
                latency_score = (1.0 - (math.log(latency_time - (acceptable_lag_time - 1.0)) / math.log(max_time)))
            
            absolution_score: float = 1.0
            if (absolution_time := hierarchical_plan.absolution_time) > acceptable_lag_time:
                absolution_score = (1.0 - (math.log(absolution_time - (acceptable_lag_time - 1.0)) / math.log(max_time)))
            
            wait_score: float = 1.0
            wait_pa_score: float = 1.0
            if wait_time > acceptable_lag_time:
                wait_score = (1.0 - (math.log(wait_time - (acceptable_lag_time - 1.0)) / math.log(max_time)))
            if wait_pa_time > acceptable_action_minimum_execution_time:
                wait_pa_score = (1.0 - (math.log(wait_pa_time - (acceptable_action_minimum_execution_time - 1.0)) / math.log(max_time)))
            
            minimum_execution_score: float = 1.0
            minimum_execution_pa_score: float = 1.0
            if minimum_execution_time > acceptable_lag_time:
                minimum_execution_score = (1.0 - (math.log(minimum_execution_time - (acceptable_lag_time - 1.0)) / math.log(max_time)))
            if minimum_execution_pa_time > acceptable_action_minimum_execution_time:
                minimum_execution_pa_score = (1.0 - (math.log(minimum_execution_pa_time - (acceptable_action_minimum_execution_time - 1.0)) / math.log(max_time)))
            
            ## Time scores
            data_dict["GLOBALS"]["EX_SCORE"].append(latency_score)
            data_dict["GLOBALS"]["HA_SCORE"].append(absolution_score)
            
            data_dict["GLOBALS"]["AW_SCORE"].append(wait_score)
            data_dict["GLOBALS"]["AW_PA_SCORE"].append(wait_pa_score)
            
            data_dict["GLOBALS"]["AME_SCORE"].append(minimum_execution_score)
            data_dict["GLOBALS"]["AME_PA_SCORE"].append(minimum_execution_pa_score)
            
            time_score: float = 0.0
            if (hierarchical_plan.is_hierarchical_refinement
                and len(hierarchical_plan.partial_plans[hierarchical_plan.bottom_level]) > 1):
                time_score = statistics.mean([latency_score, minimum_execution_score, minimum_execution_pa_score])
            else: time_score = absolution_score
            data_dict["GLOBALS"]["TI_SCORE"].append(time_score)
            
            ## Time grades
            data_dict["GLOBALS"]["EX_GRADE"].append(latency_grade := quality_score * latency_score)
            data_dict["GLOBALS"]["HA_GRADE"].append(absolution_grade := quality_score * absolution_score)
            
            data_dict["GLOBALS"]["AW_GRADE"].append(quality_score * wait_score)
            data_dict["GLOBALS"]["AW_PA_GRADE"].append(quality_score * wait_pa_score)
            
            data_dict["GLOBALS"]["AME_GRADE"].append(minimum_execution_grade := quality_score * minimum_execution_score)
            data_dict["GLOBALS"]["AME_PA_GRADE"].append(minimum_execution_pa_grade := quality_score * minimum_execution_pa_score)
            
            ## Overall grade
            overall_grade: float = 0.0
            if (hierarchical_plan.is_hierarchical_refinement
                and len(hierarchical_plan.partial_plans[hierarchical_plan.bottom_level]) > 1):
                ## For online planning the latency time accounts for time to get the initial ground-level partial plan,
                ## minimum execution time accounts for the wait time to generate all non-initial
                ## ground-level partial plans, relative to the total actions yielded by the plan.
                overall_grade = statistics.mean([latency_grade, minimum_execution_grade, minimum_execution_pa_grade])
            ## For offline planning since the planner does not yield partial plans such that there is no wait time
            ##        (the robot does not have to wait beyond the latency time since it gets the complete plan on yield);
            ##      - The execution latency, absolution, and average wait scores are the same,
            ##      - The average minimum execution time is irrelevant.
            else: overall_grade = absolution_grade
            data_dict["GLOBALS"]["GRADE"].append(overall_grade)
            
            for sequence_number, level, increment, problem_number in hierarchical_plan.get_hierarchical_problem_sequence():
                data_dict["PROBLEM_SEQUENCE"]["RU"].append(run)
                data_dict["PROBLEM_SEQUENCE"]["SN"].append(sequence_number)
                data_dict["PROBLEM_SEQUENCE"]["AL"].append(level)
                data_dict["PROBLEM_SEQUENCE"]["IT"].append(increment)
                data_dict["PROBLEM_SEQUENCE"]["PN"].append(problem_number)
                
                solution: Planner.MonolevelPlan = hierarchical_plan.partial_plans[level][increment]
                data_dict["PROBLEM_SEQUENCE"]["START_S"].append(solution.action_start_step)
                data_dict["PROBLEM_SEQUENCE"]["IS_INITIAL"].append(solution.is_initial)
                data_dict["PROBLEM_SEQUENCE"]["IS_FINAL"].append(solution.is_final)
                
                problem_size: int = 1
                sgoal_literals_total: int = 0
                sgoals_range = SubGoalRange(1, 1)
                # l_blend = 0; r_blend = 0
                if solution.is_refined:
                    problem_size = solution.conformance_mapping.problem_size
                    sgoal_literals_total = solution.conformance_mapping.total_sgoal_literals
                    sgoals_range = solution.conformance_mapping.constraining_sgoals_range
                    # r_blend = hierarchical_plan.get_division_points(level + 1)[sequence_number - 1].blend.left
                    # r_blend = hierarchical_plan.get_division_points(level + 1)[sequence_number].blend.right
                
                data_dict["PROBLEM_SEQUENCE"]["SIZE"].append(problem_size)
                data_dict["PROBLEM_SEQUENCE"]["SGLITS_T"].append(sgoal_literals_total)
                data_dict["PROBLEM_SEQUENCE"]["FIRST_I"].append(sgoals_range.first_index)
                data_dict["PROBLEM_SEQUENCE"]["LAST_I"].append(sgoals_range.last_index)
                # data_dict["PROBLEM_SEQUENCE"]["L_BLEND"].append(l_blend)
                # data_dict["PROBLEM_SEQUENCE"]["R_BLEND"].append(r_blend)
            
            for level in reversed(hierarchical_plan.level_range):
                
                ## Division Points
                for division_number, division_point in enumerate(hierarchical_plan.get_division_points(level + 1)):
                    data_dict["DIVISIONS"]["RU"].append(run)
                    data_dict["DIVISIONS"]["AL"].append(level)
                    data_dict["DIVISIONS"]["DN"].append(division_number)
                    
                    data_dict["DIVISIONS"]["APP_INDEX"].append(division_point.index)
                    data_dict["DIVISIONS"]["COM_INDEX"].append(division_point.committed_index)
                    data_dict["DIVISIONS"]["COM_STEP"].append(division_point.committed_step)
                    data_dict["DIVISIONS"]["L_BLEND"].append(division_point.blend.left)
                    data_dict["DIVISIONS"]["R_BLEND"].append(division_point.blend.right)
                    
                    data_dict["DIVISIONS"]["IS_INHERITED"].append(division_point.inherited)
                    data_dict["DIVISIONS"]["IS_PROACTIVE"].append(division_point.proactive)
                    data_dict["DIVISIONS"]["IS_INTERRUPT"].append(division_point.interrupting)
                    data_dict["DIVISIONS"]["PREEMPTIVE"].append(division_point.preemptive)
                
                concatenated_plan: Planner.MonolevelPlan = hierarchical_plan.concatenated_plans[level]
                concatenated_totals: Planner.ASH_Statistics = concatenated_plan.planning_statistics.grand_totals
                data_dict["CAT"]["RU"].append(run)
                data_dict["CAT"]["AL"].append(level)
                
                ## Raw timing statistics
                data_dict["CAT"]["GT"].append(concatenated_totals.grounding_time)
                data_dict["CAT"]["ST"].append(concatenated_totals.solving_time)
                data_dict["CAT"]["OT"].append(concatenated_totals.overhead_time)
                data_dict["CAT"]["TT"].append(concatenated_totals.total_time)
                
                ## Hierarchical timing statistics
                data_dict["CAT"]["LT"].append(hierarchical_plan.get_latency_time(level))
                data_dict["CAT"]["CT"].append(hierarchical_plan.get_completion_time(level))
                data_dict["CAT"]["WT"].append(wait_time := hierarchical_plan.get_average_wait_time(level))
                data_dict["CAT"]["WT_PA"].append(wait_pa_time := hierarchical_plan.get_average_wait_time(level, per_action=True))
                data_dict["CAT"]["MET"].append(minimum_execution_time := hierarchical_plan.get_average_minimum_execution_time(level))
                data_dict["CAT"]["MET_PA"].append(minimum_execution_pa_time := hierarchical_plan.get_average_minimum_execution_time(level, per_action=True))
                
                ## Required memory usage
                data_dict["CAT"]["RSS"].append(concatenated_totals.memory.rss)
                data_dict["CAT"]["VMS"].append(concatenated_totals.memory.vms)
                
                ## Concatenated plan quality
                data_dict["CAT"]["LE"].append(concatenated_plan.plan_length)
                data_dict["CAT"]["AC"].append(concatenated_plan.total_actions)
                data_dict["CAT"]["CF"].append(concatenated_plan.compression_factor)
                data_dict["CAT"]["PSG"].append(concatenated_plan.total_produced_sgoals)
                
                ## Conformance constraints
                problem_size: int = 1
                sgoal_literals_total: int = 0
                if concatenated_plan.is_refined:
                    problem_size = concatenated_plan.conformance_mapping.problem_size
                    sgoal_literals_total = concatenated_plan.conformance_mapping.total_sgoal_literals
                data_dict["CAT"]["SIZE"].append(problem_size)
                data_dict["CAT"]["SGLITS_T"].append(sgoal_literals_total)
                
                optimum: int = 0
                if (self.__optimums is not None
                    and level in self.__optimums
                    and self.__optimums[level] is not None):
                    optimum = self.__optimums[level]
                else: optimum = min(h_plan[level].total_actions
                                    for h_plan in self.__plans)
                
                data_dict["CAT"]["QL_SCORE"].append(quality_score := optimum / concatenated_plan.total_actions)
                
                latency_score: float = 1.0
                if (latency_time := hierarchical_plan.get_latency_time(level)) > acceptable_lag_time:
                    latency_score = (1.0 - (math.log(latency_time - (acceptable_lag_time - 1.0)) / math.log(max_time)))
                
                completion_score: float = 1.0
                if (completion_time := hierarchical_plan.get_completion_time(level)) > acceptable_lag_time:
                    completion_score = (1.0 - (math.log(completion_time - (acceptable_lag_time - 1.0)) / math.log(max_time)))
                
                wait_score: float = 1.0
                wait_pa_score: float = 1.0
                if wait_time > acceptable_lag_time:
                    wait_score = (1.0 - (math.log(wait_time - (acceptable_lag_time - 1.0)) / math.log(max_time)))
                if wait_pa_time > acceptable_action_minimum_execution_time:
                    wait_pa_score = (1.0 - (math.log(wait_pa_time - (acceptable_action_minimum_execution_time - 1.0)) / math.log(max_time)))
                
                minimum_execution_score: float = 1.0
                minimum_execution_pa_score: float = 1.0
                if minimum_execution_time > acceptable_lag_time:
                    minimum_execution_score = (1.0 - (math.log(minimum_execution_time - (acceptable_lag_time - 1.0)) / math.log(max_time)))
                if minimum_execution_pa_time > acceptable_action_minimum_execution_time:
                    minimum_execution_pa_score = (1.0 - (math.log(minimum_execution_pa_time - (acceptable_action_minimum_execution_time - 1.0)) / math.log(max_time)))
                
                data_dict["CAT"]["LT_SCORE"].append(latency_score)
                data_dict["CAT"]["CT_SCORE"].append(completion_score)
                
                data_dict["CAT"]["AW_SCORE"].append(wait_score)
                data_dict["CAT"]["AW_PA_SCORE"].append(wait_pa_score)
                
                data_dict["CAT"]["AME_SCORE"].append(minimum_execution_score)
                data_dict["CAT"]["AME_PA_SCORE"].append(minimum_execution_pa_score)
                
                time_score: float = 0.0
                if (hierarchical_plan.is_hierarchical_refinement
                    and len(hierarchical_plan.partial_plans[hierarchical_plan.bottom_level]) > 1):
                    time_score = statistics.mean([latency_score, minimum_execution_score, minimum_execution_pa_score])
                else: time_score = completion_score
                data_dict["CAT"]["TI_SCORE"].append(time_score)
                
                data_dict["CAT"]["LT_GRADE"].append(latency_grade := quality_score * latency_score)
                data_dict["CAT"]["CT_GRADE"].append(completion_grade := quality_score * completion_score)
                
                ## TODO Add average wait time non-initial? This would be less affected by the execution latency time?
                data_dict["CAT"]["AW_GRADE"].append(quality_score * wait_score)
                data_dict["CAT"]["AW_PA_GRADE"].append(quality_score * wait_pa_score)
                
                data_dict["CAT"]["AME_GRADE"].append(minimum_execution_grade := quality_score * minimum_execution_score)
                data_dict["CAT"]["AME_PA_GRADE"].append(minimum_execution_pa_grade := quality_score * minimum_execution_pa_score)
                
                ## Overall grade
                if (concatenated_plan.is_refined
                    and len(hierarchical_plan.partial_plans[level]) > 1):
                    overall_grade = statistics.mean([latency_grade, minimum_execution_grade, minimum_execution_pa_grade])
                else: overall_grade = completion_grade
                data_dict["CAT"]["GRADE"].append(overall_grade)
                
                ## Trailing plans
                data_dict["CAT"]["HAS_TRAILING"].append(concatenated_plan.has_trailing_plan)
                
                ## Final-goal preemptive achievement
                data_dict["CAT"]["TOT_CHOICES"].append(concatenated_plan.total_choices)
                data_dict["CAT"]["PRE_CHOICES"].append(concatenated_plan.preemptive_choices)
                
                ## Final-goal intermediate ordering preferences
                data_dict["CAT"]["FGOALS_ORDER"].append(bool(concatenated_plan.fgoal_ordering_correct))
                
                ## Sub-plan Expansion
                factor: Planner.Expansion = concatenated_plan.get_plan_expansion_factor()
                deviation: Planner.Expansion = concatenated_plan.get_expansion_deviation()
                balance: Planner.Expansion = concatenated_plan.get_degree_of_balance()
                data_dict["CAT"]["CP_EF_L"].append(factor.length)
                data_dict["CAT"]["CP_EF_A"].append(factor.action)
                data_dict["CAT"]["SP_ED_L"].append(deviation.length)
                data_dict["CAT"]["SP_ED_A"].append(deviation.action)
                data_dict["CAT"]["SP_EB_L"].append(balance.length)
                data_dict["CAT"]["SP_EB_A"].append(balance.action)
                
                length_balance_score: float = 1.0
                action_balance_score: float = 1.0
                if deviation.length > 0.0:
                    length_balance_score = (1.0 - (math.log(deviation.length + 1.0) / math.log(problem_size)))
                if deviation.action > 0.0:
                    action_balance_score = (1.0 - (math.log(deviation.action + 1.0) / math.log(problem_size)))
                data_dict["CAT"]["SP_EBS_L"].append(length_balance_score)
                data_dict["CAT"]["SP_EBS_A"].append(action_balance_score)
                
                sub_plan_expansion: list[Planner.Expansion] = []
                length_expansion = Quantiles()
                action_expansion = Quantiles()
                if concatenated_plan.is_refined:
                    for index in concatenated_plan.conformance_mapping.constraining_sgoals:
                        sub_plan_expansion.append(concatenated_plan.get_expansion_factor(index))
                    length_expansion = Quantiles(*numpy.quantile([sp.length for sp in sub_plan_expansion], [0.0, 0.25, 0.5, 0.75, 1.0]))
                    action_expansion = Quantiles(*numpy.quantile([sp.action for sp in sub_plan_expansion], [0.0, 0.25, 0.5, 0.75, 1.0]))
                
                data_dict["CAT"]["SP_MIN_L"].append(length_expansion.min)
                data_dict["CAT"]["SP_MIN_A"].append(action_expansion.min)
                data_dict["CAT"]["SP_LOWER_L"].append(length_expansion.lower)
                data_dict["CAT"]["SP_LOWER_A"].append(action_expansion.lower)
                data_dict["CAT"]["SP_MED_L"].append(length_expansion.med)
                data_dict["CAT"]["SP_MED_A"].append(action_expansion.med)
                data_dict["CAT"]["SP_UPPER_L"].append(length_expansion.upper)
                data_dict["CAT"]["SP_UPPER_A"].append(action_expansion.upper)
                data_dict["CAT"]["SP_MAX_L"].append(length_expansion.max)
                data_dict["CAT"]["SP_MAX_A"].append(action_expansion.max)
                
                ## Interleaving
                interleaving: tuple[tuple[int, float], tuple[int, float]] = ((0, 0.0), (0, 0.0))
                if concatenated_plan.is_refined:
                    interleaving = concatenated_plan.interleaving
                data_dict["CAT"]["T_INTER_SP"].append(interleaving[0][0])
                data_dict["CAT"]["P_INTER_SP"].append(interleaving[0][1])
                data_dict["CAT"]["T_INTER_Q"].append(interleaving[1][0])
                data_dict["CAT"]["P_INTER_Q"].append(interleaving[1][1])
                
                ## Sub-plan (refinement tree) balancing, partial plan balancing, and division spread
                rmse_mchild = nrmse_mchild = mae_mchild = nmae_mchild = 0.0
                rmse_div_indices = nrmse_div_indices = mae_div_indices = nmae_div_indices = 0.0
                rmse_div_steps = nrmse_div_steps = mae_div_steps = nmae_div_steps = 0.0
                
                if concatenated_plan.is_refined:
                    perfect_mchild_spacing: float = concatenated_plan.plan_length / problem_size
                    perfect_mchild_spread: list[float] = [perfect_mchild_spacing * index for index in concatenated_plan.conformance_mapping.constraining_sgoals_range]
                    mchilds: list[int] = list(concatenated_plan.conformance_mapping.sgoals_achieved_at.values())
                    rmse_mchild = rmse(mchilds, perfect_mchild_spread)
                    nrmse_mchild = rmse_mchild / perfect_mchild_spacing
                    mae_mchild = mae(mchilds, perfect_mchild_spread)
                    nmae_mchild = mae_mchild / perfect_mchild_spacing
                    
                    total_divisions: int = len(hierarchical_plan.get_division_points(level + 1))
                    total_problems: int = (total_divisions - 2) + 1
                    
                    if total_problems > 1:
                        perfect_div_index_spacing: float = concatenated_plan.conformance_mapping.problem_size / total_problems
                        perfect_div_index_spread: list[float] = [perfect_div_index_spacing * index for index in range(0, total_divisions)]
                        div_indices: list[int] = [point.index for point in hierarchical_plan.get_division_points(level + 1)]
                        rmse_div_indices = rmse(div_indices, perfect_div_index_spread)
                        nrmse_div_indices = rmse_div_indices / perfect_div_index_spacing
                        mae_div_indices = mae(div_indices, perfect_div_index_spread)
                        nmae_div_indices = mae_div_indices / perfect_div_index_spacing
                        
                        perfect_div_step_spacing: float = concatenated_plan.plan_length / total_problems
                        perfect_div_step_spread: list[float] = [perfect_div_step_spacing * index for index in range(0, total_divisions)]
                        div_steps: list[int] = [concatenated_plan.conformance_mapping.sgoals_achieved_at.get(point.index, 0) for point in hierarchical_plan.get_division_points(level + 1)]
                        rmse_div_steps = rmse(div_steps, perfect_div_step_spread)
                        nrmse_div_steps = rmse_div_steps / perfect_div_step_spacing
                        mae_div_steps = mae(div_steps, perfect_div_step_spread)
                        nmae_div_steps = mae_div_steps / perfect_div_step_spacing
                    
                    _EXP_logger.debug(f"Refinement spread at {run=}, {level=}:\n"
                                      f"Root Mean Squared Errors: {rmse_mchild=}, {rmse_div_indices=}, {rmse_div_steps=}\n"
                                      f"Mean Absolute Errors: {mae_mchild=}, {mae_div_indices=}, {mae_div_steps=}")
                
                ## The spread is the root mean squared error between;
                ##      - The final achieved matching child steps (representing the observed data),
                ##      - The theoretical perfectly balanced spread of matching child steps (representing the predicted data).
                ## The facet is that the perfect spacing is usually not achievable since the spacing will usually lie between steps since the plan length is usually not perfect
                data_dict["CAT"]["M_CHILD_RMSE"].append(rmse_mchild)
                data_dict["CAT"]["M_CHILD_RMSE_SCORE"].append(math.exp(-rmse_mchild))
                data_dict["CAT"]["M_CHILD_NRMSE"].append(nrmse_mchild)
                data_dict["CAT"]["M_CHILD_NRMSE_SCORE"].append(math.exp(-nrmse_mchild))
                data_dict["CAT"]["M_CHILD_MAE"].append(mae_mchild)
                data_dict["CAT"]["M_CHILD_MAE_SCORE"].append(math.exp(-mae_mchild))
                data_dict["CAT"]["M_CHILD_NMAE"].append(nmae_mchild)
                data_dict["CAT"]["M_CHILD_NMAE_SCORE"].append(math.exp(-nmae_mchild))
                
                data_dict["CAT"]["DIV_INDEX_RMSE"].append(rmse_div_indices)
                data_dict["CAT"]["DIV_INDEX_RMSE_SCORE"].append(math.exp(-rmse_div_indices))
                data_dict["CAT"]["DIV_INDEX_NRMSE"].append(nrmse_div_indices)
                data_dict["CAT"]["DIV_INDEX_NRMSE_SCORE"].append(math.exp(-nrmse_div_indices))
                data_dict["CAT"]["DIV_INDEX_MAE"].append(mae_div_indices)
                data_dict["CAT"]["DIV_INDEX_MAE_SCORE"].append(math.exp(-mae_div_indices))
                data_dict["CAT"]["DIV_INDEX_NMAE"].append(nmae_div_indices)
                data_dict["CAT"]["DIV_INDEX_NMAE_SCORE"].append(math.exp(-nmae_div_indices))
                
                data_dict["CAT"]["DIV_STEP_RMSE"].append(rmse_div_steps)
                data_dict["CAT"]["DIV_STEP_RMSE_SCORE"].append(math.exp(-rmse_div_steps))
                data_dict["CAT"]["DIV_STEP_NRMSE"].append(nrmse_div_steps)
                data_dict["CAT"]["DIV_STEP_NRMSE_SCORE"].append(math.exp(-nrmse_div_steps))
                data_dict["CAT"]["DIV_STEP_MAE"].append(mae_div_steps)
                data_dict["CAT"]["DIV_STEP_MAE_SCORE"].append(math.exp(-mae_div_steps))
                data_dict["CAT"]["DIV_STEP_NMAE"].append(nmae_div_steps)
                data_dict["CAT"]["DIV_STEP_NMAE_SCORE"].append(math.exp(-nmae_div_steps))
                
                ## Division Scenarios
                division_tree_level: list[DivisionScenario] = hierarchical_plan.problem_division_tree.get(level, [])
                total_scenarios: int = len(division_tree_level)
                data_dict["CAT"]["DS_T"].append(total_scenarios)
                
                divisions_per_scenario: list[int] = [scenario.get_total_divisions(False) for scenario in division_tree_level]
                mean_divisions: float = 0.0
                stdev_divisions: float = 0.0
                bal_divisions: float = 0.0
                quantiles_divisions = Quantiles()
                
                sizes_per_scenario: list[int] = [scenario.size for scenario in division_tree_level]
                mean_size: float = 0.0
                stdev_size: float = 0.0
                bal_size: float = 0.0
                quantiles_sizes = Quantiles()
                
                total_divisions: int = sum(divisions_per_scenario)
                data_dict["CAT"]["DIVS_T"].append(total_divisions)
                
                if total_scenarios != 0:
                    mean_divisions = statistics.mean(divisions_per_scenario)
                    if len(divisions_per_scenario) >= 2:
                        stdev_divisions = statistics.stdev(divisions_per_scenario)
                    else: stdev_divisions = 0.0
                    if mean_divisions != 0.0:
                        bal_divisions = stdev_divisions / mean_divisions
                    quantiles_divisions = Quantiles(*numpy.quantile(divisions_per_scenario, [0.0, 0.25, 0.5, 0.75, 1.0]))
                    
                    mean_size = statistics.mean(sizes_per_scenario)
                    if len(sizes_per_scenario) >= 2:
                        stdev_size = statistics.stdev(sizes_per_scenario)
                    else: stdev_size = 0.0
                    bal_size = stdev_size / mean_size
                    quantiles_sizes = Quantiles(*numpy.quantile(sizes_per_scenario, [0.0, 0.25, 0.5, 0.75, 1.0]))
                
                ## Scenario divisions
                data_dict["CAT"]["DS_TD_MEAN"].append(mean_divisions)
                data_dict["CAT"]["DS_TD_STD"].append(stdev_divisions)
                data_dict["CAT"]["DS_TD_CD"].append(bal_divisions)
                data_dict["CAT"]["DS_TD_MIN"].append(quantiles_divisions.min)
                data_dict["CAT"]["DS_TD_LOWER"].append(quantiles_divisions.lower)
                data_dict["CAT"]["DS_TD_MED"].append(quantiles_divisions.med)
                data_dict["CAT"]["DS_TD_UPPER"].append(quantiles_divisions.upper)
                data_dict["CAT"]["DS_TD_MAX"].append(quantiles_divisions.max)
                
                ## Scenario sizes
                data_dict["CAT"]["DS_TS_MEAN"].append(mean_size)
                data_dict["CAT"]["DS_TS_STD"].append(stdev_size)
                data_dict["CAT"]["DS_TS_CD"].append(bal_size)
                data_dict["CAT"]["DS_TS_MIN"].append(quantiles_sizes.min)
                data_dict["CAT"]["DS_TS_LOWER"].append(quantiles_sizes.lower)
                data_dict["CAT"]["DS_TS_MED"].append(quantiles_sizes.med)
                data_dict["CAT"]["DS_TS_UPPER"].append(quantiles_sizes.upper)
                data_dict["CAT"]["DS_TS_MAX"].append(quantiles_sizes.max)
                
                ## Partial Problems Size Balancing
                partial_plans: dict[int, Planner.MonolevelPlan] = hierarchical_plan.partial_plans.get(level, {})
                total_problems: int = len(partial_plans)
                
                ## Classical problems have size 1 (since they only include the final-goal),
                ## for refinement problems the final-goal takes the same index as the final-sub-goal stage (the stage produced from the final-goal achieving abstract action),
                ## this also relates to the representation of the last refinement tree abopting trailing plans.
                sizes_per_problem: list[int] = []
                if concatenated_plan.is_refined:
                    sizes_per_problem = [partial_plan.conformance_mapping.problem_size for partial_plan in partial_plans.values()]
                mean_problem_size: float = 1.0
                stdev_problem_size: float = 0.0
                bal_problem_size: float = 0.0
                quantiles_problem_size = Quantiles()
                
                if concatenated_plan.is_refined:
                    mean_problem_size = statistics.mean(sizes_per_problem)
                    if len(sizes_per_problem) >= 2:
                        stdev_problem_size = statistics.stdev(sizes_per_problem)
                    else: stdev_problem_size = 0.0
                    bal_problem_size = stdev_problem_size / mean_problem_size
                    quantiles_problem_size = Quantiles(*numpy.quantile(sizes_per_problem, [0.0, 0.25, 0.5, 0.75, 1.0]))
                
                data_dict["CAT"]["PR_T"].append(total_problems)
                data_dict["CAT"]["PR_TS_MEAN"].append(mean_problem_size)
                data_dict["CAT"]["PR_TS_STD"].append(stdev_problem_size)
                data_dict["CAT"]["PR_TS_CD"].append(bal_problem_size)
                data_dict["CAT"]["PR_TS_MIN"].append(quantiles_problem_size.min)
                data_dict["CAT"]["PR_TS_LOWER"].append(quantiles_problem_size.lower)
                data_dict["CAT"]["PR_TS_MED"].append(quantiles_problem_size.med)
                data_dict["CAT"]["PR_TS_UPPER"].append(quantiles_problem_size.upper)
                data_dict["CAT"]["PR_TS_MAX"].append(quantiles_problem_size.max)
                
                ## Partial Plan Length Balancing
                length_per_plan: list[int] = []
                actions_per_plan: list[int] = []
                partial_plan_expansion: list[Planner.Expansion] = []
                
                mean_plan_length: float = 1.0
                mean_total_actions: float = 1.0
                stdev_plan_length: float = 0.0
                stdev_total_actions: float = 0.0
                bal_plan_length: float = 0.0
                bal_total_actions: float = 0.0
                quantiles_plan_length = Quantiles()
                quantiles_total_actions = Quantiles()
                
                partial_plan_length_expansion_deviation: float = 0.0
                partial_plan_action_expansion_deviation: float = 0.0
                partial_plan_length_expansion_balance: float = 0.0
                partial_plan_action_expansion_balance: float = 0.0
                partial_plan_length_expansion_balance_score: float = 0.0
                partial_plan_action_expansion_balance_score: float = 0.0
                quantiles_plan_length_expansion = Quantiles()
                quantiles_total_actions_expansion = Quantiles()
                
                if concatenated_plan.is_refined:
                    for partial_plan in partial_plans.values():
                        length_per_plan.append(partial_plan.plan_length) 
                        actions_per_plan.append(partial_plan.total_actions)
                        partial_plan_expansion.append(partial_plan.get_plan_expansion_factor())
                    
                    mean_plan_length = statistics.mean(length_per_plan)
                    mean_total_actions = statistics.mean(actions_per_plan)
                    if len(length_per_plan) >= 2:
                        stdev_plan_length = statistics.stdev(length_per_plan)
                        stdev_total_actions = statistics.stdev(actions_per_plan)
                    else:
                        stdev_plan_length = 0.0
                        stdev_total_actions = 0.0
                    bal_plan_length = stdev_plan_length / mean_plan_length
                    bal_total_actions = stdev_total_actions / mean_total_actions
                    quantiles_plan_length = Quantiles(*numpy.quantile(length_per_plan, [0.0, 0.25, 0.5, 0.75, 1.0]))
                    quantiles_total_actions = Quantiles(*numpy.quantile(actions_per_plan, [0.0, 0.25, 0.5, 0.75, 1.0]))
                    
                    if total_problems > 1:
                        ## The mean partial plan expansion factor/deviation/balance is identical to the concatenated plan expansion factor/deviation/balance
                        partial_plan_length_expansion_deviation = statistics.stdev([pp.length for pp in partial_plan_expansion])
                        partial_plan_action_expansion_deviation = statistics.stdev([pp.action for pp in partial_plan_expansion])
                        partial_plan_length_expansion_balance = partial_plan_length_expansion_deviation / concatenated_plan.get_plan_expansion_factor().length
                        partial_plan_action_expansion_balance = partial_plan_action_expansion_deviation / concatenated_plan.get_plan_expansion_factor().action
                        if partial_plan_length_expansion_deviation > 0.0:
                            partial_plan_length_expansion_balance_score = (1.0 - (math.log(partial_plan_length_expansion_deviation + 1.0) / math.log(problem_size)))
                        if partial_plan_action_expansion_deviation > 0.0:
                            partial_plan_action_expansion_balance_score = (1.0 - (math.log(partial_plan_action_expansion_deviation + 1.0) / math.log(problem_size)))
                    quantiles_plan_length_expansion = Quantiles(*numpy.quantile([pp.length for pp in partial_plan_expansion], [0.0, 0.25, 0.5, 0.75, 1.0]))
                    quantiles_total_actions_expansion = Quantiles(*numpy.quantile([pp.action for pp in partial_plan_expansion], [0.0, 0.25, 0.5, 0.75, 1.0]))
                
                data_dict["CAT"]["PP_LE_MEAN"].append(mean_plan_length)
                data_dict["CAT"]["PP_AC_MEAN"].append(mean_total_actions)
                data_dict["CAT"]["PP_LE_STD"].append(stdev_plan_length)
                data_dict["CAT"]["PP_AC_STD"].append(stdev_total_actions)
                data_dict["CAT"]["PP_LE_CD"].append(bal_plan_length)
                data_dict["CAT"]["PP_AC_CD"].append(bal_total_actions)
                data_dict["CAT"]["PP_LE_MIN"].append(quantiles_plan_length.min)
                data_dict["CAT"]["PP_AC_MIN"].append(quantiles_total_actions.min)
                data_dict["CAT"]["PP_LE_LOWER"].append(quantiles_plan_length.lower)
                data_dict["CAT"]["PP_AC_LOWER"].append(quantiles_total_actions.lower)
                data_dict["CAT"]["PP_LE_MED"].append(quantiles_plan_length.med)
                data_dict["CAT"]["PP_AC_MED"].append(quantiles_total_actions.med)
                data_dict["CAT"]["PP_LE_UPPER"].append(quantiles_plan_length.upper)
                data_dict["CAT"]["PP_AC_UPPER"].append(quantiles_total_actions.upper)
                data_dict["CAT"]["PP_LE_MAX"].append(quantiles_plan_length.max)
                data_dict["CAT"]["PP_AC_MAX"].append(quantiles_total_actions.max)
                
                data_dict["CAT"]["PP_ED_L"].append(partial_plan_length_expansion_deviation)
                data_dict["CAT"]["PP_ED_A"].append(partial_plan_action_expansion_deviation)
                data_dict["CAT"]["PP_EB_L"].append(partial_plan_length_expansion_balance)
                data_dict["CAT"]["PP_EB_A"].append(partial_plan_action_expansion_balance)
                data_dict["CAT"]["PP_EBS_L"].append(partial_plan_length_expansion_balance_score)
                data_dict["CAT"]["PP_EBS_A"].append(partial_plan_action_expansion_balance_score)
                
                data_dict["CAT"]["PP_EF_LE_MIN"].append(quantiles_plan_length_expansion.min)
                data_dict["CAT"]["PP_EF_AC_MIN"].append(quantiles_total_actions_expansion.min)
                data_dict["CAT"]["PP_EF_LE_LOWER"].append(quantiles_plan_length_expansion.lower)
                data_dict["CAT"]["PP_EF_AC_LOWER"].append(quantiles_total_actions_expansion.lower)
                data_dict["CAT"]["PP_EF_LE_MED"].append(quantiles_plan_length_expansion.med)
                data_dict["CAT"]["PP_EF_AC_MED"].append(quantiles_total_actions_expansion.med)
                data_dict["CAT"]["PP_EF_LE_UPPER"].append(quantiles_plan_length_expansion.upper)
                data_dict["CAT"]["PP_EF_AC_UPPER"].append(quantiles_total_actions_expansion.upper)
                data_dict["CAT"]["PP_EF_LE_MAX"].append(quantiles_plan_length_expansion.max)
                data_dict["CAT"]["PP_EF_AC_MAX"].append(quantiles_total_actions_expansion.max)
                
                ## Step-wise
                grounding_time_sum: float = 0.0
                solving_time_sum: float = 0.0
                total_time_sum: float = 0.0
                rss_max: float = 0.0
                vms_max: float = 0.0
                
                for step in concatenated_plan:
                    current_stat: Statistics = Statistics(0.0, 0.0)
                    for stat in concatenated_plan.planning_statistics.incremental.values():
                        if max(stat.step_range) == step:
                            current_stat = stat
                    
                    data_dict["STEP_CAT"]["RU"].append(run)
                    data_dict["STEP_CAT"]["AL"].append(level)
                    data_dict["STEP_CAT"]["SL"].append(step)
                    
                    ## Incremental and accumlating planning times
                    data_dict["STEP_CAT"]["S_GT"].append(current_stat.grounding_time)
                    data_dict["STEP_CAT"]["S_ST"].append(current_stat.solving_time)
                    data_dict["STEP_CAT"]["S_TT"].append(current_stat.total_time)
                    data_dict["STEP_CAT"]["C_GT"].append(grounding_time_sum := grounding_time_sum + current_stat.grounding_time)
                    data_dict["STEP_CAT"]["C_ST"].append(solving_time_sum := solving_time_sum + current_stat.solving_time)
                    data_dict["STEP_CAT"]["C_TT"].append(total_time_sum := total_time_sum + current_stat.total_time)
                    
                    ## Incremental and maximal memory
                    data_dict["STEP_CAT"]["T_RSS"].append(current_stat.memory.rss)
                    data_dict["STEP_CAT"]["T_VMS"].append(current_stat.memory.vms)
                    data_dict["STEP_CAT"]["M_RSS"].append(rss_max := max(rss_max, current_stat.memory.rss))
                    data_dict["STEP_CAT"]["M_VMS"].append(vms_max := max(vms_max, current_stat.memory.vms))
                    
                    ## Conformance mapping
                    current_sgoals_index: int = 1
                    is_matching_child: bool = False
                    is_trailing_plan: bool = False
                    if concatenated_plan.is_refined:
                        current_sgoals_index = concatenated_plan.conformance_mapping.current_sgoals.get(step, -1)
                        is_matching_child = step in concatenated_plan.conformance_mapping.sgoals_achieved_at.values()
                        is_trailing_plan = current_sgoals_index == -1
                    if is_trailing_plan:
                        current_sgoals_index = concatenated_plan.conformance_mapping.constraining_sgoals_range.last_index
                    data_dict["STEP_CAT"]["C_TACHSGOALS"].append(current_sgoals_index if is_matching_child else current_sgoals_index - 1)
                    data_dict["STEP_CAT"]["S_SGOALI"].append(current_sgoals_index)
                    data_dict["STEP_CAT"]["IS_MATCHING"].append(is_matching_child)
                    data_dict["STEP_CAT"]["IS_TRAILING"].append(is_trailing_plan)
                    
                    ## Accumulating expansion factor
                    index_range = range(1, current_sgoals_index + 1)
                    step_factor: Planner.Expansion = concatenated_plan.get_expansion_factor(index_range, accu_step=step)
                    step_deviation: Planner.Expansion = concatenated_plan.get_expansion_deviation(index_range, accu_step=step)
                    step_balance: Planner.Expansion = concatenated_plan.get_degree_of_balance(index_range, accu_step=step)
                    data_dict["STEP_CAT"]["C_CP_EF_L"].append(step_factor.length)
                    data_dict["STEP_CAT"]["C_CP_EF_A"].append(step_factor.action)
                    data_dict["STEP_CAT"]["C_SP_ED_L"].append(step_deviation.length)
                    data_dict["STEP_CAT"]["C_SP_ED_A"].append(step_deviation.action)
                    data_dict["STEP_CAT"]["C_SP_EB_L"].append(step_balance.length)
                    data_dict["STEP_CAT"]["C_SP_EB_A"].append(step_balance.action)
                    
                    step_length_balance_score: float = 1.0
                    step_action_balance_score: float = 1.0
                    if step_deviation.length > 0.0:
                        step_length_balance_score = max(0.0, 1.0 - (math.log(step_deviation.length + 1.0) / math.log(current_sgoals_index)))
                    if step_deviation.action > 0.0:
                        step_action_balance_score = max(0.0, 1.0 - (math.log(step_deviation.action + 1.0) / math.log(current_sgoals_index)))
                    data_dict["STEP_CAT"]["C_SP_EBS_L"].append(step_length_balance_score)
                    data_dict["STEP_CAT"]["C_SP_EBS_A"].append(step_action_balance_score)
                    
                    ## Problem divisions
                    division_points: list[DivisionPoint] = []
                    if concatenated_plan.is_refined:
                        division_points = hierarchical_plan.get_division_points(level + 1)
                    reached_point: DivisionPoint = None
                    committed_point: DivisionPoint = None
                    for point in division_points:
                        if is_matching_child and point.index == current_sgoals_index:
                            reached_point = point
                        if step == point.committed_step:
                            committed_point = point
                    data_dict["STEP_CAT"]["IS_DIV_APP"].append(reached_point is not None)
                    data_dict["STEP_CAT"]["IS_INHERITED"].append(reached_point is not None and reached_point.inherited)
                    data_dict["STEP_CAT"]["IS_PROACTIVE"].append(reached_point is not None and reached_point.proactive)
                    data_dict["STEP_CAT"]["IS_INTERRUPT"].append(reached_point is not None and reached_point.interrupting)
                    data_dict["STEP_CAT"]["PREEMPTIVE"].append(reached_point is not None and reached_point.preemptive != 0)
                    data_dict["STEP_CAT"]["IS_DIV_COM"].append(committed_point is not None)
                    data_dict["STEP_CAT"]["DIV_COM_APP_AT"].append(committed_point.index if committed_point is not None else -1)
                    
                    ## Sub-plan majority action type
                    sub_plan_type: Planner.ActionType = concatenated_plan.get_action_type(step)
                    data_dict["STEP_CAT"]["IS_LOCO"].append(sub_plan_type == Planner.ActionType.Locomotion)
                    data_dict["STEP_CAT"]["IS_MANI"].append(sub_plan_type == Planner.ActionType.Manipulation)
                    data_dict["STEP_CAT"]["IS_CONF"].append(sub_plan_type == Planner.ActionType.Configuration)
                
                ## Index-wise
                if concatenated_plan.is_refined:
                    conformance_mapping: Planner.ConformanceMapping = concatenated_plan.conformance_mapping
                    constraining_sgoals: dict[int, list[Planner.SubGoal]] = conformance_mapping.constraining_sgoals
                    
                    for index in constraining_sgoals:
                        data_dict["INDEX_CAT"]["RU"].append(run)
                        data_dict["INDEX_CAT"]["AL"].append(level)
                        data_dict["INDEX_CAT"]["INDEX"].append(index)
                        
                        ## Number of sub-goal literals in the stage
                        data_dict["INDEX_CAT"]["NUM_SGOALS"].append(len(constraining_sgoals[index]))
                        
                        ## Final and sequential yield achievement step of the stage
                        data_dict["INDEX_CAT"]["ACH_AT"].append(conformance_mapping.sgoals_achieved_at[index])
                        yield_step: int = -1
                        if (yield_steps := conformance_mapping.sequential_yield_steps) is not None:
                            yield_step = yield_steps[index]
                        data_dict["INDEX_CAT"]["YLD_AT"].append(yield_step)
                        
                        ## Problem divisions
                        division_points: list[DivisionPoint] = []
                        if concatenated_plan.is_refined:
                            division_points = hierarchical_plan.get_division_points(level + 1)
                        
                        division_point: Optional[DivisionPoint] = None
                        for point in division_points:
                            if point.index == index:
                                division_point = point
                        data_dict["INDEX_CAT"]["IS_DIV"].append(division_point is not None)
                        data_dict["INDEX_CAT"]["IS_INHERITED"].append(division_point is not None and division_point.inherited)
                        data_dict["INDEX_CAT"]["IS_PROACTIVE"].append(division_point is not None and division_point.proactive)
                        data_dict["INDEX_CAT"]["IS_INTERRUPT"].append(division_point is not None and division_point.interrupting)
                        data_dict["INDEX_CAT"]["PREEMPTIVE"].append(division_point is not None and division_point.preemptive != 0)
                        
                        ## Sub-plan wise planning times
                        inc_stats: dict[int, Statistics] = concatenated_plan.planning_statistics.incremental
                        sub_plan_steps: list[int] = conformance_mapping.current_sgoals(index)
                        inc_stats = {step : inc_stats.get(step, Statistics(0.0, 0.0)) for step in sub_plan_steps}
                        data_dict["INDEX_CAT"]["SP_RE_GT"].append(sum(stat.grounding_time for stat in inc_stats.values()))
                        data_dict["INDEX_CAT"]["SP_RE_ST"].append(sum(stat.solving_time for stat in inc_stats.values()))
                        data_dict["INDEX_CAT"]["SP_RE_TT"].append(sum(stat.total_time for stat in inc_stats.values()))
                        
                        ## Refined sub-plan quality
                        index_factor: Planner.Expansion = concatenated_plan.get_expansion_factor(index)
                        data_dict["INDEX_CAT"]["SP_START_S"].append(min(sub_plan_steps))
                        data_dict["INDEX_CAT"]["SP_END_S"].append(max(sub_plan_steps))
                        data_dict["INDEX_CAT"]["SP_L"].append(index_factor.length)
                        data_dict["INDEX_CAT"]["SP_A"].append(index_factor.action)
                        
                        ## Sub-plan interleaving quantity
                        data_dict["INDEX_CAT"]["INTER_Q"].append(concatenated_plan.interleaving_quantity(index))
                        
                        ## Sub-plan majority action type
                        sub_plan_type: Planner.ActionType = concatenated_plan.get_sub_plan_type(index)
                        data_dict["INDEX_CAT"]["IS_LOCO"].append(sub_plan_type == Planner.ActionType.Locomotion)
                        data_dict["INDEX_CAT"]["IS_MANI"].append(sub_plan_type == Planner.ActionType.Manipulation)
                        data_dict["INDEX_CAT"]["IS_CONF"].append(sub_plan_type == Planner.ActionType.Configuration)
                
                ## Partial-Plans
                if hierarchical_plan.is_hierarchical_refinement:
                    for problem_number, iteration in enumerate(hierarchical_plan.partial_plans[level], start=1):
                        partial_plan: Planner.MonolevelPlan = hierarchical_plan.partial_plans[level][iteration]
                        partial_totals: Planner.ASH_Statistics = partial_plan.planning_statistics.grand_totals
                        data_dict["PAR"]["RU"].append(run)
                        data_dict["PAR"]["AL"].append(level)
                        data_dict["PAR"]["IT"].append(iteration)
                        data_dict["PAR"]["PN"].append(problem_number)
                        
                        ## Raw timing statistics
                        data_dict["PAR"]["GT"].append(partial_totals.grounding_time)
                        data_dict["PAR"]["ST"].append(partial_totals.solving_time)
                        data_dict["PAR"]["OT"].append(partial_totals.overhead_time)
                        data_dict["PAR"]["TT"].append(partial_totals.total_time)
                        
                        ## Online hierarchical planning statistics
                        data_dict["PAR"]["YT"].append(hierarchical_plan.get_yield_time(level, iteration))
                        data_dict["PAR"]["WT"].append(hierarchical_plan.get_wait_time(level, iteration))
                        data_dict["PAR"]["ET"].append(hierarchical_plan.get_minimum_execution_time(level, iteration))
                        
                        ## Required memory usage
                        data_dict["PAR"]["RSS"].append(partial_totals.memory.rss)
                        data_dict["PAR"]["VMS"].append(partial_totals.memory.vms)
                        
                        ## Partal plan quality
                        data_dict["PAR"]["LE"].append(partial_plan.plan_length)
                        data_dict["PAR"]["AC"].append(partial_plan.total_actions)
                        data_dict["PAR"]["CF"].append(partial_plan.compression_factor)
                        data_dict["PAR"]["PSG"].append(partial_plan.total_produced_sgoals)
                        data_dict["PAR"]["START_S"].append(partial_plan.action_start_step)
                        data_dict["PAR"]["END_S"].append(partial_plan.end_step)
                        
                        ## Conformance constraints
                        problem_size: int = 1
                        sgoal_literals_total: int = 0
                        sgoals_range = SubGoalRange(1, 1)
                        if partial_plan.is_refined:
                            problem_size = partial_plan.conformance_mapping.problem_size
                            sgoal_literals_total = partial_plan.conformance_mapping.total_sgoal_literals
                            sgoals_range = partial_plan.conformance_mapping.constraining_sgoals_range
                        data_dict["PAR"]["SIZE"].append(problem_size)
                        data_dict["PAR"]["SGLITS_T"].append(sgoal_literals_total)
                        data_dict["PAR"]["FIRST_I"].append(sgoals_range.first_index)
                        data_dict["PAR"]["LAST_I"].append(sgoals_range.last_index)
                        
                        factor: Planner.Expansion = partial_plan.get_plan_expansion_factor()
                        deviation: Planner.Expansion = partial_plan.get_expansion_deviation()
                        balance: Planner.Expansion = partial_plan.get_degree_of_balance()
                        data_dict["PAR"]["PP_EF_L"].append(factor.length)
                        data_dict["PAR"]["PP_EF_A"].append(factor.action)
                        data_dict["PAR"]["SP_ED_L"].append(deviation.length)
                        data_dict["PAR"]["SP_ED_A"].append(deviation.action)
                        data_dict["PAR"]["SP_EB_L"].append(balance.length)
                        data_dict["PAR"]["SP_EB_A"].append(balance.action)
                        
                        length_balance_score: float = 1.0
                        action_balance_score: float = 1.0
                        if deviation.length > 0.0:
                            length_balance_score = (1.0 - (math.log(deviation.length + 1.0) / math.log(problem_size)))
                        if deviation.action > 0.0:
                            action_balance_score = (1.0 - (math.log(deviation.action + 1.0) / math.log(problem_size)))
                        data_dict["PAR"]["SP_EBS_L"].append(length_balance_score)
                        data_dict["PAR"]["SP_EBS_A"].append(action_balance_score)
                        
                        ## Final-goal preemptive achievement
                        data_dict["PAR"]["TOT_CHOICES"].append(partial_plan.total_choices)
                        data_dict["PAR"]["PRE_CHOICES"].append(partial_plan.preemptive_choices)
        
        ## Create a Pandas dataframe from the data dictionary
        self.__dataframes = {key : pandas.DataFrame(data_dict[key]) for key in data_dict}
        return self.__dataframes
    
    def to_dsv(self, file: str, sep: str = " ", endl: str = "\n", index: bool = True) -> None:
        """Save the currently collected data to a Delimiter-Seperated Values (DSV) file."""
        dataframes = self.process()
        dataframes["CAT"].to_csv(file, sep=sep, line_terminator=endl, index=index)
    
    def to_excel(self, file: str) -> None:
        """Save the currently collected data to an excel file."""
        dataframes = self.process()
        top_level: int = self.__plans[-1].top_level
        writer = pandas.ExcelWriter(file, engine="xlsxwriter") # pylint: disable=abstract-class-instantiated
        
        ## General global statistics
        dataframes["GLOBALS"].to_excel(writer, sheet_name="Globals")
        dataframes["GLOBALS"].describe().to_excel(writer, sheet_name="Globals", startrow=len(self.__plans) + 2)
        worksheet = writer.sheets["Globals"]
        worksheet.write(len(self.__plans) + 12, 0, "Successful Runs")
        worksheet.write(len(self.__plans) + 12, 1, self.__successful_runs)
        worksheet.write(len(self.__plans) + 13, 0, "Failed Runs")
        worksheet.write(len(self.__plans) + 13, 1, self.__failed_runs)
        
        ## Problem definitions statistics
        if "PROBLEM_SEQUENCE" in dataframes:
            dataframes["PROBLEM_SEQUENCE"].to_excel(writer, sheet_name="Problem Sequence")
        if "DIVISIONS" in dataframes:
            dataframes["DIVISIONS"].to_excel(writer, sheet_name="Division Points")
        
        ## Concatenated plan statistics
        dataframes["CAT"].to_excel(writer, sheet_name="Cat Plans")
        self.cat_level_wise_means.to_excel(writer, sheet_name="Cat Level-Wise Aggregates", startrow=1)
        self.cat_level_wise_stdev.to_excel(writer, sheet_name="Cat Level-Wise Aggregates", startrow=((top_level + 3) * 1) + 1)
        worksheet = writer.sheets["Cat Level-Wise Aggregates"]
        worksheet.write(0, 0, "Means")
        worksheet.write(top_level + 3, 0, "Standard Deviation")
        quantiles = self.cat_level_wise_quantiles
        for order, quantile in enumerate([0.0, 0.25, 0.5, 0.75, 1.0], start=2):
            quantile_data = quantiles[quantiles["level_1"].isin([quantile])].drop("level_1", axis="columns")
            quantile_data.to_excel(writer, sheet_name="Cat Level-Wise Aggregates", startrow=((top_level + 3) * order) + 1)
            worksheet.write((top_level + 3) * order, 0, f"Quantile {quantile}")
        
        ## Partial plan statistics
        if "PAR" in dataframes:
            dataframes["PAR"].to_excel(writer, sheet_name="Partial Plans")
            self.par_level_wise_means.to_excel(writer, sheet_name="Par Level-Wise Aggregates", startrow=1)
            self.par_level_wise_stdev.to_excel(writer, sheet_name="Par Level-Wise Aggregates", startrow=((top_level + 3) * 1) + 1)
            worksheet = writer.sheets["Par Level-Wise Aggregates"]
            worksheet.write(0, 0, "Means")
            worksheet.write(top_level + 3, 0, "Standard Deviation")
            quantiles = self.par_level_wise_quantiles
            for order, quantile in enumerate([0.0, 0.25, 0.5, 0.75, 1.0], start=2):
                quantile_data = quantiles[quantiles["level_1"].isin([quantile])].drop("level_1", axis="columns")
                quantile_data.to_excel(writer, sheet_name="Par Level-Wise Aggregates", startrow=((top_level + 3) * order) + 1)
                worksheet.write((top_level + 3) * order, 0, f"Quantile {quantile}")
            max_problems: int = len(self.par_problem_wise_means["PN"])
            self.par_problem_wise_means.to_excel(writer, sheet_name="Par Problem-Wise Aggregates", startrow=1)
            self.par_problem_wise_stdev.to_excel(writer, sheet_name="Par Problem-Wise Aggregates", startrow=((max_problems + 3) * 1) + 1)
            worksheet = writer.sheets["Par Problem-Wise Aggregates"]
            worksheet.write(0, 0, "Means")
            worksheet.write(max_problems + 3, 0, "Standard Deviation")
            quantiles = self.par_problem_wise_quantiles
            for order, quantile in enumerate([0.0, 0.25, 0.5, 0.75, 1.0], start=2):
                quantile_data = quantiles[quantiles["level_2"].isin([quantile])].drop("level_2", axis="columns")
                quantile_data.to_excel(writer, sheet_name="Par Problem-Wise Aggregates", startrow=((max_problems + 3) * order) + 1)
                worksheet.write((max_problems + 3) * order, 0, f"Quantile {quantile}")
        
        ## Step- and index-wise statistics
        dataframes["STEP_CAT"].to_excel(writer, sheet_name="Concat Step-wise")
        self.step_wise_means.to_excel(writer, sheet_name="Concat Step-wise Mean")
        self.step_wise_stdev.to_excel(writer, sheet_name="Concat Step-wise Stdev")
        if "INDEX_CAT" in dataframes:
            dataframes["INDEX_CAT"].to_excel(writer, sheet_name="Concat Index-wise")
            self.index_wise_means.to_excel(writer, sheet_name="Concat Index-wise Mean")
            self.index_wise_stdev.to_excel(writer, sheet_name="Concat Index-wise Stdev")
        
        writer.save()

class Experiment:
    """Encapsulates an experiment to be ran."""
    
    __slots__ = ("__planner",
                 "__planning_function",
                 "__optimums",
                 "__bottom_level",
                 "__top_level",
                 "__initial_runs",
                 "__experimental_runs",
                 "__enable_tqdm")
    
    def __init__(self,
                 planner: Planner.HierarchicalPlanner,
                 planning_function: Callable[[], Any],
                 optimums: Optional[Union[int, dict[int, int]]],
                 bottom_level: int,
                 top_level: int,
                 initial_runs: int,
                 experimental_runs: int,
                 enable_tqdm: bool
                 ) -> None:
        """Create an experiment."""
        self.__planner: Planner.HierarchicalPlanner = planner
        self.__planning_function: Callable[[], Any] = planning_function
        self.__optimums: Optional[dict[int, int]] = None
        if optimums is not None:
            if isinstance(optimums, int):
                self.__optimums = {bottom_level : optimums}
            else: self.__optimums = optimums
        self.__bottom_level: int = bottom_level
        self.__top_level: int = top_level
        self.__initial_runs: int = initial_runs
        self.__experimental_runs: int = experimental_runs
        self.__enable_tqdm: bool = enable_tqdm
    
    def run_experiments(self) -> Results:
        """Run the all experiments and return a results object containing obtained statistics."""
        results: Results = self.__run_all()
        results.process()
        
        columns: list[str] = ["LE", "AC", "QL_SCORE",
                              "GT", "ST", "OT", "TT", "LT", "CT", "WT", "MET_PA", "TI_SCORE",
                              "RSS", "VMS", "GRADE"]
        _EXP_logger.info("\n\n" + center_text("Experimental Results",
                                              framing_char='=', framing_width=54, centering_width=60)
                         + "\n\n" + center_text("Concatenated Plan Level-Wise Means",
                                                frame_after=False, framing_char='~', framing_width=50, centering_width=60)
                         + "\n" + results.cat_level_wise_means.to_string(columns=columns)
                         + "\n\n" + center_text("Concatenated Plan Level-Wise Standard Deviation",
                                                frame_after=False, framing_char='~', framing_width=50, centering_width=60)
                         + "\n" + results.cat_level_wise_stdev.to_string(columns=columns))
        
        return results
    
    def __run_all(self) -> Results:
        """Worker function that actually runs the experiments and returns an unprocessed results object containing obtained statistics."""
        _EXP_logger.info("\n\n" + center_text(f"Running experiments : Initial runs = {self.__initial_runs} : Experimental runs = {self.__experimental_runs}",
                                              framing_width=96, centering_width=100, framing_char="#"))
        
        results = Results(self.__optimums)
        hierarchical_plan: Planner.HierarchicalPlan
        planning_time: float
        
        ## Do initial runs
        for run in tqdm.tqdm(range(1, self.__initial_runs + 1), desc="Initial runs completed", disable=not self.__enable_tqdm, leave=False, ncols=180, colour="white", unit="run"):
            hierarchical_plan, planning_time = self.__run()
            _EXP_logger.log(logging.DEBUG if self.__enable_tqdm else logging.INFO,
                            "\n\n" + center_text(f"Initial run {run} : Time {planning_time:.6f}s",
                                                 framing_width=48, centering_width=60))
        
        experiment_real_start_time = time.perf_counter()
        experiment_process_start_time = time.process_time()
        successful_runs: int = 0
        failed_runs: int = 0
        
        ## Do experimental runs
        for run in tqdm.tqdm(range(1, self.__experimental_runs + 1), desc="Experimental runs completed", disable=not self.__enable_tqdm, leave=False, ncols=180, colour="white", unit="run"):
            hierarchical_plan, planning_time = self.__run()
            
            if (success := hierarchical_plan is not None):
                results.add(hierarchical_plan)
                successful_runs += 1
            else: failed_runs += 1
            
            _EXP_logger.log(logging.DEBUG if self.__enable_tqdm else logging.INFO,
                            "\n\n" + center_text(f"Experimental run {run} : {'SUCCESSFUL' if success else 'FAILED'} : Time {planning_time:.6f}s",
                                                 framing_width=54, centering_width=60))
            
            if successful_runs == 0 and failed_runs > 10:
                _EXP_logger.info("\n\n" + center_text(f"Experiment abandoned after all of first 10 runs failed : "
                                                      f"Real time {experiment_real_total_time:.6f}s, "
                                                      f"Proccess time {experiment_process_total_time:.6f}s",
                                                      framing_width=96, centering_width=100, framing_char="#"))
                return results
        
        results.runs_completed(successful_runs, failed_runs)
        
        experiment_real_total_time: float = time.perf_counter() - experiment_real_start_time
        experiment_process_total_time: float = time.process_time() - experiment_process_start_time
        
        _EXP_logger.info("\n\n" + center_text(f"Completed {self.__experimental_runs} experimental runs : "
                                              f"Real time {experiment_real_total_time:.6f}s, "
                                              f"Proccess time {experiment_process_total_time:.6f}s",
                                              framing_width=96, centering_width=100, framing_char="#"))
        
        return results
    
    def __run(self) -> tuple[Optional[Planner.HierarchicalPlan], float]:
        """Run the planner with this experiment's planning function once and return the plan and total run time."""
        run_start_time: float = time.perf_counter()
        
        ## Generate one plan per run
        hierarchical_plan: Optional[Planner.HierarchicalPlan] = None
        try:
            self.__planning_function()
            hierarchical_plan = self.__planner.get_hierarchical_plan(bottom_level=self.__bottom_level,
                                                                     top_level=self.__top_level)
        except Planner.ASH_NoSolutionError as error: pass
        
        ## Ensure that the planner is purged after reach run
        self.__planner.purge_solutions()
        
        run_total_time: float = time.perf_counter() - run_start_time
        
        return hierarchical_plan, run_total_time
