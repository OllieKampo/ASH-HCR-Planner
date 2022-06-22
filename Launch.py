###########################################################################
###########################################################################
## Interactive terminal script for ASH                                   ##
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

import argparse
import datetime
import functools
import json
import logging
import logging.handlers
import os
import sys
import time
from typing import Any, Optional, Sequence, Type, Union

import numpy
import pandas
from matplotlib import pyplot
from matplotlib.backend_bases import FigureManagerBase
from pandas.core.frame import DataFrame

import core.Planner as Planner
import core.Strategies as Strategies
import Experiment
from core.Helpers import center_text

## Main module logger
_Launcher_logger: logging.Logger = logging.getLogger(__name__)
_Launcher_logger.setLevel(logging.DEBUG)

Number = Union[int, float]

_ASH_TITLE: str = """
░█████╗░░██████╗██╗░░██╗
██╔══██╗██╔════╝██║░░██║
███████║╚█████╗░███████║
██╔══██║░╚═══██╗██╔══██║
██║░░██║██████╔╝██║░░██║
╚═╝░░╚═╝╚═════╝░╚═╝░░╚═╝
 
=======================================================================
ASH - The ASP based Hierarchical Conformance Refinement Planner
Copyright (C)  2021  Oliver Michael Kamperis
=======================================================================
 
This program comes with ABSOLUTELY NO WARRANTY; for details use `--warranty'. This is free software,
and you are welcome to redistribute it under certain conditions; use `--conditions' for details.
For help and usage instructions use `--help' and `--instructions' respectively.
"""

_ASH_VERSION: str = "PROTOTYPE v0.4.2"

_ASH_WARRANTY: str = """
THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.
"""

_ASH_CONDITIONS: str = """
ASH - The ASP based Hierarchical Conformance Refinement Planner
Copyright (C)  2021  Oliver Michael Kamperis
Email: o.m.kamperis@gmail.com
 
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

def __main() -> int:
    "Main method which creates a console session, runs the planner, and returns 0 if the console returns cleanly."
    
    ## Run initial setup and get CLI arguments
    namespace: argparse.Namespace = __setup()
    
    ## Print the headers; title, warranty and distribution conditions as requested
    print(center_text(_ASH_TITLE, prefix_blank_line=True))
    if namespace.warranty:
        print(center_text(_ASH_WARRANTY, framing_width=80, append_blank_line=True))
    if namespace.conditions:
        print(center_text(_ASH_CONDITIONS, framing_width=80, append_blank_line=True))
    
    ## Pause to let the user read the headers
    if not namespace.disable_pause_on_start:
        input("Press any key to begin...")
        sys.stdout.write("\033[A")
        sys.stdout.flush()
    
    ## Find the verbosity mode;
    ##      - Experiment ouput mode disables outputs from the planner and only experimental results are shown
    if namespace.ash_output != "experiment":
        verbosity = Planner.Verbosity[namespace.ash_output.capitalize()]
    else: verbosity = Planner.Verbosity.Minimal
    
    ## Setup the planner
    planner = Planner.HierarchicalPlanner(namespace.files, name="Main", threads=namespace.threads,
                                          verbosity=verbosity, silence_clingo=not namespace.clingo_output)
    
    ## Determine valid abstaction level range
    level_range: range = planner.constrained_level_range(bottom_level=namespace.bottom_level,
                                                         top_level=(namespace.bottom_level
                                                                    if namespace.planning_mode == "mcl" else
                                                                    namespace.top_level))
    bottom_level: int = min(level_range)
    top_level: int = max(level_range)
    
    ## Initialise the planning problem
    find_inconsistencies: bool = namespace.operation == "find-problem-inconsistencies"
    planner.initialise_problem(find_inconsistencies)
    if find_inconsistencies: return 0
    
    conformance_type: Optional[Planner.ConformanceType] = None
    division_strategy: Optional[Strategies.DivisionStrategy] = None
    
    if namespace.conformance_type is not None:
        conformance_type = Planner.ConformanceType(namespace.conformance_type)
    
    def get_hierarchical_arg(dict_or_number: Union[Number, dict[int, Number]],
                         level: int,
                         default: Optional[Number] = None
                         ) -> Optional[Number]:
        "Get the value of a single hierarchical argument at a given abstraction level."
        if isinstance(dict_or_number, dict):
            if (arg := dict_or_number.get(level, default)) is not None:
                return arg
            else: return default
        return dict_or_number
    
    def get_hierarchical_args(dict_or_number: Union[Number, dict[int, Number]],
                              default: Optional[Number] = None
                              ) -> Optional[Union[Number, dict[int, Number]]]:
        if isinstance(dict_or_number, dict):
            hierarchical_args: dict[int, Number] = {}
            for level in planner.domain.level_range:
                if (arg := dict_or_number.get(level, default)) is not None:
                    hierarchical_args[level] = arg
                else: hierarchical_args[level] = default
            return hierarchical_args
        return dict_or_number
    
    ## Planning mode is hierarchical (conformance refinement or classical)
    if namespace.planning_mode in ["hcr", "hcl"]:
        
        ## Determine division strategy for conformance refinement planning
        refinement_planning: bool = namespace.planning_mode == "hcr"
        division_strategy: Optional[Strategies.DivisionStrategy] = None
        
        if refinement_planning:
            ## If a division strategy is given then online planning is enabled
            if namespace.division_strategy != "none":
                
                ## Obtain the specified division strategy
                division_strategy_name: str
                proactive_basis_name: str = ""
                division_strategy_class: Type[Strategies.DivisionStrategy]
                proactive_basis: Type[Strategies.DivisionStrategy]
                
                if len(strategy_name_split := namespace.division_strategy.split('-')) == 2:
                    proactive_basis_name = strategy_name_split[1]
                    proactive_basis = Strategies.GetStrategy[proactive_basis_name].value
                
                division_strategy_name = strategy_name_split[0]
                division_strategy_class = Strategies.GetStrategy[division_strategy_name].value
                
                ## Obtain the bounds, blends and backwards horizon for the strategy
                bounds: dict[int, Union[Number, tuple[Number]]] = {}
                blend: dict[int, Strategies.Blend] = {}
                horizon: dict[int, Number] = {}
                
                for level in level_range:
                    bounds[level] = get_hierarchical_arg(namespace.division_strategy_bounds, level)
                    blend[level] = Strategies.Blend(get_hierarchical_arg(namespace.left_blend_quantities, level, 0),
                                                    get_hierarchical_arg(namespace.right_blend_quantities, level, 0))
                    horizon[level] = get_hierarchical_arg(namespace.backwards_horizon, level)
                    
                    ## Blending together more than half of partial problems is generally not advisable;
                    ##      - In particular, if the right blend of problem X overlaps with the left blend of problem X + 2, then problem X + 1
                    if ((isinstance((left := blend[level].left), float) and left > 0.5)
                        or isinstance((right := blend[level].right), float) and right > 0.5):
                        _Launcher_logger.warn("Blend quantities of greater than 0.5 are generally detrimental to performance.")
                    
                    ## Ensure that there is not a left blend requiring revision of a refined plan on a saved grounding;
                    ##      - It is not possible to do this since it is not possible to change any sub-goal stages that have already been committed.
                    if (namespace.save_grounding
                        and not namespace.avoid_refining_sgoals_marked_for_blending
                        and blend[level].left > 0):
                        raise ValueError("It is not possible to left blend on a saved grounding without avoiding refining sgoals marked for blending.")
                
                reactive_bound_type = Strategies.ReactiveBoundType(f"{namespace.bound_type}_bound")
                moving_average: int = namespace.moving_average
                preemptive: bool = namespace.preemptive_division
                interrupting: bool = namespace.interrupting_division
                independent_tasks: bool = namespace.treat_tasks_as_independent
                order_tasks: bool = namespace.divide_tasks_on_final_goal_intermediate_achievement_ordering
                
                if issubclass(division_strategy_class, Strategies.Basic):
                    division_strategy = division_strategy_class(top_level=top_level,
                                                                problems=bounds,
                                                                blend=blend,
                                                                independent_tasks=independent_tasks,
                                                                order_tasks=order_tasks)
                
                if issubclass(division_strategy_class, Strategies.NaiveProactive):
                    division_strategy = division_strategy_class(top_level=top_level,
                                                                size_bound=bounds,
                                                                blend=blend,
                                                                independent_tasks=independent_tasks,
                                                                order_tasks=order_tasks)
                
                if division_strategy_class == Strategies.Relentless:
                    division_strategy = division_strategy_class(top_level=top_level,
                                                                time_bound=bounds,
                                                                bound_type=reactive_bound_type,
                                                                backwards_horizon=horizon,
                                                                moving_average=moving_average,
                                                                preemptive=preemptive,
                                                                interrupting=interrupting,
                                                                independent_tasks=independent_tasks,
                                                                order_tasks=order_tasks)
                
                if issubclass(division_strategy_class, Strategies.Impetuous):
                    if not isinstance(_bound := next(iter(bounds.values())), tuple) or len(_bound) != 2:
                        raise ValueError("The impetuous division strategy requires exactly two bounds per level.")
                    
                    division_strategy = division_strategy_class(top_level=top_level,
                                                                cumulative_time_bound={level : bound[0] if bound is not None else None
                                                                                       for level, bound in bounds.items()},
                                                                continuous_time_bound={level : bound[1] if bound is not None else None
                                                                                       for level, bound in bounds.items()},
                                                                continuous_bound_type=reactive_bound_type,
                                                                backwards_horizon=horizon,
                                                                moving_average=moving_average,
                                                                preemptive=preemptive,
                                                                independent_tasks=independent_tasks,
                                                                order_tasks=order_tasks)
                
                if issubclass(division_strategy_class, Strategies.Rapid):
                    division_strategy = division_strategy_class(top_level=top_level,
                                                                proactive_basis=proactive_basis,
                                                                size_bound={level : bound[0] if bound is not None else None
                                                                            for level, bound in bounds.items()},
                                                                reactive_time_bound={level : bound[1] if bound is not None else None
                                                                                     for level, bound in bounds.items()},
                                                                reactive_bound_type=reactive_bound_type,
                                                                backwards_horizon=horizon,
                                                                moving_average=moving_average,
                                                                independent_tasks=independent_tasks,
                                                                order_tasks=order_tasks)
        
        planning_function = functools.partial(planner.hierarchical_plan,
                                              bottom_level,
                                              top_level,
                                              namespace.enable_concurrency,
                                              refinement_planning,
                                              
                                              conformance_type=conformance_type,
                                              sequential_yield=namespace.sequential_yielding,
                                              division_strategy=division_strategy,
                                              online_method=Planner.OnlineMethod(namespace.online_method),
                                              save_grounding=namespace.save_grounding,
                                              use_search_length_bound=namespace.minimum_search_length_bound,
                                              avoid_refining_sgoals_marked_for_blending=namespace.avoid_refining_sgoals_marked_for_blending,
                                              make_observable=namespace.make_observable,
                                              
                                              minimise_actions=namespace.minimise_actions,
                                              order_fgoals_achievement=namespace.final_goal_intermediate_achievement_ordering_preferences,
                                              preempt_pos_fgoals=namespace.positive_final_goal_preemptive_achievement,
                                              preempt_neg_fgoals=namespace.negative_final_goal_preemptive_achievement,
                                              preempt_mode=Planner.PreemptMode(namespace.final_goal_preemptive_achievement_mode),
                                              
                                              detect_interleaving=namespace.detect_interleaving,
                                              generate_search_space=False,
                                              generate_solution_space=False,
                                              
                                              time_limit=get_hierarchical_args(namespace.planning_time_limit),
                                              length_limit=get_hierarchical_args(namespace.search_length_limit),
                                              
                                              pause_on_level_change=namespace.pause_on_level_change,
                                              pause_on_increment_change=namespace.pause_on_increment_change)
    
    ## Planning mode is monolevel classical
    else:
        planning_function = functools.partial(planner.monolevel_plan,
                                              bottom_level,
                                              namespace.enable_concurrency,
                                              False,
                                              minimise_actions=namespace.minimise_actions,
                                              time_limit=get_hierarchical_arg(namespace.planning_time_limit, bottom_level),
                                              length_limit=get_hierarchical_arg(namespace.search_length_limit, bottom_level))
    
    ## Test mode runs the planner once, the outputs are;
    ##      - A standard log file,
    ##      - A plan and refinement schema in json format,
    ##      - A figure file displaying basic statistics.
    if namespace.operation == "test":
        planning_function()
        
        ## Get the resulting plans
        hierarchical_plan: Planner.HierarchicalPlan = planner.get_hierarchical_plan(bottom_level=bottom_level,
                                                                                    top_level=top_level)
        
        ## Save the plans as requested
        if (plan_file := namespace.plan_file) is not None:
            
            if namespace.config_file_naming:
                plan_file = plan_file.split(".txt")[0] + f"_{config_file_name}" + ".txt"
            _Launcher_logger.info(f"Saving generated plan to file: {plan_file}")
            
            try:
                with open(plan_file, 'w') as file_writer:
                    file_writer.write(json.dumps(hierarchical_plan.serialisable_dict, indent=4))
            except:
                _Launcher_logger.error("Failed to save plan to file.", exc_info=1)
        
        ## Save the refinement schema as requested
        if (namespace.planning_mode == "hcr"
            and (schema_file := namespace.save_schema) is not None):
            
            if namespace.config_file_naming:
                schema_file = schema_file.split(".txt")[0] + f"_{config_file_name}" + ".txt"
            _Launcher_logger.info(f"Saving generated refinement schema to file: {schema_file}")
            
            try:
                with open(schema_file, 'w') as file_writer:
                    file_writer.write(json.dumps(hierarchical_plan.get_refinement_schema(namespace.schema_level).serialisable_dict, indent=4))
            except:
                _Launcher_logger.error("Failed to save schema to file.", exc_info=1)
        
        ## Graphify statistics as requested
        if namespace.display_figure:
            
            bar_width: float
            
            ## Find the regression plots for each partial plan
            regression_lines: dict[int, dict[str, Any]] = {"total" : {}, "ground" : {}, "search" : {}}
            for problem_number, partial_plan in enumerate(hierarchical_plan.get_plan_sequence(bottom_level)):
                try:
                    func, x_points, y_points, popt, pcov = partial_plan.regress_total_time
                    regression_lines["total"][problem_number] = {"func" : func, "x_points" : x_points, "y_points" : y_points, "popt" : popt, "pcov" : pcov}
                    func, x_points, y_points, popt, pcov = partial_plan.regress_grounding_time
                    regression_lines["ground"][problem_number] = {"func" : func, "x_points" : x_points, "y_points" : y_points, "popt" : popt, "pcov" : pcov}
                    func, x_points, y_points, popt, pcov = partial_plan.regress_solving_time
                    regression_lines["search"][problem_number] = {"func" : func, "x_points" : x_points, "y_points" : y_points, "popt" : popt, "pcov" : pcov}
                except: pass
            
            ## Generate four graphs;
            ##      - Planning statistics per abstraction level bar chart,
            ##      - Ground level planning times against search length (one line per abstraction level), (conformance mapping here too?)
            ##      - Total number of achieved sub-goal stages against search length (one line per abstraction level), (deviation and balance here too?)
            ##      - Planning time per online planning increment.
            figure, axes = pyplot.subplots(2, 2)
            
            xlabels = [str(n) for n in reversed(hierarchical_plan.level_range)]
            x = numpy.arange(len(xlabels))
            
            grounding_times, solving_times, total_times, latency_times, completion_times = [], [], [], [], []
            memory_rss, memory_vms = [], []
            concat_length, concat_actions = [], []
            concat_length_expansion, concat_actions_expansion = [], []
            concat_subplan_length_deviation, concat_subplan_actions_deviation = [], []
            concat_subplan_length_balance, concat_subplan_actions_balance = [], []
            
            for level in reversed(hierarchical_plan.level_range):
                overall_totals = hierarchical_plan.get_grand_totals(level)
                
                grounding_times.append(overall_totals.grounding_time)
                solving_times.append(overall_totals.solving_time)
                total_times.append(overall_totals.total_time)
                latency_times.append(hierarchical_plan.get_latency_time(level))
                completion_times.append(hierarchical_plan.get_completion_time(level))
                
                memory_rss.append(overall_totals.memory.rss)
                memory_vms.append(overall_totals.memory.vms)
                
                concat_length.append(hierarchical_plan.concatenated_plans[level].plan_length)
                concat_actions.append(hierarchical_plan.concatenated_plans[level].total_actions)
                
                factor: Planner.Expansion = hierarchical_plan.concatenated_plans[level].get_plan_expansion_factor()
                deviation: Planner.Expansion = hierarchical_plan.concatenated_plans[level].get_expansion_deviation()
                balance: Planner.Expansion = hierarchical_plan.concatenated_plans[level].get_degree_of_balance()
                
                concat_length_expansion.append(factor.length)
                concat_actions_expansion.append(factor.action)
                concat_subplan_length_deviation.append(deviation.length)
                concat_subplan_actions_deviation.append(deviation.action)
                concat_subplan_length_balance.append(balance.length)
                concat_subplan_actions_balance.append(balance.action)
            
            ## Hierarchical grand total planning time statistics
            bar_width = 0.19
            axes[0, 0].bar(x - (bar_width * 2.0), grounding_times, bar_width, label="Grounding Time (s)")
            axes[0, 0].bar(x - (bar_width * 1.0), solving_times, bar_width, label="Solving Time (s)")
            axes[0, 0].bar(x, total_times, bar_width, label="Total Time (s)")
            axes[0, 0].bar(x + (bar_width * 1.0), latency_times, bar_width, label="Latency Time (s)")
            axes[0, 0].bar(x + (bar_width * 2.0), completion_times, bar_width, label="Completion Time (s)")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(xlabels)
            axes[0, 0].set_ylabel("Time (s)")
            axes[0, 0].set_xlabel("Abstraction level")
            axes[0, 0].legend()
            
            ## Hierarchical required memory statistics
            bar_width = 0.45
            axes[0, 1].bar(x - (bar_width * 0.5), memory_rss, bar_width, label="Resident Set Size")
            axes[0, 1].bar(x + (bar_width * 0.5), memory_vms, bar_width, label="Virtual Memory Size")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(xlabels)
            axes[0, 1].set_ylabel("Required Memory (Mb)")
            axes[0, 1].set_xlabel("Abstraction level")
            axes[0, 1].legend()
            
            ## Hierarchical concatenated plan quality statistics
            bar_width = 0.45
            axes[1, 0].bar(x - (bar_width * 0.5), concat_length, bar_width, label="Concatenated Length")
            axes[1, 0].bar(x + (bar_width * 0.5), concat_actions, bar_width, label="Concatenated Actions")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(xlabels)
            axes[1, 0].set_ylabel("Plan Quality")
            axes[1, 0].set_xlabel("Abstraction level")
            axes[1, 0].legend()
            
            ## Problems per level
            bar_width = 0.15
            axes[1, 1].bar(x - (bar_width * 2.5), concat_length_expansion, bar_width, label="Concatenated length expansion")
            axes[1, 1].bar(x - (bar_width * 1.5), concat_actions_expansion, bar_width, label="Concatenated action expansion")
            axes[1, 1].bar(x - (bar_width * 0.5), concat_subplan_length_deviation, bar_width, label="Sub-plan length deviation")
            axes[1, 1].bar(x + (bar_width * 0.5), concat_subplan_actions_deviation, bar_width, label="Sub-plan action deviation")
            axes[1, 1].bar(x + (bar_width * 1.5), concat_subplan_length_balance, bar_width, label="Sub-plan length balance")
            axes[1, 1].bar(x + (bar_width * 2.5), concat_subplan_actions_balance, bar_width, label="Sub-plan action balance")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(xlabels)
            axes[1, 1].set_ylabel("Plan Refinement Expansion")
            axes[1, 1].set_xlabel("Abstraction level")
            axes[1, 1].legend()
            
            pyplot.show()
    
    else:
        _Launcher_logger.info("Experimental setup:" + (f" [configuration file loaded = {config_file_name}]\n\n" if config_file else "\n\n")
                              + f"Planning Mode            :: {'Hierarchical' if namespace.planning_mode.startswith('h') else 'Monolevel'} {'conformance refinement' if namespace.planning_mode.endswith('cr') else 'classical'}\n"
                              + "Domain and Problem Files :: " + f"\n{' '*len('Domain and Problem Files :: ')}".join(planner.domain.domain_files))
        
        ## Run the experiments
        experiment = Experiment.Experiment(planner=planner,
                                           planning_function=planning_function,
                                           optimums=get_hierarchical_args(namespace.optimum),
                                           bottom_level=bottom_level,
                                           top_level=top_level,
                                           initial_runs=namespace.initial_runs,
                                           experimental_runs=namespace.experimental_runs,
                                           enable_tqdm=namespace.ash_output == "experiment")
        results: Experiment.Results = experiment.run_experiments()
        dataframes: dict[str, DataFrame] = results.process()
        is_refinement: bool = namespace.planning_mode == "hcr"
        
        ## Save the results as requseted
        if (excel_file := namespace.excel_file) is not None:
            if namespace.config_file_naming:
                excel_file = excel_file.split(".xlsx")[0] + f"_{config_file_name}" + ".xlsx"
            _Launcher_logger.info(f"Saving results to excel file: {excel_file}")
            results.to_excel(excel_file)
        
        if (data_file := namespace.data_file) is not None:
            if namespace.config_file_naming:
                data_file = data_file.split(".dat")[0] + f"_{config_file_name}" + ".dat"
            _Launcher_logger.info(f"Saving results to data file: {data_file}")
            results.to_dsv(data_file, sep=namespace.data_sep, endl=namespace.data_end)
        
        ## Display a summary of results in simple graphs
        if (namespace.display_figure
            or namespace.figure_file is not None):
            
            means: DataFrame = results.cat_level_wise_means
            std: DataFrame = results.cat_level_wise_stdev
            step_wise_means: DataFrame = results.step_wise_means
            step_wise_std: DataFrame = results.step_wise_stdev
            index_wise_means: DataFrame
            index_wise_std: DataFrame
            if is_refinement:
                index_wise_means = results.index_wise_means
                index_wise_std = results.index_wise_stdev
            
            figure, axes = pyplot.subplots(3, 3)
            
            levels: list[int] = means["AL"].to_list()
            al_range = numpy.arange(len(levels))
            al_labels: list[str] = [str(al) for al in levels]
            
            bars: int = 1
            padding: float = 0.10
            bar_width: float = (1.0 / bars) - (padding / bars)
            
            def set_bars(bars: int) -> None:
                "Set the number of bars in the current plot."
                bars = bars
                nonlocal bar_width
                bar_width = (1.0 / bars) - (padding / bars)
            
            def get_std(name: str) -> Optional[pandas.Series]:
                """
                Get the standard deviation of a statistic safely,
                if there is no deviation (because only one run was done)
                then None is returned instead.
                """
                if not std[name].isnull().any():
                    return std[name]
                return None
            
            ## Level-wise timing statistics
            set_bars(5)
            axes[0, 0].bar(al_range - (bar_width * 2), means["GT"], bar_width, yerr=get_std("GT"), capsize=5, label="Mean Grounding Time")
            axes[0, 0].bar(al_range - bar_width, means["ST"], bar_width, yerr=get_std("ST"), capsize=5, label="Mean Solving")
            axes[0, 0].bar(al_range, means["TT"], bar_width, yerr=get_std("TT"), capsize=5, label="Mean Total")
            axes[0, 0].bar(al_range + bar_width, means["LT"], bar_width, yerr=get_std("LT"), capsize=5, label="Mean Latency")
            axes[0, 0].bar(al_range + (bar_width * 2), means["CT"], bar_width, yerr=get_std("CT"), capsize=5, label="Mean Completion")
            axes[0, 0].set_title("Mean Grand Total Computation Times per Abstraction Level")
            axes[0, 0].set_ylabel("Time (s)")
            axes[0, 0].set_xlabel("Abstraction Level")
            axes[0, 0].set_xticks(al_range)
            axes[0, 0].set_xticklabels(al_labels)
            axes[0, 0].legend(prop={"size" : "xx-small"})
            
            ## Level-wise memory statistics
            set_bars(2)
            axes[0, 1].bar(al_range - (bar_width / 2), means["VMS"], bar_width, yerr=get_std("VMS"), capsize=5, label="VMS")
            axes[0, 1].bar(al_range + (bar_width / 2), means["RSS"], bar_width, yerr=get_std("RSS"), capsize=5, label="RSS")
            axes[0, 1].set_title("Required Memory per Abstraction Level")
            axes[0, 1].set_ylabel("Total Memory (MBs)")
            axes[0, 1].set_xlabel("Abstraction Level")
            axes[0, 1].set_xticks(al_range)
            axes[0, 1].set_xticklabels(al_labels)
            axes[0, 1].legend(prop={"size" : "xx-small"})
            
            ## Level-wise quality statistics
            set_bars(2)
            axes[0, 2].bar(al_range - (bar_width / 2), means["LE"], bar_width, yerr=get_std("LE"), capsize=5, label="Length")
            axes[0, 2].bar(al_range + (bar_width / 2), means["AC"], bar_width, yerr=get_std("AC"), capsize=5, label="Actions")
            axes[0, 2].set_title("Concatenated Plan Quality per Abstraction Level")
            axes[0, 2].set_ylabel("Total")
            axes[0, 2].set_xlabel("Abstraction Level")
            axes[0, 2].set_xticks(al_range)
            axes[0, 2].set_xticklabels(al_labels)
            axes[0, 2].legend(prop={"size" : "xx-small"})
            
            ## Bottom-level step-wise timing statistics
            bottom_step_wise_means: DataFrame = step_wise_means[step_wise_means["AL"].isin([bottom_level])].sort_index()
            bottom_step_wise_std: DataFrame = step_wise_std[step_wise_std["AL"].isin([bottom_level])].sort_index()
            bottom_index_wise_means: DataFrame = DataFrame({})
            if is_refinement: bottom_index_wise_means = index_wise_means[index_wise_means["AL"].isin([bottom_level])].sort_index()
            steps: list[int] = bottom_step_wise_means["SL"].to_list()
            max_time: float = bottom_step_wise_means["S_TT"].add(bottom_step_wise_std["S_TT"], fill_value=0).max()
            axes[1, 0].plot(steps, bottom_step_wise_means["S_GT"], "g", label="Mean Step-Wise Grounding")
            axes[1, 0].plot(steps, bottom_step_wise_means["S_ST"], "b", label="Mean Step-Wise Solving")
            axes[1, 0].plot(steps, bottom_step_wise_means["S_TT"], "r", label="Mean Step-Wise Total")
            if is_refinement:
                axes[1, 0].bar(bottom_index_wise_means["YLD_AT"], max_time, width=0.20, color="magenta", label="Mean Yield Steps")
                axes[1, 0].bar(bottom_step_wise_means["SL"], [max_time if fuzzy_truth > 0.01 else 0 for fuzzy_truth in bottom_step_wise_means["IS_DIV_APP"]], width=0.20,
                               color=['#' + f"{round(int('FFFFFF', base=16) * (1.0 - fuzzy_truth)):06x}" for fuzzy_truth in bottom_step_wise_means["IS_DIV_APP"]], label="Problem Divisions")
            if namespace.experimental_runs > 1:
                axes[1, 0].plot(steps, bottom_step_wise_means["S_GT"] + bottom_step_wise_std["S_GT"], "--g")
                axes[1, 0].plot(steps, bottom_step_wise_means["S_ST"] + bottom_step_wise_std["S_ST"], "--b")
                axes[1, 0].plot(steps, bottom_step_wise_means["S_TT"] + bottom_step_wise_std["S_TT"], "--r")
                axes[1, 0].plot(steps, bottom_step_wise_means["S_GT"] - bottom_step_wise_std["S_GT"], "--g")
                axes[1, 0].plot(steps, bottom_step_wise_means["S_ST"] - bottom_step_wise_std["S_ST"], "--b")
                axes[1, 0].plot(steps, bottom_step_wise_means["S_TT"] - bottom_step_wise_std["S_TT"], "--r")
            axes[1, 0].set_title("Bottom-Level Search Times")
            axes[1, 0].set_ylabel("Time (s)")
            axes[1, 0].set_xlabel("Search Length")
            axes[1, 0].legend(prop={"size" : "xx-small"})
            
            ## Bottom-level step-wise memory statistics
            max_memory: float = bottom_step_wise_means["T_RSS"].add(bottom_step_wise_std["T_RSS"], fill_value=0).max()
            axes[1, 1].plot(steps, bottom_step_wise_means["T_VMS"], "g", label="Mean Step-Wise VMS")
            axes[1, 1].plot(steps, bottom_step_wise_means["T_RSS"], "b", label="Mean Step-Wise RSS")
            axes[1, 1].bar(bottom_step_wise_means["SL"], [max_memory if fuzzy_truth > 0.01 else 0 for fuzzy_truth in bottom_step_wise_means["IS_DIV_APP"]], width=0.20,
                           color=['#' + f"{round(int('FFFFFF', base=16) * (1.0 - fuzzy_truth)):06x}" for fuzzy_truth in bottom_step_wise_means["IS_DIV_APP"]], label="Problem Divisions")
            if namespace.experimental_runs > 1:
                axes[1, 1].plot(steps, bottom_step_wise_means["T_VMS"] + bottom_step_wise_std["T_VMS"], "--g")
                axes[1, 1].plot(steps, bottom_step_wise_means["T_RSS"] + bottom_step_wise_std["T_RSS"], "--b")
                axes[1, 1].plot(steps, bottom_step_wise_means["T_VMS"] - bottom_step_wise_std["T_VMS"], "--g")
                axes[1, 1].plot(steps, bottom_step_wise_means["T_RSS"] - bottom_step_wise_std["T_RSS"], "--b")
            axes[1, 1].set_title("Bottom-Level Memory Usage")
            axes[1, 1].set_ylabel("Total Memory (MBs)")
            axes[1, 1].set_xlabel("Search Length")
            axes[1, 1].legend(prop={"size" : "xx-small"})
            
            if is_refinement:
                ## Bottom-level index-wise sub-plan length
                set_bars(2)
                axes[1, 2].bar(numpy.array(bottom_index_wise_means["INDEX"]) - (bar_width / 2), bottom_index_wise_means["SP_L"], width=bar_width, color="cyan", label="Mean Sub-Plan Lengths")
                axes[1, 2].bar(numpy.array(bottom_index_wise_means["INDEX"]) + (bar_width / 2), bottom_index_wise_means["INTER_Q"], width=bar_width, color="magenta", label="Mean Interleaving Quantity")
                axes[1, 2].set_title("Bottom-Level Sub-Plan Lengths")
                axes[1, 2].set_ylabel("Length")
                axes[1, 2].set_xlabel("Sub-goal Stage Index")
                axes[1, 2].legend(prop={"size" : "xx-small"})
                
                ## Achieved sub-goal stages against search length
                axes[2, 0].plot(steps, bottom_step_wise_means["C_TACHSGOALS"], "g", label="Achieved Sub-goal Stages")
                axes[2, 0].bar(bottom_index_wise_means["ACH_AT"], max(bottom_step_wise_means["C_TACHSGOALS"]), width=0.20, color="cyan", label="Mean Achievement Steps")
                if namespace.experimental_runs > 1:
                    axes[2, 0].plot(steps, bottom_step_wise_means["C_TACHSGOALS"] + bottom_step_wise_std["C_TACHSGOALS"], "--g")
                    axes[2, 0].plot(steps, bottom_step_wise_means["C_TACHSGOALS"] - bottom_step_wise_std["C_TACHSGOALS"], "--g")
                axes[2, 0].set_title("Bottom-Level Sub-Goal Achievement")
                axes[2, 0].set_ylabel("Total")
                axes[2, 0].set_xlabel("Plan Length")
                axes[2, 0].legend(prop={"size" : "xx-small"})
                
                ## Plan expansion against search length
                axes[2, 1].plot(steps, bottom_step_wise_means["C_CP_EF_L"], "g", label="Length Factor")
                axes[2, 1].plot(steps, bottom_step_wise_means["C_CP_EF_A"], "y", label="Action Factor")
                axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_ED_L"], "b", label="Length Deviation")
                axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_ED_A"], "c", label="Action Deviation")
                axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_EB_L"], "r", label="Length Balance")
                axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_EB_A"], "m", label="Action Balance")
                if namespace.experimental_runs > 1:
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_CP_EF_L"] + bottom_step_wise_std["C_CP_EF_L"], "--g")
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_CP_EF_L"] - bottom_step_wise_std["C_CP_EF_L"], "--g")
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_CP_EF_A"] + bottom_step_wise_std["C_CP_EF_A"], "--y")
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_CP_EF_A"] - bottom_step_wise_std["C_CP_EF_A"], "--y")
                    
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_ED_L"] + bottom_step_wise_std["C_SP_ED_L"], "--b")
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_ED_L"] - bottom_step_wise_std["C_SP_ED_L"], "--b")
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_ED_A"] + bottom_step_wise_std["C_SP_ED_A"], "--c")
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_ED_A"] - bottom_step_wise_std["C_SP_ED_A"], "--c")
                    
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_EB_L"] + bottom_step_wise_std["C_SP_EB_L"], "--r")
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_EB_L"] - bottom_step_wise_std["C_SP_EB_L"], "--r")
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_EB_A"] + bottom_step_wise_std["C_SP_EB_A"], "--m")
                    axes[2, 1].plot(steps, bottom_step_wise_means["C_SP_EB_A"] - bottom_step_wise_std["C_SP_EB_A"], "--m")
                axes[2, 1].set_title("Bottom-Level Refinement Expansion")
                axes[2, 1].set_ylabel("Accumulating Expansion")
                axes[2, 1].set_xlabel("Plan Length")
                axes[2, 1].legend(prop={"size" : "xx-small"})
                
                ## Partial-problem balancing
                set_bars(7)
                axes[2, 2].bar(al_range - (bar_width * 3.0), means["PR_T"], bar_width, label="Total Problems")
                axes[2, 2].bar(al_range - (bar_width * 2.0), means["PR_TS_MEAN"], bar_width, label="Mean Size")
                axes[2, 2].bar(al_range - (bar_width * 1.0), means["PR_TS_STD"], bar_width, label="Stdev Size")
                axes[2, 2].bar(al_range, means["PP_LE_MEAN"], bar_width, label="Mean Refinement Length")
                axes[2, 2].bar(al_range + (bar_width * 1.0), means["PP_LE_STD"], bar_width, label="Stdev Refinement Length")
                axes[2, 2].bar(al_range + (bar_width * 2.0), means["DIV_INDEX_MAE"], bar_width, label="Divisions Index Spread")
                axes[2, 2].bar(al_range + (bar_width * 3.0), means["DIV_STEP_MAE"], bar_width, label="Divisions Step Spread")
                axes[2, 2].set_title("Problem Balance")
                axes[2, 2].set_ylabel("Total")
                axes[2, 2].set_xlabel("Abstraction Level")
                axes[2, 2].set_xticks(al_range)
                axes[2, 2].set_xticklabels(al_labels)
                axes[2, 2].legend(prop={"size" : "xx-small"})
            
            ## Add a little more space vertically to make room for plot titles
            figure.subplots_adjust(hspace=0.4)
            
            ## Save the plots
            if (figure_file := namespace.figure_file) is not None:
                if namespace.config_file_naming:
                    figure_file = figure_file.split(".png")[0] + f"_{config_file_name}" + ".png"
                _Launcher_logger.info(f"Saving results to figure file: {figure_file}")
                pyplot.pause(1) # This is needed for some strange reason
                figure.set_size_inches(16, 9.8) # Set to a fixed size so the output looks clean
                pyplot.savefig(figure_file, bbox_inches="tight", dpi=120)
            
            ## Display the plots maximised on the screen
            if namespace.display_figure:
                figure_manager: FigureManagerBase = pyplot.get_current_fig_manager()
                figure_manager.window.showMaximized()
                pyplot.show()
    
    ## Return a clean exit
    return 0

def __setup() -> argparse.Namespace:
    ## Record launch time
    launch_time: datetime.datetime = datetime.datetime.now()
    output_file_append: str = launch_time.strftime('%Y-%m-%d_%H-%M-%S')
    
    ## Declare command line arguments parser
    parser = argparse.ArgumentParser(description="Launcher script for generating plans and running experiments with ASH. "
                                                 "Copyright (C) 2021 Oliver Michael Kamperis.")
    
    ## Inner function for creating options for Boolean arguments
    def bool_options(default: Optional[bool],
                     const: Optional[bool] = True,
                     add_none: bool = False
                     ) -> dict[str, Any]:
        """
        Create a Boolean argument.
        
        Parameters
        ----------
        `default: Optional[bool]` - The default argument value used when the argument is not given by the user.
        
        `const: Optional[bool] = True` - The standard argument value used when the argument is given without a value.
        
        `add_none: bool = False` - Whether to allow None to be valid arguments value.
        
        Returns
        -------
        `dict[str, Any]` - A dictionary of options for creating a Boolean argument.
        """
        choices: list[Optional[bool]] = [True, False]
        if add_none: choices.append(None)
        return dict(nargs="?", choices=choices, default=default, const=const,
                    type=lambda input_: None if input_ == "None" else input_ == "True")
    
    ## Inner function for processing optional number argument types
    def optional_number(value: str) -> Optional[Number]:
        if not value or value == "None":
            return None
        try:
            if '.' in value:
                return float(value)
            else: return int(value)
        except ValueError as error:
            print(f"Cannot parse {value} as a float or int: {error}")
            raise error
    
    ## Inner function for processing optional integer argument types
    def optional_int(value: str) -> Optional[int]:
        if not value or value == "None":
            return None
        try:
            return int(value)
        except ValueError as error:
            print(f"Cannot parse {value} as a float or int: {error}")
            raise error
    
    ## Inner function for processing optional string arguments types
    def optional_str(value: str) -> Optional[str]:
        if not value or value == "None":
            return None
        return value
    
    ## Special action for storing arguments of parameters that can have a different values for each abstraction level in the hierarchy
    class StoreHierarchicalArguments(argparse.Action):
        def __call__(self,
                     parser: argparse.ArgumentParser,
                     namespace: argparse.Namespace,
                     values: Sequence[str],
                     option_string: Optional[str] = None):
            
            _values: Union[Number, dict[int, Number]] = {}
            
            try:
                if len(values) == 1 and '=' not in values[0]:
                    if ',' in values[0]:
                        _values = tuple(optional_number(v) for v in values[0].split(','))
                    else: _values = optional_number(values[0])
                else:
                    for key_value in values:
                        key, value = key_value.split('=', 1)
                        if ',' in value:
                            _values[int(key)] = tuple(optional_number(v) for v in value.split(','))
                        else: _values[int(key)] = optional_number(value)
            except ValueError as error:
                print(f"Error during parsing hierarchical argument '{option_string}': {error}")
                raise error
            
            setattr(namespace, self.dest, _values)
    
    ## Input files
    parser.add_argument("files", nargs="*", type=str,
                        help="a list of planning domain and problem files to load, at least one must be given")
    parser.add_argument("--load_schema", default=None, type=str,
                        help="specify a file to load a refinement schema from to generate problem spaces and check for dependencies between partial problems")
    
    ## Output files
    parser.add_argument("-cfn", "--config_file_naming", action="store_true",
                        help="name output files based on the configuration file name, format ./")
    parser.add_argument("-pf", "--plan_file", default=f"./solutions/plans/ASH_Plan_{output_file_append}.txt", type=str,
                        help="specify a custom file to save the generated plans to during standard operation, "
                             f"by default ./solutions/plans/ASH_Plan_{output_file_append}.txt")
    parser.add_argument("-lf", "--log_file", default=f"./logs/ASH_Log_{output_file_append}.log", type=str,
                        help=f"specify a custom file to save the log file to, by default ./logs/ASH_Log_{output_file_append}.log")
    parser.add_argument("-xf", "--excel_file", nargs="?", default=None, const=f"./experiments/results/ASH_Excel_{output_file_append}.xlsx", type=optional_str,
                        help="output experimental results to an excel (.xlsx) file, optionally specify a file name, "
                             f"as standard ./experiments/results/ASH_Excel_{output_file_append}.xlsx")
    parser.add_argument("-df", "--data_file", nargs="?", default=None, const=f"./experiments/results/ASH_Data_{output_file_append}.dat", type=optional_str,
                        help="output experimental results to a Delimiter-Seperated Values (DSV) (.dat) file, "
                             f"optionally specify a file name, as standard ./experiments/results/ASH_Data_{output_file_append}.dat")
    parser.add_argument("-df_ds", "--data_sep", default=" ", type=str,
                        help="string specifying the delimiter between fields (values) of the output data file, by default ' '")
    parser.add_argument("-df_de", "--data_end", default="\n", type=str,
                        help="string specifying the delimiter between records (rows) of the output data file, by default '\\n'")
    parser.add_argument("-ff", "--figure_file", nargs="?", default=None, const=f"./experiments/results/ASH_Figure_{output_file_append}.png", type=optional_str,
                        help="output experimental results displayed as a set of graphs on a figure to Portable Network Graphics (PNG) (.png) file, "
                             f"optionally specify a file name, as standard ./experiments/results/ASH_Figure_{output_file_append}.dat")
    parser.add_argument("-sf", "--save_schema", default=f"./solutions/schemas/ASH_Schema_{output_file_append}.txt", type=str,
                        help=f"specify a file name to save a schema to, by default ./solutions/schemas/ASH_Schema_{output_file_append}.txt")
    parser.add_argument("--schema_level", default=1, type=int,
                        help="specify an abstraction level to make a schema for, by default 1 (the ground level)")
    
    ## Header options
    parser.add_argument("-v", "--version", action="version", version=f"ASH - The ASP based Hierarchical Conformance Refinement Planner :: {_ASH_VERSION}")
    parser.add_argument("-w", "--warranty", action="store_true", help="show the program's warranty information on launch")
    parser.add_argument("-c", "--conditions", action="store_true", help="show the program's warranty conditions on launch")
    
    ## Launcher options
    parser.add_argument("-ao", "--ash_output", choices=["verbose", "standard", "simple", "experiment"], default="standard", type=str,
                        help="the output verbosity of ASH; 'verbose' (full details of planned actions and the achievement sub-goal stages), "
                        "'standard' (only planned actions), 'simple' (tqdm progress bars, plans and results), "
                        "or 'experiment' (experiment tracking tqdm progress bars and results only), by default 'standard'")
    parser.add_argument("-co", "--clingo_output", **bool_options(default=False),
                        help="whether to enable output from Clingo, by default False")
    parser.add_argument("-cl", "--console_logging", choices=["DEBUG", "INFO", "WARN"], default="INFO",
                        help="the logging level to print to the console; 'DEBUG', 'INFO', or 'WARN', by default 'INFO'")
    parser.add_argument("--disable_logging", action="store_true",
                        help="disable all logging, removing all overhead on producing logs")
    parser.add_argument("-dpos", "--disable_pause_on_start", action="store_true", default=False,
                        help="disable the pause for user input when the planner starts")
    parser.add_argument("-disp_fig", "--display_figure", **bool_options(default=True),
                        help="whether to display experimental results in a simple set of graphs upon completion of all experimental runs")
    
    ## Experimentation options
    parser.add_argument("-op", "--operation", choices=["test", "experiment", "find-problem-inconsistencies"], default="test", type=str,
                        help="the operating mode of ASH; 'test' (generate a single hierarchical refinement diagram and save the generated plans), "
                             "'experiment' (run the planner several times, generating a fresh plan each time, and gathering aggregate experimental statistics), "
                             "'find-problem-inconsistencies' (only generate the initial states and final goals, then return), by default 'standard'")
    parser.add_argument("-er", "--experimental_runs", default=1, type=int,
                        help="integer specifying number of experimental runs, by default 1")
    parser.add_argument("-ir", "--initial_runs", nargs="?", default=0, const=1, type=int,
                        help="integer specifying number of initial 'dry' runs before experimental results are recorded, by default 0, as standard 1")
    parser.add_argument("-opti", "--optimum", nargs="+", default=None, action=StoreHierarchicalArguments, type=str, metavar="level1=value1 level_i=value_i [...] level_n=value_n",
                        help="the classical optimum for each level in the abstraction hierarchy, by default None (takes the optimum as the best quality plan over all experimental runs)")
    
    ## ASP solver options
    parser.add_argument("-th", "--threads", default=os.cpu_count(), type=int,
                        help=f"integer specifying number of solver threads, by default {os.cpu_count()} (your cpu count)")
    parser.add_argument("-tl", "--planning_time_limit", nargs="+", default=3600, action=StoreHierarchicalArguments, type=str, metavar="value | level1=value1 level_i=value_i [...] level_n=value_n",
                        help="maximum cumulative planning time limit in seconds, given as either a single value (used at all abstraction levels) or a dictionary of level-value pairs, by default 3600 (one hour)")
    parser.add_argument("-ll", "--search_length_limit", nargs="+", default=None, action=StoreHierarchicalArguments, type=str, metavar="value | level1=value1 level_i=value_i [...] level_n=value_n",
                        help="maximum search length limit, given as either a single value (used at all abstraction levels) or a dictionary of level-value pairs, by default None")
    
    ## ASH general options
    parser.add_argument("-m", "--planning_mode", choices=["hcr", "scr", "mcl", "hcl"], default="hcr", type=str,
                        help="what planning mode to use; 'hcr' (hierarchical conformance refinement), 'scr' (schema conformance refinement), 'mcl' (monolevel classical), or 'hcl' (hierarchical classical), by default 'hcr'")
    parser.add_argument("-t", "--conformance_type", choices=["sequential", "simultaneous"], default="sequential", type=str,
                        help="what conformance constraint type to use; 'sequential' (sub-goals of a stage can be achieved by any of its producing abstract action's children), "
                             "'simultaneous' (all sub-goals of a stage must be achieved by its producing abstract action's matching child), by default 'sequential'")
    parser.add_argument("-conc", "--enable_concurrency", **bool_options(default=False),
                        help="whether to enable action concurrency, by default False, as standard True")
    parser.add_argument("-mini_act", "--minimise_actions", **bool_options(default=None, add_none=True),
                        help="whether to enable the optimisation statement that minimises the number of actions in generated plans, "
                             "if None then the planner decides (chooses True if concurrency is enabled else False), by default None, as standard True")
    parser.add_argument("-yield", "--sequential_yielding", **bool_options(default=True),
                        help="whether to sequentially yield sub-plans as they are found by the incremental solver, this is more expensive than one-shot "
                             "(complete or divided) planning but allows refinement planning progress to be observed, by default False, as standard True (must be enabled to use reactive division strategies)")
    parser.add_argument("-detect_int", "--detect_interleaving", **bool_options(default=False),
                        help="whether to detect interleaving during sequential yielding, this is an expensive operation, by default False")
    parser.add_argument("-min_bound", "--minimum_search_length_bound", **bool_options(default=True),
                        help="whether to use the minimum search length bound to reduce search time at low search lengths during one-shot planning, by default True (disabled in sequential yield mode)")
    parser.add_argument("-obs", "--make_observable", **bool_options(default=False),
                        help="whether to make the plan that minimally achieves the previous in sequence sub-goal stage observable in sequential yield mode, "
                             "this is an expensive operation and ASH currently only uses this for debugging purposes, by default False, as standard True")
    
    ## Hierarchical planning options
    parser.add_argument("-top", "--top_level", default=None, type=optional_int,
                        help="override the top level used in hierarchical planning, if not given or None then the hierarchical system law definition's top-level is used, by default None")
    parser.add_argument("-bot", "--bottom_level", default=1, type=int,
                        help="override the bottom level used in hierarchical planning, if not given then the bottom level is the ground level, by default 1 (the ground level)")
    parser.add_argument("-plc", "--pause_on_level_change", **bool_options(default=False),
                        help="whether to pause execution of the planner when the current planning level changes in hierarchical planning, by default False, as standard True")
    parser.add_argument("-pic", "--pause_on_increment_change", **bool_options(default=False),
                        help="whether to pause execution of the planner when the current planning increment changes in online planning, by default False, as standard True")
    
    ## Online planning options
    parser.add_argument("-method", "--online_method", choices=["ground-first", "complete-first", "hybrid"], default="ground-first", type=str,
                        help="what divided planning method to use; ground-first (solve only the initial partial problems first, propagating directly down to the ground level "
                             "as fast as possible, afterwards solve the lowest unsolved partial problem until the ground level is complete), 'complete-first' (solve all "
                             "partial problems at each level before moving to the next level, successively completing each level until the ground is reached), "
                             "or 'hybrid' (uses a mix of the prior, solve only the initial partial problems first, propagating directly down to the ground level as fast as possible, then successively complete each level), by default 'ground-first'")
    parser.add_argument("-strat", "--division_strategy", choices=["none", "basic", "steady", "hasty", "jumpy", "relentless", "impetuous", "rapid-basic", "rapid-steady", "rapid-hasty", "rapid-jumpy"], default="none", type=str,
                        help="the division strategy to use; 'none', '(rapid-)basic', '(rapid-)steady', '(rapid-)hasty', '(rapid-)jumpy', 'relentless', 'impetuous', by default 'none'")
    parser.add_argument("-bound", "--division_strategy_bounds", nargs="+", default=None, action=StoreHierarchicalArguments, type=str, metavar="value | level1=value1 level_i=value_i [...] level_n=value_n",
                        help="the bound used on the division strategy, this a maximum or minimum bound on the size, complexity, or planning time of the partial "
                             "problems as defined by the nature of the strategy itself, given as either a tuple of values (used at all abstraction levels) "
                             "or a dictionary of level-tuple pairs to set different bounds for each level, if None then the strategy takes its specific default bound, by default None")
    parser.add_argument("--bound_type", choices=["search_length", "incremental_time", "differential_time", "integral_time", "cumulative_time"], default="incremental_time", type=str,
                        help="the type of bound used by reactive or adaptive strategies for their modifiable bound; 'search_length' (the current planning search length), 'incremental_time' (the total incremental planning time (seconds/step), averaged over the moving range), "
                             "'differential_time' (the rate of change (increase) in the total incremental planning time (seconds/step/step), averaged over the moving range), 'integral_time' (the sum of the total incremental planning times (seconds) over the moving range), "
                             "'cumulative_time' (the sum of all total incremental planning times (seconds) since the last reactive division, essentially an alias for an integral bound type with no range bound), by default 'incremental'")
    parser.add_argument("-save", "--save_grounding", **bool_options(default=False),
                        help="whether to save the ASP program grounding between divided planning increments, by default False, as standard True")
    parser.add_argument("-horizon", "--backwards_horizon", nargs="+", default=0, action=StoreHierarchicalArguments, type=str, metavar="value | level1=value1 level_i=value_i [...] level_n=value_n",
                        help="the backwards horizon used when making continuous reactive divisions, continuous divisions cannot be commited until the horizon is reached, by default 0")
    parser.add_argument("-preempt", "--preemptive_division", **bool_options(default=False),
                        help="whether to use preemptive reactive division, this allows a reactive division to be commited on any child step of the current sub-goal stage, "
                             "otherwise if disabled reactive divisions can only be commited on a minimal matching child step (the step at which the current sub-goal stage was originally minimally achieved), by default False")
    parser.add_argument("-interrupt", "--interrupting_division", **bool_options(default=False),
                        help="whether to commit an interruting division when a reactive division is made, interrupting divisions are problem shifting, by default False")
    parser.add_argument("-average", "--moving_average", default=1, type=optional_int,
                        help="the number of previous incremental times to use to calculate a moving average when using an incremental time bound, "
                             "if a differential search time gradient bound is used then this is the amount of steps used to calculate the differential, "
                             "by default 1 (disables moving average mode), as standard None (uses a minimum number of steps to allow a gradient calculation)")
    parser.add_argument("-lblend", "--left_blend_quantities", nargs="+", default=0, action=StoreHierarchicalArguments, type=str, metavar="value | level1=value1 level_i=value_i [...] level_n=value_n",
                        help="the number of sub-goal stages to blend each partial problem with the previous adjacent partial problem in divided planning, "
                             "given as either a single value (used at all abstraction levels) or a dictionary of level-value pairs, by default 0 (disables left blending)")
    parser.add_argument("-rblend", "--right_blend_quantities", nargs="+", default=0, action=StoreHierarchicalArguments, type=str, metavar="value | level1=value1 level_i=value_i [...] level_n=value_n",
                        help="the number of sub-goal stages to blend each partial problem with the following adjacent partial problem in divided planning, "
                             "given as either a single value (used at all abstraction levels) or a dictionary of level-value pairs, by default 0 (disables right blending)")
    parser.add_argument("-avoid", "--avoid_refining_sgoals_marked_for_blending", **bool_options(default=False),
                        help="whether to avoid refining sub-goal stages marked for blending at the previous level, this is done by prohibiting proactively generated division scenarios "
                             "from dividing sub-goal stage inside the left blend of the right point of the current (partial) refinement planning problem, this can save overall planning time "
                             "by avoiding refining sub-goal stages that are guaranteed to be revised by blending at the higher abstraction levels but can reduce plan quality at the lower abstraction levels, "
                             "the more abstraction levels there are the more prevalent this trade-off becomes, by default False, as standard True")
    parser.add_argument("-inde_tasks", "--treat_tasks_as_independent", **bool_options(default=False),
                        help="overrides the division strategy used in tasking models to treat refining each individual task as an independent sub-problem at the next level, by default True")
    parser.add_argument("-order_tasks", "--divide_tasks_on_final_goal_intermediate_achievement_ordering", **bool_options(default=False),
                        help="overrides the division strategy used in tasking models to make divisions only upon the intermediate achievement of a final-goal literal, by default True (requires final-goal intermediate ordering preferences)")
    
    ## Online optimisation options
    parser.add_argument("-order_fgoals", "--final_goal_intermediate_achievement_ordering_preferences", **bool_options(default=None, const=True, add_none=True),
                        help="whether to enable optimisation of ordering preferences over the intermediate achievement of task-critical final-goal literals in tasking models, "
                             "if None then the planner decides (chooses True if a planning problem has a tasking model, False otherwise), by default None, as standard True")
    parser.add_argument("-preempt_pos_fgoals", "--positive_final_goal_preemptive_achievement", **bool_options(default=None, const=True, add_none=True),
                        help="whether to enable positive final-goal preemptive achievement, this prefers choosing actions whose effects preemptively achieve positive final-"
                             "goal literals, when there is an arbitrary choice available in non-final partial problems, by default None, as standard True")
    parser.add_argument("-preempt_neg_fgoals", "--negative_final_goal_preemptive_achievement", **bool_options(default=None, const=True, add_none=True),
                        help="whether to enable negative final-goal preemptive achievement, this prefers choosing actions whose effects preemptively achieve negative final-"
                             "goal literals, when there is an arbitrary choice available in non-final partial problems, by default None, as standard True")
    parser.add_argument("-preempt_mode", "--final_goal_preemptive_achievement_mode", choices=["heuristic", "optimise"], default="heuristic", type=str,
                        help="The final-goal preemptive achievement enforcement mode; 'heuristic' (preemptive achievement is applied as a domain heuristic to the ASP solver, affecting solving at all search steps in the planning progression),"
                             "'optimise' (preemptive achievement is applied as a model optimisation process, affecting the generation of optimal answer sets at the end of search), by default 'heuristic'")
    
    ## Search for a configuration file in the argument list
    options: Optional[list[str]] = None
    global config_file
    config_file = None
    for index, arg in enumerate(sys.argv):
        if "--config" in arg:
            options = []
            
            ## Split the options in the configuration file
            config_file = arg.split("=")[1]
            with open(config_file, "r") as file_reader:
                in_desc: bool = False
                for line in file_reader.readlines():
                    if "description" in line.lower():
                        in_desc = True
                    if "options" in line.lower():
                        in_desc = False
                        continue
                    if not in_desc:
                        options.extend(line.split())
            
            options.extend(sys.argv[1 : index])
            options.extend(sys.argv[index + 1 : len(sys.argv)])
            break
    
    ## Parse the arguments and obtain the namespace
    namespace: argparse.Namespace = parser.parse_args(options)
    
    ## Record the name of the configuration file in the global scope
    global config_file_name
    config_file_name = ""
    if config_file is not None:
        if "\\" in config_file:
            config_file_name = config_file.split(".config")[0].split("\\")[-1]
        elif "/" in config_file:
            config_file_name = config_file.split(".config")[0].split("/")[-1]
        else: config_file_name = config_file.split(".config")[0]
    
    ## Setup the logger
    if not namespace.disable_logging:
        
        ## Determine log file name
        log_file_name: str = namespace.log_file
        if namespace.config_file_naming:
            log_file_name = log_file_name.split(".log")[0] + f"_{config_file_name}" + ".log"
        
        ## Use a rotating log file handler, that generates up to 10 log files, each of which are at most 95 MBs in size
        logging.basicConfig(handlers=[logging.handlers.RotatingFileHandler(log_file_name, mode="w",
                                                                           maxBytes=(95 * (1024 ** 2)),
                                                                           backupCount=10, encoding="utf-8")],
                            format="[%(asctime)s] %(levelname)-4s :: %(name)-8s >> %(message)s\n",
                            datefmt="%d-%m-%Y_%H-%M-%S",
                            level=logging.DEBUG)
        
        ## Add the title and warranty conditions to the log file
        _Launcher_logger.debug(center_text(_ASH_TITLE, prefix_blank_line=True, framing_width=116, framing_char='#'))
        _Launcher_logger.debug(center_text(_ASH_WARRANTY, prefix_blank_line=True, framing_width=80))
        _Launcher_logger.debug(center_text(_ASH_CONDITIONS, prefix_blank_line=True, framing_width=80))
        
        ## Log the command line arguments for debugging purposes
        _Launcher_logger.debug("Command line arguments:\n" + "\n".join(repr(item) for item in sys.argv[1:]))
        if options:
            _Launcher_logger.debug(f"Configuration file loaded: {config_file}")
            _Launcher_logger.debug("Configuration file arguments:\n" + "\n".join(repr(item) for item in options))
        _Launcher_logger.debug("Parsed command line arguments:\n" + "\n".join(repr(item) for item in namespace.__dict__.items()))
        if namespace.config_file_naming:
            _Launcher_logger.debug(f"Configuration file output file naming enabled: {config_file_name}")
        
        ## Setup the console stream
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.__dict__[namespace.console_logging])
        console_handler.setFormatter(logging.Formatter("%(levelname)-4s :: %(name)-12s >> %(message)s\n"))
        logging.getLogger("").addHandler(console_handler)
    
    return namespace

## Launch ASH and enter main method
if __name__ == "__main__":
    start_real_time: float = time.perf_counter()
    start_process_time: float = time.process_time()
    exit_code: int = -1
    
    try:
        exit_code = __main()
    except BaseException as exception:
        _Launcher_logger.exception("Exception during main:\n", exc_info=exception)
    
    _Launcher_logger.info(f"Overall time: Real = {time.perf_counter() - start_real_time}, Process = {time.process_time() - start_process_time}")
    _Launcher_logger.info(f"Exiting with code {exit_code}")
    sys.exit(exit_code)
