%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ASH - The ASP based Hierarchical Abstraction Planner for Robots       %%
%% Copyright (C)  2021  Oliver Michael Kamperis                          %%
%% Email: o.m.kamperis@gmail.com                                         %%
%%                                                                       %%
%% This program is free software: you can redistribute it and/or modify  %%
%% it under the terms of the GNU General Public License as published by  %%
%% the Free Software Foundation, either version 3 of the License, or     %%
%% any later version.                                                    %%
%%                                                                       %%
%% This program is distributed in the hope that it will be useful,       %%
%% but WITHOUT ANY WARRANTY; without even the implied warranty of        %%
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          %%
%% GNU General Public License for more details.                          %%
%%                                                                       %%
%% You should have received a copy of the GNU General Public License     %%
%% along with this program. If not, see <https://www.gnu.org/licenses/>. %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Abstraction levels module - Generate the available abstraction levels
%**% #program abstraction_levels(planning_level, mode).
%%%%
%%%% Parameters
%%%% ----------
%%%%    planning_level: int - The abstraction level at which to plan.
%%%%    mode: {classical | refinement | hierarchical} - The state representation mode named according to the type of planning they are used for, as described below.
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Define abstraction levels
al(1..abstraction_max).

%% Current planning level
pl(AL) :- AL = planning_level, al(AL).

%% Define abstraction levels at which the state should be represented at:
%%      - Classical planning: Represent the state at only the current planning level,
%%      - Refinement planning: The state is representated at the current planning level and previous adjacent level to allow reasoning about conformance,
%%      - Hierarchical representation: The state is represented at the current planning level and all higher levels to allow generation of the initial state and final-goals.
sl(AL) :- AL = planning_level, mode = classical, al(AL).
sl(AL) :- AL >= planning_level, AL <= planning_level + 1, mode = refinement, al(AL).
sl(AL) :- AL >= planning_level, AL <= abstraction_max, mode = hierarchical, al(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Entity instance relations module - Generate the instance and ancestry relations from the domain specific sort declarations and entities
%**% #program instance_module.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Entity instances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Define boolean values
boolean(true;false).

%% All entities are instances of their own class at the same abstraction level
insta_of(AL, CLS, EN) :- entity(CLS, EN), class(AL, CLS), al(AL).

%% And are instances of all their super-classes at the same abstraction level
insta_of(AL_3, CLS_1, EN) :- insta_of(AL_2, CLS_2, EN), super_class(CLS_1, CLS_2),
                             entity(CLS_3, EN), class(AL_3, CLS_3), class(AL_1, CLS_1),
                             AL_1 >= AL_2, al(AL_1), al(AL_2), al(AL_3).

%% As well as the next lower abstraction level if its type is not overridden by one of its descendants of the same type
insta_of(AL - 1, CLS, EN) :- insta_of(AL, CLS, EN), not overridden(AL - 1, CLS, EN), class(_, CLS), al(AL), al(AL - 1).

%% The type of an entity is overriden at a given abstraction level if there is any entity that exists at that level that overrides that type
overridden(AL - 1, CLS, EN_1) :- overridden_by(AL - 1, CLS, EN_1, EN_2),
                                 insta_of(AL, CLS, EN_1), insta_of(AL - 1, CLS, EN_2),
                                 class(_, CLS), al(AL), al(AL - 1).

%% The class type CLS_3 of an entity EN_1 at abstraction level AL is overridden by an entity EN_2 of the same type that is discovered at abstraction level AL - 1 if;
%%      - There is a type override such that instances of class CLS_2 override the class type CLS_3 of instances of class CLS_1,
%%      - Entities EN_1 and EN_2 are instances of class CLS_1 and CLS_2 at abstraction levels AL and AL - 1 respectively,
%%      - Entities EN_1 and EN_2 are both instances of class CLS_3 at abstraction levels AL and AL - 1 respectively,
%%      - Entity EN_2 is a descendant of entity EN_1 at abstraction level AL - 1.
overridden_by(AL - 1, CLS_3, EN_1, EN_2) :- insta_of(AL, CLS_1, EN_1), insta_of(AL - 1, CLS_2, EN_2),
                                            insta_of(AL, CLS_3, EN_1), insta_of(AL - 1, CLS_3, EN_2),
                                            desce_of(AL - 1, EN_1, EN_2), override_class(CLS_1, CLS_2, CLS_3),
                                            class(_, CLS_1), class(_, CLS_2), class(_, CLS_3), al(AL), al(AL - 1).

%% All states are states of all entities of the same class type at the same abstraction level
state_of(AL, EN, S) :- state(AL, CLS, S), insta_of(AL, CLS, EN), al(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Ancestry relations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% An entity is a child of another at a given abstraction level if there is a ancestry relation between them
child_of(AL, EN, DESCE) :- ancestry_relation(EN, DESCE), insta_of(AL, _, EN), insta_of(AL, _, DESCE), al(AL).

%% An entity is a descendant of another if either;
%%      - The entity is the child of the other entity,
%%      - By transitivity, if the entity is a descendant of some different entity which is itself a descendant of the other entity.
desce_of(AL, EN, DESCE) :- child_of(AL, EN, DESCE), insta_of(AL, _, EN), insta_of(AL, _, DESCE), al(AL).
desce_of(AL, EN, DESCE_1) :- desce_of(AL, DESCE_2, DESCE_1), desce_of(AL, EN, DESCE_2),
                             insta_of(AL, _, EN), insta_of(AL, _, DESCE_1), insta_of(AL, _, DESCE_2), al(AL).

%% An entity has descendants at a given abstraction level if any of its descendants exist at that abstraction level
has_desce(AL, EN) :- desce_of(AL, EN, DESCE), insta_of(AL, _, EN), insta_of(AL, _, DESCE), al(AL).

%% An entity has descendants of a certain type at a given abstraction level if any of its descendants are an instance of that class type and exist at that abstraction level
has_desce(AL, EN, CLS) :- desce_of(AL, EN, DESCE), insta_of(AL, _, EN), insta_of(AL, CLS, DESCE), al(AL).

%% Entities are siblings if they are both a child of the same entity at a given abstraction level
siblings(AL, DESCE_1, DESCE_2) :- child_of(AL, EN, DESCE_1), child_of(AL, EN, DESCE_2), DESCE_1 != DESCE_2,
                                  insta_of(AL, _, EN), insta_of(AL, _, DESCE_1), insta_of(AL, _, DESCE_2), al(AL).

%% Siblingship is reversible
siblings(AL, DESCE_2, DESCE_1) :- siblings(AL, DESCE_1, DESCE_2), insta_of(AL, _, DESCE_1), insta_of(AL, _, DESCE_2), al(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% State Module - Ensure the state is valid and remains consistent through time
%**% #program state_module(step, start).
%%%%
%%%% Parameters
%%%% ----------
%%%%    step: int - A natural number defining the current planning horizon (equivalent to the search length).
%%%%    start: int - A natural number defining the step to begin planning from.
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Law of Awareness - All inertial fluents can take any value from their range in the initial state
{ holds(AL, F, V, step) } :- fluent(AL, inertial, F, V), sl(AL), step = start.

%% Law of Continuity - All Fluents must have unique values to ensure the state is complete
:- not { holds(AL, F, V, step) : fluent(AL, _, F, V) } = 1, fluent(AL, _, F, _), sl(AL).

%% Law of Inertia - Propagate the value of fluents through time unless there is reason to believe their value has been changed
holds(AL, F, V, step) :- holds(AL, F, V, step - 1), not not holds(AL, F, V, step), fluent(AL, inertial, F, V), sl(AL), step > start.

%% Closed world assumption for defined fluents with a boolean range - If there is no reason to believe a defined fluent is true then it must be false
holds(AL, F, false, step) :- not holds(AL, F, true, step), fluent(AL, defined, F, B), boolean(B), sl(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plan Module - Generate legal plans
%**% #program plan_module(step, start, concurrency, achieve_fgoals).
%%%%
%%%% Parameters
%%%% ----------
%%%%    step: int - A natural number defining the current planning horizon (equivalent to the search length).
%%%%    start: int - A natural number defining the step to begin planning from.
%%%%    concurrency: bool - A Boolean, True to enable action concurrency (set of at least one action is planned per step), False to disable (exactly one action is planned per step).
%%%%    achieve_fgoals: bool - A Boolean, True to enforce achievement of the final-goal (used for final single-return problems), False otherwise.
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Rules for plan generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generator rule for actions at the current planning level whilst either;
%%      - at least one unsatisfied goal existed at the previous step,
%%      - the search space is being generated.
{ occurs(AL, ROBOT, A, step) : action(AL, _, _, ROBOT, A) } :- pl(AL), step > start.

%% Rules defining whether actions are taken on each step
occurs(AL, ROBOT, step) :- occurs(AL, ROBOT, A, step), action(AL, _, _, ROBOT, A), pl(AL), step > start.
occurs(AL, step) :- occurs(AL, ROBOT, step), insta_of(AL, robot, ROBOT), pl(AL), step > start.

%% Activity constraint - Prevent the robots from being idle whilst the plan is incomplete;
%%      - Plan exactly one action on each step in sequential planning,
%%      - Plan at least one action on each step in concurrent planning.
:- not { occurs(AL, R, A, step) : action(AL, _, _, ROBOT, A) } = 1, pl(AL), step > start, concurrency = false.
:- not { occurs(AL, R, A, step) : action(AL, _, _, ROBOT, A) } >= 1, pl(AL), step > start, concurrency = true.

%% Actions planned at the current planning level have their effects applied at the current step
holds(AL, F, V, step) :- occurs(AL, ROBOT, A, step), effect(AL, ROBOT, A, F, V, step),
                         action(AL, _, _, ROBOT, A), fluent(AL, inertial, F, V), pl(AL), step > start.
holds(AL, F, V, step) :- occurs(AL, ROBOT, A, step), effect(AL, ROBOT, A, F, V),
                         action(AL, _, _, ROBOT, A), fluent(AL, inertial, F, V), pl(AL), step > start.

%% If the positive preconditions of an action are not satisfied in a given state then the action is impossible
:- occurs(AL, ROBOT, A, step), precond(AL, ROBOT, A, F, V, true, step), not holds(AL, F, V, step - 1),
   action(AL, _, _, ROBOT, A), fluent(AL, inertial, F, V), pl(AL), step > start.
:- occurs(AL, ROBOT, A, step), precond(AL, ROBOT, A, F, V, true), not holds(AL, F, V, step - 1),
   action(AL, _, _, ROBOT, A), fluent(AL, inertial, F, V), pl(AL), step > start.

%% If the negative preconditions of an action are not satisfied in a given state then the action is impossible
:- occurs(AL, ROBOT, A, step), precond(AL, ROBOT, A, F, V, false, step), holds(AL, F, V, step - 1),
   action(AL, _, _, ROBOT, A), fluent(AL, inertial, F, V), pl(AL), step > start.
:- occurs(AL, ROBOT, A, step), precond(AL, ROBOT, A, F, V, false), holds(AL, F, V, step - 1),
   action(AL, _, _, ROBOT, A), fluent(AL, inertial, F, V), pl(AL), step > start.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Rules for action concurrency
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% If concurrency is disabled then plan actions sequentially
:- occurs(AL, ROBOT, A_1, step), occurs(AL, ROBOT, A_2, step), A_1 != A_2,
   action(AL, _, _, ROBOT, A_1), action(AL, _, _, ROBOT, A_2), pl(AL), step > start, concurrency = false.

%% If a robot plans a sequentially restricted action then it cannot concurrently plan another action on the same step
:- occurs(AL, ROBOT, A_1, step), occurs(AL, ROBOT, A_2, step), A_1 != A_2,
   action(AL, sequential, _, ROBOT, A_1), action(AL, _, _, ROBOT, A_2), pl(AL), step > start, concurrency = true.

%% Concurrently planned actions should not effect the same fluent
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V_1, step), effect(AL, ROBOT_2, A_2, F, V_2, step),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V_1), fluent(AL, inertial, F, V_2),
   A_1 != A_2, ROBOT_1 != ROBOT_2, pl(AL), step > start, concurrency = true.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V_1, step), effect(AL, ROBOT_2, A_2, F, V_2),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V_1), fluent(AL, inertial, F, V_2),
   A_1 != A_2, ROBOT_1 != ROBOT_2, pl(AL), step > start, concurrency = true.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V_1), effect(AL, ROBOT_2, A_2, F, V_2),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V_1), fluent(AL, inertial, F, V_2),
   A_1 != A_2, ROBOT_1 != ROBOT_2, pl(AL), step > start, concurrency = true.

%% Concurrently planned actions must also be sequentially plannable in all possible permutations;
%%      - Do not plan an action whose effect would falsify the positive precondition of a concurrently planned action,
%%      - Do not plan an action whose effect would truify the negative precondition of a concurrently planned action.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V_1, step), precond(AL, ROBOT_2, A_2, F, V_2, true, step),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V_1), fluent(AL, inertial, F, V_2),
   A_1 != A_2, V_1 != V_2, pl(AL), step > start, concurrency = true.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V_1, step), precond(AL, ROBOT_2, A_2, F, V_2, true),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V_1), fluent(AL, inertial, F, V_2),
   A_1 != A_2, V_1 != V_2, pl(AL), step > start, concurrency = true.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V, step), precond(AL, ROBOT_2, A_2, F, V, false, step),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V),
   A_1 != A_2, pl(AL), step > start, concurrency = true.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V, step), precond(AL, ROBOT_2, A_2, F, V, false),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V),
   A_1 != A_2, pl(AL), step > start, concurrency = true.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V_1), precond(AL, ROBOT_2, A_2, F, V_2, true, step),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V_1), fluent(AL, inertial, F, V_2),
   A_1 != A_2, V_1 != V_2, pl(AL), step > start, concurrency = true.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V_1), precond(AL, ROBOT_2, A_2, F, V_2, true),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V_1), fluent(AL, inertial, F, V_2),
   A_1 != A_2, V_1 != V_2, pl(AL), step > start, concurrency = true.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V), precond(AL, ROBOT_2, A_2, F, V, false, step),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V),
   A_1 != A_2, pl(AL), step > start, concurrency = true.
:- occurs(AL, ROBOT_1, A_1, step), occurs(AL, ROBOT_2, A_2, step), effect(AL, ROBOT_1, A_1, F, V), precond(AL, ROBOT_2, A_2, F, V, false),
   action(AL, concurrent, _, ROBOT_1, A_1), action(AL, concurrent, _, ROBOT_2, A_2), fluent(AL, inertial, F, V),
   A_1 != A_2, pl(AL), step > start, concurrency = true.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Rules for ensuring plans achieve all goals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% External defining when to enforce achieving the final-goal in sequential yield planning
#external seq_achieve_fgoals(step).

%% Rules defining whether to enforce achievement of the final-goal
enforce_achieve_fgoals(step) :- achieve_fgoals = true.
enforce_achieve_fgoals(step) :- 0 <= S, S <= step, seq_achieve_fgoals(S).

%% A goal is satisfied by the plan if there is step in which the associated fluent holds the current boolean truth value
goal_satisfied(AL, F, V, true, step) :- final_goal(AL, F, V, true), holds(AL, F, V, step), fluent(AL, _, F, V), pl(AL), enforce_achieve_fgoals(step).
goal_satisfied(AL, F, V, false, step) :- final_goal(AL, F, V, false), not holds(AL, F, V, step), fluent(AL, _, F, V), pl(AL), enforce_achieve_fgoals(step).

%% The plan is incomplete at some step I if there is at least one goal that is not satisfied in the state at step I
incomplete_plan(step) :- final_goal(AL, F, V, B), not goal_satisfied(AL, F, V, B, step), fluent(AL, _, F, V), pl(AL), boolean(B), enforce_achieve_fgoals(step).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Rules for generating sub-goals from the current planning level
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Effects of actions planned at the current abstraction level are passed as sub-goals to the next level
sub_goal(AL, ROBOT, A, F, V, step) :- occurs(AL, ROBOT, A, step), effect(AL, ROBOT, A, F, V, step),
                                      action(AL, _, _, ROBOT, A), fluent(AL, inertial, F, V),
                                      pl(AL), AL != 1, step > start.
sub_goal(AL, ROBOT, A, F, V, step) :- occurs(AL, ROBOT, A, step), effect(AL, ROBOT, A, F, V),
                                      action(AL, _, _, ROBOT, A), fluent(AL, inertial, F, V),
                                      pl(AL), AL != 1, step > start.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Optimisation Module
%**% #program optimisation_module(step, start, minimise_actions, order_fgoals, preempt_pos_fgoals, preempt_neg_fgoals, preempt_mode).
%%%%
%%%% Parameters
%%%% ----------
%%%%    step: int - A natural number defining the current planning horizon.
%%%%    start: int - A natural number defining the step to begin planning from.
%%%%    minimise_actions: bool - A Boolean, True to enable minimisation of the total quantity of actions in a plan, False to disable.
%%%%    order_fgoals: bool - A Boolean, True to enable enforce the task critical final-goal literal intermediate ordering preferences as a minimisation problem, False to disable.
%%%%    preempt_pos_fgoals: bool - A Boolean, True to enable the search heuristic that prefers pre-mptively achieving positive final-goal literals, False to disable.
%%%%    preempt_neg_fgoals: bool - A Boolean, True to enable the search heuristic that prefers pre-emptively achieving negative final-goal literals, False to disable.
%%%%    preempt_mode: {heuristic, optimise} - A string defining the type of statements used to enforce pre-emptive achievement of final-goal literals.
%%%%                                          Heuristic uses domain heuristic rules which affect solving throughout all search steps.
%%%%                                          Optimise uses weak constraint rules which maximise pre-emptive achievement of final-goals in the end state only.
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Eliminate any unnecessary actions from the plan by minimising the total quantity of actions in the minimal length plan.
:~ occurs(AL, ROBOT, A, step), action(AL, _, _, ROBOT, A), pl(AL), minimise_actions = true. [1 @ 2, ROBOT, A, step]

%% Final-goal intermediate achievement ordering preferences;
%%      - Solve the perference order problem as a minimisation problem;
%%          - Final-goal literals with a lower preference order have a large achievement cost weight that higher preference order,
%%          - The total cost of achieving a final-goal literal is the cost weight multiplied by the step upon which it is first satisfied,
%%          - This means that the minimal cost if attained by achieving lower preference order final-goal literals earlier than higher preference order literals.
:~ goal_satisfied(AL, F, V, B, step), not goal_satisfied(AL, F, V, B, step - 1), goal_order(AL, F, V, B, ORDER), final_goal(AL, F, V, B),
   fluent(AL, _, F, V), pl(AL), boolean(B), max_order(M), order_fgoals = true. [(((M + 1) - ORDER) * (step)) @ 3, AL, F, V, B]
max_order(M) :- M = #max { ORDER : goal_order(AL, F, V, B, ORDER), final_goal(AL, F, V, B), fluent(AL, _, F, V), boolean(B) }, pl(AL), order_fgoals = true.

%% Final-goal pre-emptive achievement heuristic based enforcement
#heuristic occurs(AL, ROBOT, A, step) : action(AL, _, _, ROBOT, A), effect(AL, ROBOT, A, F, V), final_goal(AL, F, V, true), fluent(AL, inertial, F, V), pl(AL),
                                        step > start, preempt_pos_fgoals = true, preempt_mode = heuristic. [10 @ 1, true]
#heuristic occurs(AL, ROBOT, A, step) : action(AL, _, _, ROBOT, A), effect(AL, ROBOT, A, F, V, step), final_goal(AL, F, V, true),
                                        fluent(AL, inertial, F, V), pl(AL), step > start, preempt_pos_fgoals = true, preempt_mode = heuristic. [10 @ 1, true]
#heuristic occurs(AL, ROBOT, A, step) : action(AL, _, _, ROBOT, A), effect(AL, ROBOT, A, F, V), final_goal(AL, F, V, false),
                                        fluent(AL, inertial, F, V), pl(AL), step > start, preempt_neg_fgoals = true, preempt_mode = heuristic. [10 @ 0, false]
#heuristic occurs(AL, ROBOT, A, step) : action(AL, _, _, ROBOT, A), effect(AL, ROBOT, A, F, V, step), final_goal(AL, F, V, false),
                                        fluent(AL, inertial, F, V), pl(AL), step > start, preempt_neg_fgoals = true, preempt_mode = heuristic. [10 @ 0, false]

%% Final-goal pre-emptive achievement optimisation based enforcement
:~ occurs(AL, ROBOT, A, step), action(AL, _, _, ROBOT, A), effect(AL, ROBOT, A, F, V), final_goal(AL, F, V, true),
   fluent(AL, inertial, F, V), pl(AL), step > start, preempt_pos_fgoals = true, preempt_mode = optimise. [-1 @ 1, AL, ROBOT, A, F, V, step]
:~ occurs(AL, ROBOT, A, step), action(AL, _, _, ROBOT, A), effect(AL, ROBOT, A, F, V, step), final_goal(AL, F, V, true),
   fluent(AL, inertial, F, V), pl(AL), step > start, preempt_pos_fgoals = true, preempt_mode = optimise. [-1 @ 1, AL, ROBOT, A, F, V, step]
:~ occurs(AL, ROBOT, A, step), action(AL, _, _, ROBOT, A), effect(AL, ROBOT, A, F, V), final_goal(AL, F, V, false),
   fluent(AL, inertial, F, V), pl(AL), step > start, preempt_neg_fgoals = true, preempt_mode = optimise. [1 @ 0, AL, ROBOT, A, F, V, step]
:~ occurs(AL, ROBOT, A, step), action(AL, _, _, ROBOT, A), effect(AL, ROBOT, A, F, V, step), final_goal(AL, F, V, false),
   fluent(AL, inertial, F, V), pl(AL), step > start, preempt_neg_fgoals = true, preempt_mode = optimise. [1 @ 0, AL, ROBOT, A, F, V, step]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Solution checking module
%**% #program check(step).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Solution checking constraint;
%%      - The problem is satisfiable only iff;
%%          - All currently included goals are achieved,
%%          - or the current step is less than the minimum search length bound minus one
%%            (the minus one is important since the initial state is at step zero, but the first plan step is step one).
%%      - There is no need to make the program unsatisfiable below the minimum search length bound,
%%        since the program is never solved below that step, and thus an answer set can only be found
%%        when the solver finds a plan that achieves all included goals beyond the minimum search length bound.
#external query(step).
#const minimum_search_length_bound = -1.
:- step > (minimum_search_length_bound - 1), incomplete_plan(step), query(step).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Conformance Module - Enforce conformance constraints by ensuring plans achieve all sub-goals from actions planned at the previous level in the correct order
%**% #program conformance_module(step, start, first, last, type, yield).
%%%%
%%%% Parameters
%%%% ----------
%%%%    step: int - A natural number defining the current planning horizon.
%%%%    start: int - A natural number defining the step to begin planning from.
%%%%    first: int - A natural number defining the first inclusive in sequence sub-goal stage index of the current conformance refinement planning problem.
%%%%    last: int - A natural number defining the last inclusive in sequence sub-goal stage index of the current conformance refinement planning problem.
%%%%    type: {simultaneous, sequential} - A string defining the sub-goal stage achievement type of subgoal stages.
%%%%    yield: bool - A Boolean, True to enable sequential yield planning mode, False to use standard planning mode.
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% External atom defining the current last in sequence sub-goal stage being enforced by the conformance constraint
%% and the step it was first commited at during sequential yield planning progression through the goal sequence.
#external current_last_sgoals(I, step) : sgoals_index(I), yield = true.

%% Include sub-goal stages as such;
%%      - Up to the current last in sequence sub-goal stage index in sequential yield planning mode,
%%      - Include all sub-goal stages simultaneously in standard planning mode,
%%      - The term "step >= S" is needed because "current_last_sgoals(I_L, S)" is only inserted once, where S is the achievement step of the previous sub-goal stage index "I_L - 1".
include_sgoals(I, step) :- I <= I_L, step >= S, current_last_sgoals(I_L, S), sgoals_index(I).
include_sgoals(I, step) :- sgoals_index(I), yield = false.

%% Define the available sub-goal stage index range;
%%      - The first index is always given (regardless of saved groundings),
%%      - The last is given;
%%          - Explicitly and fixed iff this program grounding will not be saved,
%%          - By a dynamic external context function (allowing it to be modified) if this program
%%            grounding will be saved and resumed to continue its refinement goal progression.
sgoals_index(first..last) :- last != none.
sgoals_index(first..L) :- last = none, L = @get_total_last_sgoals(PL), pl(PL).

%% The current sub-goal stage is initially that with whose sequence index is equal to the first included index
current_sub_goal(AL + 1, ROBOT, A, F, V, first, start + 1) :- sub_goal(AL + 1, ROBOT, A, F, V, first),
                                                              action(AL + 1, _, _, ROBOT, A), fluent(AL + 1, inertial, F, V),
                                                              pl(AL).

%% A current sub-goal is satisfied on the step its value holds
sub_goal_satisfied(AL + 1, ROBOT, A, F, V, I_SG, step) :- sub_goal(AL + 1, ROBOT, A, F, V, I_SG),
                                                          current_sub_goal(AL + 1, ROBOT, A, F, V, I_SG, step),
                                                          holds(AL + 1, F, V, step),
                                                          action(AL + 1, _, _, ROBOT, A), fluent(AL + 1, inertial, F, V),
                                                          pl(AL).

%% The current sub-goal stage is unachieved if any current sub-goal with that index is not satisfied
unachieved_sgoals(AL + 1, I_SG, step) :- current_sub_goal_index(AL + 1, I_SG, step),
                                         sub_goal(AL + 1, ROBOT, A, F, V, I_SG),
                                         current_sub_goal(AL + 1, ROBOT, A, F, V, I_SG, step),
                                         not sub_goal_satisfied(AL + 1, ROBOT, A, F, V, I_SG, step),
                                         action(AL + 1, _, _, ROBOT, A), fluent(AL + 1, inertial, F, V),
                                         pl(AL).

%% Sequential sub-goal stage achievement: A current sub-goal is propagated to the next time step if it has not yet been satisifed
current_sub_goal(AL + 1, ROBOT, A, F, V, I_SG, step) :- sub_goal(AL + 1, ROBOT, A, F, V, I_SG),
                                                        current_sub_goal(AL + 1, ROBOT, A, F, V, I_SG, step - 1),
                                                        not sub_goal_satisfied(AL + 1, ROBOT, A, F, V, I_SG, step - 1),
                                                        action(AL + 1, _, _, ROBOT, A), fluent(AL + 1, inertial, F, V),
                                                        type = sequential, pl(AL).

%% Simultaneous sub-goal stage achievement: A current sub-goal is propagated to the next time step if the current sub-goal stage is unachieved
current_sub_goal(AL + 1, ROBOT, A, F, V, I_SG, step) :- sub_goal(AL + 1, ROBOT, A, F, V, I_SG),
                                                        current_sub_goal(AL + 1, ROBOT, A, F, V, I_SG, step - 1),
                                                        unachieved_sgoals(AL + 1, I_SG, step - 1),
                                                        action(AL + 1, _, _, ROBOT, A), fluent(AL + 1, inertial, F, V),
                                                        type = simultaneous, pl(AL).

%% If there are no current sub-goals from the index value that was current at the previous time step then the next sub-goals in the sequence become current
current_sub_goal(AL + 1, ROBOT_1, A_1, F_1, V_1, I_SG + 1, step) :- include_sgoals(I_SG + 1, step),
                                                                    sub_goal(AL + 1, ROBOT_1, A_1, F_1, V_1, I_SG + 1),
                                                                    current_sub_goal_index(AL + 1, I_SG, step - 1),
                                                                    not current_sub_goal_index(AL + 1, I_SG, step),
                                                                    action(AL + 1, _, _, ROBOT_1, A_1), fluent(AL + 1, inertial, F_1, V_1),
                                                                    pl(AL).

%% A sub-goal stage is achieved on the last step upon which its sequence index is current;
%%      - This means that the sub-goal stage is current on all steps of the child state transitions that refine this stage,
%%      - and the sub-goal stage is achieved on the step of the end state of the matching child state transition.
sgoals_ach_at(AL + 1, I_SG, step) :- current_sub_goal_index(AL + 1, I_SG, step),
                                     not unachieved_sgoals(AL + 1, I_SG, step),
                                     pl(AL).

%% The current sub-goal stage sequence index is equal to the index value of the current sub-goals
current_sub_goal_index(AL + 1, I_SG, step) :- sub_goal(AL + 1, ROBOT, A, F, V, I_SG),
                                              current_sub_goal(AL + 1, ROBOT, A, F, V, I_SG, step),
                                              action(AL + 1, _, _, ROBOT, A), fluent(AL + 1, inertial, F, V),
                                              pl(AL).

%% The plan is incomplete at some step I if there is at least one sub-goal that is current at step I
incomplete_plan(step) :- sub_goal(AL + 1, ROBOT, A, F, V, I_SG),
                         current_sub_goal(AL + 1, ROBOT, A, F, V, I_SG, step),
                         not sub_goal_satisfied(AL + 1, ROBOT, A, F, V, I_SG, step),
                         action(AL + 1, _, _, ROBOT, A), fluent(AL + 1, inertial, F, V),
                         pl(AL).
incomplete_plan(step) :- sub_goal(AL + 1, ROBOT, A, F, V, I_SG),
                         current_sub_goal(AL + 1, ROBOT, A, F, V, I_SG, step),
                         include_sgoals(I_SG + 1, step),
                         action(AL + 1, _, _, ROBOT, A), fluent(AL + 1, inertial, F, V),
                         pl(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Goal Abstraction Module - Generate a conforming final-goal over all levels in the abstraction hierarchy
%**% #program goal_abstraction_module.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initial final-goals are applied in the initial state
holds(AL, F, V, 0) :- final_goal(AL, F, V, true), goal_fluent(AL, F), fluent(AL, inertial, F, V), al(AL).
not holds(AL, F, V, 0) :- final_goal(AL, F, V, false), goal_fluent(AL, F), fluent(AL, inertial, F, V), al(AL).

%% Defined final-goals must be satisfied in the initial state
:- holds(AL, F, false, 0), final_goal(AL, F, true, true), goal_fluent(AL, F), fluent(AL, defined, F, _), al(AL).
:- holds(AL, F, false, 0), final_goal(AL, F, false, false), goal_fluent(AL, F), fluent(AL, defined, F, _), al(AL).
:- holds(AL, F, true, 0), final_goal(AL, F, false, true), goal_fluent(AL, F), fluent(AL, defined, F, _), al(AL).
:- holds(AL, F, true, 0), final_goal(AL, F, true, false), goal_fluent(AL, F), fluent(AL, defined, F, _), al(AL).

%% Convert the initial state back to final-goal literals
final_goal(AL, F, V, true) :- holds(AL, F, V, 0), goal_fluent(AL, F), fluent(AL, _, F, V), al(AL).
final_goal(AL, F, V, false) :- not holds(AL, F, V, 0), goal_fluent(AL, F), fluent(AL, _, F, V), al(AL).

%% Law of Awareness - All inertial fluents can take any value from their range in the initial state
{ holds(AL, F, V, 0) } :- goal_fluent(AL, F), fluent(AL, inertial, F, V), al(AL).

%% Law of Continuity - All Fluents must have unique values to ensure the state is complete
:- not { holds(AL, F, V, 0) : fluent(AL, _, F, V) } = 1, goal_fluent(AL, F), fluent(AL, _, F, _), al(AL).

%% Closed world assumption for defined fluents with a boolean range - If there is no reason to believe a defined fluent is true then it must be false
holds(AL, F, false, 0) :- not holds(AL, F, true, 0), goal_fluent(AL, F), fluent(AL, defined, F, B), boolean(B), al(AL).