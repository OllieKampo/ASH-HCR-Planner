%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The Blocks World Plus Planning Domain                                 %%
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
%%%% Static Domain State - The static domain structure is defined here
%**% #program static_state.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Static Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Room Layout
%%                                ___________________
%%                               |                   |
%%                               |   starting_room   |
%%                               |___________________|
%%                                         ^
%%      ___________________       _________v_________       ___________________
%%     |                   |     |                   |     |                   |
%%     |    store_room     | <-> |      hallway      | <-> |    puzzle_room    |
%%     |___________________|     |___________________|     |___________________|
%%                                         ^
%%      ___________________       _________v_________       ___________________
%%     |                   |     |                   |     |                   |
%%     |    store_room     | <-> |      hallway      | <-> |    puzzle_room    |
%%     |___________________|     |___________________|     |___________________|
%%                                         ^
%%      ___________________       _________v_________       ___________________
%%     |                   |     |                   |     |                   |
%%     |    store_room     | <-> |      hallway      | <-> |    puzzle_room    |
%%     |___________________|     |___________________|     |___________________|
%%                                         ^
%%                                         .
%%                                         .

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Variables for changing number of puzzles

%% We test the blocks world plus with 1, 3 and 5 puzzles
total_puzzles(2).
% total_puzzles(3).
% total_puzzles(5).

%% There are 2 or 3 towers per blocks world puzzle
towers_per_puzzle(2).
% towers_per_puzzle(3).

%% There are between 3 and 5 blocks per tower
blocks_per_tower(3).
% blocks_per_tower(4).
% blocks_per_tower(5).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Cell connections between rooms
is(1, connected(((starting_room, 1), 1), ((hallway, 1), 1))).

is(1, connected(((store_room, N), 1), ((hallway, N), 0))) :- puzzle_number(N).
is(1, connected(((hallway, N), 2), ((puzzle_room, N), 0))) :- puzzle_number(N).

is(1, connected(((hallway, N), 1), ((hallway, N + 1), 1))) :- puzzle_number(N).

% is(1, connected_by_door(((starting_room, 1), door), ((starting_room, 1), 1), ((hallway, 1), 1))).

is(1, connected_by_door(((store_room, N), door), ((store_room, N), 1), ((hallway, N), 0))) :- puzzle_number(N).
is(1, connected_by_door(((puzzle_room, N), door), ((hallway, N), 2), ((puzzle_room, N), 0))) :- puzzle_number(N).

% is(1, connected_by_door(((hallway, N + 1), door), ((hallway, N), 1), ((hallway, N + 1), 1))) :- puzzle_number(N), puzzle_number(N + 1).

%% The door's handles are located on either side of it
is(1, in((((store_room, N), door), handle(inner)), ((store_room, N), 1))) :- puzzle_number(N).
is(1, in((((store_room, N), door), handle(outer)), ((hallway, N), 0))) :- puzzle_number(N).

is(1, in((((puzzle_room, N), door), handle(inner)), ((puzzle_room, N), 0))) :- puzzle_number(N).
is(1, in((((puzzle_room, N), door), handle(outer)), ((hallway, N), 2))) :- puzzle_number(N).

% is(1, in(starting_room_door_handle_inner, (starting_room, 1))).
% is(1, in(starting_room_door_handle_outer, (hallway, 1))).

%% Block colouring
% is(AL, colour_of((BLOCK, N), red)) :- static(AL, colour_of((BLOCK, N), red)),
%                                       insta_of(AL, block, (BLOCK, N)),
%                                       V = (N \ (N_B * N_T)), V <= N_B,
%                                       blocks_per_tower(N_B), towers_per_puzzle(N_T), al(AL).
% is(AL, colour_of((BLOCK, N), blue)) :- static(AL, colour_of((BLOCK, N), blue)),
%                                        insta_of(AL, block, (BLOCK, N)),
%                                        V = (N \ (N_B * N_T)), V > N_B, V <= (N_B * 2),
%                                        blocks_per_tower(N_B), towers_per_puzzle(N_T), al(AL).
% is(AL, colour_of((BLOCK, N), yellow)) :- static(AL, colour_of((BLOCK, N), yellow)),
%                                          insta_of(AL, block, (BLOCK, N)),
%                                          N > 6, N <= 9, al(AL).

is(AL, colour_of((block, 1), red)) :- al(AL).
is(AL, colour_of((block, 2), red)) :- al(AL).
is(AL, colour_of((block, 3), red)) :- al(AL).
is(AL, colour_of((block, 7), red)) :- al(AL).
is(AL, colour_of((block, 8), red)) :- al(AL).
is(AL, colour_of((block, 9), red)) :- al(AL).
is(AL, colour_of((block, 4), blue)) :- al(AL).
is(AL, colour_of((block, 5), blue)) :- al(AL).
is(AL, colour_of((block, 6), blue)) :- al(AL).
is(AL, colour_of((block, 10), blue)) :- al(AL).
is(AL, colour_of((block, 11), blue)) :- al(AL).
is(AL, colour_of((block, 12), blue)) :- al(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Static Relations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Adjacent cells are connected
is(1, connected((A, X_1), (A, X_2))) :- insta_of(1, cell, (A, X_1)),
                                        insta_of(1, cell, (A, X_2)),
                                        X_2 = X_1 + 1.

%% Reversability of cell connections
is(1, connected(C_1, C_2)) :- is(1, connected(C_2, C_1)),
                              insta_of(1, cell, C_1),
                              insta_of(1, cell, C_2).

%% Reversibilty of door connections between cells
is(1, connected_by_door(D, C_1, C_2)) :- is(1, connected_by_door(D, C_2, C_1)),
                                         insta_of(1, door, D),
                                         insta_of(1, cell, C_1),
                                         insta_of(1, cell, C_2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Specific initial state
%**% #program initial_state.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Talos starts in the starting room
holds(1, in(talos), ((starting_room, 1), 0), 0).

%% Talos start with nothing in its grasp
holds(AL, grasping(DESC), nothing, 0) :- fluent(AL, inertial, grasping(DESC), nothing),
                                         insta_of(AL, grasper, DESC),
                                         al(AL).

%% Talos starts with its extensible descendants retracted
holds(AL, configuration(DESC), retracted, 0) :- fluent(AL, inertial, configuration(DESC), retracted),
                                                insta_of(AL, extensible, DESC),
                                                sl(AL).

%% Talos starts with its end effectors aligned with nothing
holds(1, configuration(END), aligned_with(nothing), 0) :- fluent(AL, inertial, configuration(END), aligned_with(nothing)),
                                                          insta_of(AL, manipulator_end, END),
                                                          state_of(AL, END, aligned_with(nothing)),
                                                          sl(AL).

%% The doors start closed
holds(1, configuration(DOOR), closed, 0) :- fluent(AL, inertial, configuration(DOOR), closed),
                                            insta_of(AL, door, DOOR),
                                            state_of(AL, DOOR, closed),
                                            sl(AL).

%% The puzzle room table is in the puzzle room
holds(1, in(((puzzle_room, N), table)), ((puzzle_room, N), 1), 0) :- puzzle_number(N).

%% The first four blocks are stacked on the puzzle room table
holds(1, on((block, (N * 6) - 5)), (((puzzle_room, N), table), left), 0) :- puzzle_number(N).
holds(1, on((block, (N * 6) - 4)), (((puzzle_room, N), table), right), 0) :- puzzle_number(N).
holds(1, on((block, (N * 6) - 3)), (block, (N * 6) - 5), 0) :- puzzle_number(N).

%% The other three blocks are on the floor in the store room
holds(1, in((block, (N * 6) - 2)), ((store_room, N), 0), 0) :- puzzle_number(N).
holds(1, in((block, (N * 6) - 1)), ((store_room, N), 0), 0) :- puzzle_number(N).
holds(1, in((block, (N * 6) - 0)), ((store_room, N), 0), 0) :- puzzle_number(N).
holds(1, on((block, (N * 6) - 2)), nothing, 0) :- puzzle_number(N).
holds(1, on((block, (N * 6) - 1)), nothing, 0) :- puzzle_number(N).
holds(1, on((block, (N * 6) - 0)), nothing, 0) :- puzzle_number(N).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Goals - The specific goal states the robot(s) must satisfy are defined here
%**% #program goal_state.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% The fluents pertinent to the goal state are the placement of blocks and the completeness of the towers
goal_fluent(AL, complete_puzzle(TABLE), true, true) :- insta_of(AL, table, TABLE), al(AL).
goal_fluent(AL, complete_tower(COLOUR, TABLE)) :- insta_of(AL, colour, COLOUR), insta_of(AL, table, TABLE), al(AL).
goal_fluent(AL, unordered_tower(BASE, TABLE)) :- insta_of(AL, block, BASE), insta_of(AL, table, TABLE), al(AL).
goal_fluent(AL, in_tower(BASE, BLOCK)) :- insta_of(AL, block, BASE), insta_of(AL, block, BLOCK), al(AL).
goal_fluent(AL, tower_base(BASE, TABLE)) :- insta_of(AL, block, BASE), insta_of(AL, table, TABLE), al(AL).
goal_fluent(AL, on(BLOCK)) :- insta_of(AL, block, BLOCK), al(AL).

%% The goal is to put block 2 on the left hand side of the table in puzzle room 2 whilst leaving block 1 on the table in puzzle room 1
final_goal(AL, complete_puzzle(TABLE), true, true) :- insta_of(AL, table, TABLE), al(AL).
final_goal(AL, complete_tower(COLOUR, TABLE), true, true) :- insta_of(AL, colour, COLOUR), insta_of(AL, table, TABLE), al(AL).
final_goal(1, on((block, (N * 6) - 3)), (((puzzle_room, N), table), left), true) :- puzzle_number(N).
final_goal(1, on((block, (N * 6) - 0)), (((puzzle_room, N), table), right), true) :- puzzle_number(N).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Entities and Component Relations
%**% #program entities.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%s%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Entity Instance Declarations - entity(CLS, EN) declares an entity EN of class CLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

goal_order(AL, complete_puzzle(TABLE), true, true, N) :- insta_of(AL, table, TABLE), TABLE = ((_, N), _), al(AL).
% goal_order(AL, complete_tower(COLOUR, TABLE), true, true, N) :- insta_of(AL, colour, COLOUR), insta_of(AL, table, TABLE), TABLE = ((_, N), _), al(AL).

%% Talos is an armed robot
entity(armed_robot, talos).

%% Talos has two manipulator arms, each of which is composed on a limb and end effector
entity(manipulator_arm, manipulator_arm_1).
entity(manipulator_arm, manipulator_arm_2).

entity(manipulator_limb, manipulator_limb_1).
entity(manipulator_limb, manipulator_limb_2).

entity(manipulator_end, manipulator_end_1).
entity(manipulator_end, manipulator_end_2).

puzzle_number(1..N) :- total_puzzles(N).
tower_number(1..N) :- towers_per_puzzle(N).

%% Define blocks
block_number(1..B_N) :- B_N = P_N * T_P * B_T, total_puzzles(P_N), towers_per_puzzle(T_P), blocks_per_tower(B_T).
entity(block, (block, N)) :- block_number(N).

%% There are four rooms
entity(room, (starting_room, 1)).
entity(room, (store_room, N)) :- puzzle_number(N).
entity(room, (puzzle_room, N)) :- puzzle_number(N).
entity(room, (hallway, N)) :- puzzle_number(N).

%% All the rooms have two cells except the hallway which has three
declare_room_size(starting_room, 2).
declare_room_size(store_room, 2).
declare_room_size(hallway, 3).
declare_room_size(puzzle_room, 2).
entity(cell, (ROOM, 0..MAX)) :- entity(room, ROOM), ROOM = (TYPE, N), declare_room_size(TYPE, MAX), puzzle_number(N).

%% The is one door for each room except the starting room
entity(door, (ROOM, door)) :- entity(room, ROOM), ROOM != (starting_room, 1).

%% Each door has two handles
declare_handle(inner;outer).
entity(handle, (D, handle(H))) :- entity(door, D), declare_handle(H).

%% There is one table in each puzzle room
entity(table, (ROOM, table)) :- entity(room, ROOM), ROOM = (TYPE, N), TYPE = puzzle_room, puzzle_number(N).

%% The table has three sides
declare_table_side(left;middle;right).
entity(side, (TABLE, SIDE)) :- declare_table_side(SIDE), entity(table, TABLE).

%% Create available block colours and assign each colour to exactly one tower number
declare_colour(red;blue) :- towers_per_puzzle(2).
declare_colour(red;blue;yellow) :- towers_per_puzzle(3).
entity(colour, C) :- declare_colour(C).
% { tower_colour(T_N, C) } :- tower_number(T_N), declare_colour(C).
% :- not { tower_colour(T_N, C) : declare_colour(C) } = 1, tower_number(T_N).
% :- not { tower_colour(T_N, C) : tower_number(T_N) } <= 1, declare_colour(C).

%% Entity constant for nothing
entity(surface, nothing).
entity(graspable, nothing).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Entity Ancestry Relationships - ancestry_relation(EN, DESCE) delcares that entity EN has a descendant DESCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Relations between Talos and his manipulator arms
ancestry_relation(talos, manipulator_arm_1).
ancestry_relation(talos, manipulator_arm_2).

%% Relations between Talos' manipulator arms, and the limbs and end-effectors that compose them
ancestry_relation(manipulator_arm_1, manipulator_limb_1).
ancestry_relation(manipulator_arm_2, manipulator_limb_2).
ancestry_relation(manipulator_arm_1, manipulator_end_1).
ancestry_relation(manipulator_arm_2, manipulator_end_2).

%% Relations between rooms and the cells that compose them
ancestry_relation(ROOM, (ROOM, CELL)) :- entity(cell, (ROOM, CELL)), entity(room, ROOM).

%% Relations between the puzzle room door and its handles
ancestry_relation(DOOR, (DOOR, HANDLE)) :- entity(handle, (DOOR, HANDLE)), entity(door, DOOR).

%% Relations betweens the puzzle room table and its sides
ancestry_relation(TABLE, (TABLE, SIDE)) :- entity(side, (TABLE, SIDE)), entity(table, TABLE).