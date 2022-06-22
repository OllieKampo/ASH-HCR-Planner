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
%%

%% Cell connections between rooms
is(1, connected((starting_room, 1), (hallway, 1))).
is(1, connected((store_room, 1), (hallway, 0))).
is(1, connected((hallway, 2), (puzzle_room, 0))).

%% Door connection between the hallway and the puzzle room
is(1, connected_by_door(store_room_door, (store_room, 1), (hallway, 0))).
% is(1, connected_by_door(starting_room_door, (starting_room, 1), (hallway, 1))).
is(1, connected_by_door(puzzle_room_door, (hallway, 2), (puzzle_room, 0))).

%% The door's handles are located on either side of it
is(1, in(store_room_door_handle_inner, (store_room, 1))).
is(1, in(store_room_door_handle_outer, (hallway, 0))).
is(1, in(starting_room_door_handle_inner, (starting_room, 1))).
is(1, in(starting_room_door_handle_outer, (hallway, 1))).
is(1, in(puzzle_room_door_handle_inner, (puzzle_room, 0))).
is(1, in(puzzle_room_door_handle_outer, (hallway, 2))).

%% Block colouring
is(AL, colour_of((BLOCK, N), red)) :- static(AL, colour_of((BLOCK, N), red)),
                                      insta_of(AL, block, (BLOCK, N)),
                                      N <= 3, al(AL).
is(AL, colour_of((BLOCK, N), blue)) :- static(AL, colour_of((BLOCK, N), blue)),
                                       insta_of(AL, block, (BLOCK, N)),
                                       N > 3, N <= 6, al(AL).

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
holds(1, in(talos), (starting_room, 0), 0).

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
holds(1, in(puzzle_room_table), (puzzle_room, 1), 0).

%% The first four blocks are stacked on the puzzle room table
holds(1, on((block, 1)), puzzle_room_table_left, 0).
holds(1, on((block, 2)), puzzle_room_table_right, 0).
holds(1, on((block, 3)), (block, 1), 0).

%% The other three blocks are on the floor in the store room
holds(1, in((block, 4)), (store_room, 0), 0).
holds(1, in((block, 5)), (store_room, 0), 0).
holds(1, in((block, 6)), (store_room, 0), 0).
holds(1, on((block, 4)), nothing, 0).
holds(1, on((block, 5)), nothing, 0).
holds(1, on((block, 6)), nothing, 0).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Goals - The specific goal states the robot(s) must satisfy are defined here
%**% #program goal_state.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% The fluents pertinent to the goal state are the placement of blocks and the completeness of the towers
goal_fluent(AL, complete_tower(COLOUR, TABLE)) :- insta_of(AL, colour, COLOUR), insta_of(AL, table, TABLE), al(AL).
goal_fluent(AL, unordered_tower(BASE, TABLE)) :- insta_of(AL, block, BASE), insta_of(AL, table, TABLE), al(AL).
goal_fluent(AL, in_tower(BASE, BLOCK)) :- insta_of(AL, block, BASE), insta_of(AL, block, BLOCK), al(AL).
goal_fluent(AL, tower_base(BASE, TABLE)) :- insta_of(AL, block, BASE), insta_of(AL, table, TABLE), al(AL).
goal_fluent(AL, on(BLOCK)) :- insta_of(AL, block, BLOCK), al(AL).

%% The goal is to put block 2 on the left hand side of the table in puzzle room 2 whilst leaving block 1 on the table in puzzle room 1
final_goal(AL, complete_tower(COLOUR, TABLE), true, true) :- insta_of(AL, colour, COLOUR), insta_of(AL, table, TABLE), al(AL).
final_goal(1, on((block, 3)), puzzle_room_table_left, true).
final_goal(1, on((block, 6)), puzzle_room_table_right, true).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Entities and Ancestry Relations
%**% #program entities.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%s%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Entity Instance Declarations - entity(CLS, EN) declares an entity EN of class CLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Talos is an armed robot
entity(armed_robot, talos).

%% Talos has two manipulator arms, each of which is composed on a limb and end effector
entity(manipulator_arm, manipulator_arm_1).
entity(manipulator_limb, manipulator_limb_1).
entity(manipulator_end, manipulator_end_1).
entity(manipulator_arm, manipulator_arm_2).
entity(manipulator_limb, manipulator_limb_2).
entity(manipulator_end, manipulator_end_2).

%% There are six blocks
block_number(1..6).
blocks_per_tower(3).
entity(block, (block, N)) :- block_number(N).

%% There are four rooms
entity(room, starting_room).
entity(room, store_room).
entity(room, puzzle_room).
entity(room, hallway).

%% All the rooms have two cells except the hallway which has three
x(0..3).
size(starting_room, 2).
size(store_room, 2).
size(hallway, 3).
size(puzzle_room, 2).
entity(cell, (ROOM, X)) :- entity(room, ROOM), size(ROOM, MAX), X < MAX, x(X).

%% The is one door between the hallway and the puzzle room
entity(door, store_room_door).
entity(door, starting_room_door).
entity(door, puzzle_room_door).

%% The door has two handles
entity(handle, store_room_door_handle_inner).
entity(handle, store_room_door_handle_outer).
entity(handle, starting_room_door_handle_inner).
entity(handle, starting_room_door_handle_outer).
entity(handle, puzzle_room_door_handle_inner).
entity(handle, puzzle_room_door_handle_outer).

%% There is one table in the puzzle room
entity(table, puzzle_room_table).

%% The table has three sides
entity(side, puzzle_room_table_left).
entity(side, puzzle_room_table_right).

%% Available colours
entity(colour, red).
entity(colour, blue).

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
ancestry_relation(ROOM, (ROOM, X)) :- entity(room, ROOM), entity(cell, (ROOM, X)), x(X).

%% Relations between the puzzle room door and its handles
ancestry_relation(store_room_door, store_room_door_handle_inner).
ancestry_relation(store_room_door, store_room_door_handle_outer).
ancestry_relation(starting_room_door, starting_room_door_handle_inner).
ancestry_relation(starting_room_door, starting_room_door_handle_outer).
ancestry_relation(puzzle_room_door, puzzle_room_door_handle_inner).
ancestry_relation(puzzle_room_door, puzzle_room_door_handle_outer).

%% Relations betweens the puzzle room table and its sides
ancestry_relation(puzzle_room_table, puzzle_room_table_left).
ancestry_relation(puzzle_room_table, puzzle_room_table_right).