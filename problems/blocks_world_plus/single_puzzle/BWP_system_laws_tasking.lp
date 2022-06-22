%% At first, it might seem effective to provide a tasking model, formed by abstracting away the robot's manipulator arms.
%% This means the robot no longer needs to reason about grasping objects, which simplifies the manipulation problem.
%% However, the abstract plan lengths are not reduced hugely by this abstraction.
%% In particular, the transportation problem is still quite complex because each block must still be collected individually.
%% We also still have to consider how many objects the robot could actually hold at any given time.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Domain Sorts
%**% #program domain_sorts.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Constants defining abstraction levels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Maximum abstraction level
#const abstraction_max = 4.

%% Abstraction types
#const reduc = 3. % Refine both
#const conde = 2. % Refine both
#const groun = 1. % Refine both
model_type(4, tasking). % Classical
model_type(3, reduced).
model_type(2, condensed).
model_type(1, ground).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Class Type Declarations - class(AL, CLS) declares a class CLS at abstraction level AL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Classes included in all abstractions
class(4, robot).
class(4, armed_robot).
class(4, location).
class(4, room).
class(4, object).
class(4, grounded).
class(4, graspable).
class(4, placeable).
class(4, block).
class(4, surface).
class(4, table).
class(4, colour).

%% Classes deleted by redefinition
class(3, manipulator_arm).
class(3, grasper).

%% Classes deleted by reduction
class(2, door).
class(2, handle).
class(2, configurable).
class(2, extensible).

%% Classes deleted by refinement
class(1, cell).
class(1, manipulator_limb).
class(1, manipulator_end).
class(1, side).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Class Hierarchy - super_class(C1, C2) defines C1 as a super class of C2 if class(AL_1, CLS_1), class(AL_2, CLS_2), AL_1 >= AL_2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Armed robots are robots
super_class(robot, armed_robot).

%% Location classes
super_class(location, room).
super_class(location, cell).

%% Object classes
super_class(object, robot).
super_class(object, manipulator_arm).
super_class(object, manipulator_limb).
super_class(object, manipulator_end).
super_class(object, table).
super_class(object, side).
super_class(object, block).

%% Grounded objects
super_class(object, grounded).
super_class(grounded, handle).

%% Configurable classes
super_class(configurable, extensible).
super_class(configurable, grasper).
super_class(configurable, door).

%% Extensible classes
super_class(extensible, manipulator_arm).
super_class(extensible, manipulator_limb).

%% Grasper classes
super_class(grasper, manipulator_arm).
super_class(grasper, manipulator_end).

%% Graspable classes
super_class(graspable, block).
super_class(graspable, handle).

%% Placeable classes
super_class(placeable, block).

%% Surface classes
super_class(surface, table).
super_class(surface, side).
super_class(surface, block).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Class Type Overrides - override_class(CLS_1, CLS_2, CLS_3) deletes the CLS_3 type of CLS_1 when for components of type CLS_2 exist at AL >= plan_at
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Manipulator arms can grasp objects because they are composed of a end effector
override_class(manipulator_arm, manipulator_end, grasper).

%% Manipulator arms are extensible because they are composed of a manipulator limb
override_class(manipulator_arm, manipulator_limb, extensible).

%% A room is composed from a set of cells
override_class(room, cell, location).

%% A table has multiple sides that compose its surface
override_class(table, side, surface).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Entity Configuration State Declarations - state(CLS, S) declares a state S for class CLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Extensible objects can be extended or retracted
state(AL, extensible, extended) :- AL < reduc, al(AL).
state(AL, extensible, retracted) :- AL < reduc, al(AL).

%% End effectors can be aligned with graspable objects
state(AL, manipulator_end, aligned_with(OBJ)) :- insta_of(AL, graspable, OBJ), class(AL_I, manipulator_end), AL <= AL_I, AL < reduc, al(AL).
state(AL, manipulator_end, aligned_with(nothing)) :- AL < reduc, al(AL).

%% Doors can be open or closed
state(AL, door, open) :- AL < reduc, al(AL).
state(AL, door, closed) :- AL < reduc, al(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Statics - Define the static domain properties here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Statics are given in the form - static(abstraction_level, static)

%% Locations can be connected
static(AL, connected(L_1, L_2)) :- insta_of(AL, location, L_1),
                                   insta_of(AL, location, L_2),
                                   al(AL).

%% Locations can be connected by doors
static(AL, connected_by_door(D, L_1, L_2)) :- insta_of(AL, door, D),
                                              insta_of(AL, location, L_1),
                                              insta_of(AL, location, L_2),
                                              AL < reduc, al(AL).

%% Locations for grounded entities
static(AL, in(OBJ, L)) :- insta_of(AL, object, OBJ),
                          insta_of(AL, grounded, OBJ),
                          insta_of(AL, location, L),
                          AL < reduc, al(AL).

%% Blocks have colours
static(AL, colour_of(BLOCK, COLOUR)) :- insta_of(AL, block, BLOCK),
                                        insta_of(AL, colour, COLOUR),
                                        al(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Fluents - Define the fluent domain properties here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Fluents are given in the form - fluent(abstraction_level, fluent_type, fluent, value)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inertial Fluents

%% All objects must be in a location
fluent(AL_1, inertial, in(OBJ), L) :- insta_of(AL_1, location, L),
                                      insta_of(AL_2, object, OBJ), OBJ != nothing,
                                      not insta_of(AL_2, grounded, OBJ),
                                      sl(AL_1), sl(AL_2),
                                      AL_2 >= AL_1.

%% Objects can be placed on other surfaces
fluent(AL_1, inertial, on(OBJ), SURFACE) :- OBJ != SURFACE,
                                            insta_of(AL_1, surface, SURFACE),
                                            insta_of(AL_2, placeable, OBJ),
                                            sl(AL_1), sl(AL_2),
                                            AL_2 >= AL_1.

%% Graspable objects can be grasped by a robot's graspers
fluent(AL_1, inertial, grasping(DESC), OBJ) :- insta_of(AL_1, graspable, OBJ),
                                               insta_of(AL_2, grasper, DESC),
                                               AL_1 < TL, AL_2 < TL,
                                               model_type(TL, tasking),
                                               sl(AL_1), sl(AL_2),
                                               AL_2 >= AL_1.

%% All of a robot's reconfigurable components must have a configuration
fluent(AL_1, inertial, configuration(DESC), STATE) :- state_of(AL_1, DESC, STATE),
                                                      insta_of(AL_2, configurable, DESC),
                                                      AL_1 < reduc, AL_2 < reduc,
                                                      sl(AL_1), sl(AL_2),
                                                      AL_2 >= AL_1.

%% Doors must have a configuration
fluent(AL_1, inertial, configuration(D), STATE) :- state_of(AL_1, D, STATE),
                                                   insta_of(AL_2, door, D),
                                                   AL_1 < reduc, AL_2 < reduc,
                                                   sl(AL_1), sl(AL_2), AL_2 >= AL_1.

%% In the redefined model robots can be transporting objects
fluent(AL_1, inertial, transporting(ROBO, OBJ), B) :- insta_of(AL_2, armed_robot, ROBO),
                                                      insta_of(AL_1, graspable, OBJ), OBJ != nothing,
                                                      AL_1 >= TL, model_type(TL, tasking),
                                                      sl(AL_2), sl(AL_1),
                                                      AL_2 >= AL_1,
                                                      boolean(B).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Defined Fluents

%% A block that is on a table is the base of a tower
fluent(AL, defined, tower_base(BASE, TABLE), B) :- insta_of(AL, block, BASE),
                                                   insta_of(AL, table, TABLE),
                                                   sl(AL), boolean(B).

%% A block that is the base of a tower or placed upon another block that is in a tower also in the same tower
fluent(AL, defined, in_tower(BASE, BLOCK), B) :- insta_of(AL, block, BASE),
                                                 insta_of(AL, block, BLOCK),
                                                 insta_of(AL, table, TABLE),
                                                 sl(AL), boolean(B).

%% A tower that is not stacked in descending order from top to bottom is unordered
fluent(AL, defined, unordered_tower(BASE, TABLE), B) :- insta_of(AL, block, BASE),
                                                        insta_of(AL, table, TABLE),
                                                        sl(AL), boolean(B).

%% A tower is complete if all the blocks that form it are the same colour and they are stacked in descending order from top to bottom
fluent(AL, defined, complete_tower(COLOUR, TABLE), B) :- insta_of(AL, colour, COLOUR),
                                                         insta_of(AL, table, TABLE),
                                                         sl(AL), boolean(B).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Actions - Define the available actions here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Actions are given in the form - action(abstraction_level, action_type, actor, action)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Agent Actions

%% Robots can move between locations
action(AL, sequential, locomotion, ROBO, move(L)) :- insta_of(AL, robot, ROBO),
                                                     insta_of(AL, location, L),
                                                     sl(AL).

%% Robots can grasp objects with their graspers
action(AL, concurrent, manipulation, ROBO, grasp(DESC, OBJ)) :- desce_of(AL, ROBO, DESC),
                                                                insta_of(AL, armed_robot, ROBO),
                                                                insta_of(AL, grasper, DESC),
                                                                insta_of(AL, graspable, OBJ), OBJ != nothing,
                                                                AL < TL, model_type(TL, tasking), sl(AL).

%% Robots can release objects from their graspers
action(AL, concurrent, manipulation, ROBO, release(DESC, OBJ)) :- desce_of(AL, ROBO, DESC),
                                                                  insta_of(AL, armed_robot, ROBO),
                                                                  insta_of(AL, grasper, DESC),
                                                                  insta_of(AL, graspable, OBJ), OBJ != nothing,
                                                                  AL < TL, model_type(TL, tasking), sl(AL).

%% Robots can set the configuration state of their configurable components
action(AL, concurrent, configuration, ROBO, configure(DESC, STATE)) :- desce_of(AL, ROBO, DESC),
                                                                       insta_of(AL, armed_robot, ROBO),
                                                                       insta_of(AL, configurable, DESC),
                                                                       state_of(AL, DESC, STATE),
                                                                       AL < TL, model_type(TL, tasking), AL < reduc, sl(AL).

%% Robots can put objects on surfaces
action(AL, concurrent, manipulation, ROBO, put(DESC, OBJ, SURFACE)) :- OBJ != SURFACE, SURFACE != nothing,
                                                                       desce_of(AL, ROBO, DESC),
                                                                       insta_of(AL, armed_robot, ROBO),
                                                                       insta_of(AL, grasper, DESC),
                                                                       insta_of(AL, placeable, OBJ),
                                                                       insta_of(AL, surface, SURFACE),
                                                                       AL < TL, model_type(TL, tasking), sl(AL).

%% Robots can lift objects off of surfaces
action(AL, concurrent, manipulation, ROBO, lift(DESC, OBJ)) :- desce_of(AL, ROBO, DESC),
                                                               insta_of(AL, armed_robot, ROBO),
                                                               insta_of(AL, grasper, DESC),
                                                               insta_of(AL, placeable, OBJ),
                                                               AL < TL, model_type(TL, tasking), sl(AL).

%% Robots can actuate doors
action(AL, concurrent, manipulation, ROBO, actuate(DESC, D)) :- insta_of(AL, armed_robot, ROBO),
                                                                insta_of(AL, grasper, DESC),
                                                                insta_of(AL, door, D),
                                                                AL < TL, model_type(TL, tasking), AL < reduc, sl(AL).

%% In the redefined model robots can collect objects
action(AL, concurrent, manipulation, ROBO, collect(OBJ)) :- insta_of(AL, armed_robot, ROBO),
                                              insta_of(AL, graspable, OBJ), OBJ != nothing,
                                              AL >= TL, model_type(TL, tasking), sl(AL).

%% In the redefined model robots can deposit objects
action(AL, concurrent, manipulation, ROBO, deposit(OBJ)) :- insta_of(AL, armed_robot, ROBO),
                                              insta_of(AL, graspable, OBJ), OBJ != nothing,
                                              AL >= TL, model_type(TL, tasking), sl(AL).

%% In the redefined model robots can pick up and put down objects with a single action
action(AL, concurrent, manipulation, ROBO, place_upon(OBJ, SURFACE)) :- insta_of(AL, armed_robot, ROBO),
                                                          insta_of(AL, placeable, OBJ),
                                                          insta_of(AL, surface, SURFACE),
                                                          AL >= TL, model_type(TL, tasking), sl(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Domain Specific Rules - The domain's dynamic behaviour is defined here
%**% #program action_effects(t).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Agent Action Effects - The direct effects of executing actions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% When a robot moves its location changes
effect(AL, ROBO, move(L), in(ROBO), L) :- action(AL, _, _, ROBO, move(L)),
                                          fluent(AL, inertial, in(ROBO), L),
                                          insta_of(AL, robot, ROBO),
                                          insta_of(AL, location, L),
                                          pl(AL).

%% When a robot grasps an object the object becomes grasped
effect(AL, ROBO, grasp(DESC, OBJ), grasping(DESC), OBJ) :- action(AL, _, _, ROBO, grasp(DESC, OBJ)),
                                                           fluent(AL, inertial, grasping(DESC), OBJ),
                                                           desce_of(AL, ROBO, DESC),
                                                           insta_of(AL, armed_robot, ROBO),
                                                           insta_of(AL, grasper, DESC),
                                                           insta_of(AL, graspable, OBJ), OBJ != nothing,
                                                           pl(AL).

%% When a robot releases an object the object is no longer grasped
effect(AL, ROBO, release(DESC, OBJ), grasping(DESC), nothing) :- action(AL, _, _, ROBO, release(DESC, OBJ)),
                                                                 fluent(AL, inertial, grasping(DESC), nothing),
                                                                 desce_of(AL, ROBO, DESC),
                                                                 insta_of(AL, armed_robot, ROBO),
                                                                 insta_of(AL, grasper, DESC),
                                                                 pl(AL).

%% When a robot releases an object from an end effector it is no longer aligned with that object
effect(AL, ROBO, release(END, OBJ), configuration(END), aligned_with(nothing)) :- action(AL, _, _, ROBO, release(END, OBJ)),
                                                                                  fluent(AL, inertial, configuration(END), aligned_with(nothing)),
                                                                                  desce_of(AL, ROBO, END),
                                                                                  insta_of(AL, armed_robot, ROBO),
                                                                                  insta_of(AL, manipulator_end, END),
                                                                                  insta_of(AL, graspable, OBJ),
                                                                                  state_of(AL, END, aligned_with(nothing)),
                                                                                  AL < reduc, pl(AL).

%% When a robot configures a component it causes the state of that component to change
effect(AL, ROBO, configure(DESC, STATE), configuration(DESC), STATE) :- action(AL, _, _, ROBO, configure(DESC, STATE)),
                                                                        fluent(AL, inertial, configuration(DESC), STATE),
                                                                        desce_of(AL, ROBO, DESC),
                                                                        insta_of(AL, robot, ROBO),
                                                                        insta_of(AL, configurable, DESC),
                                                                        state_of(AL, DESC, STATE),
                                                                        AL < reduc, pl(AL).

%% When a robot retracts a manipulator limb any of its sibling end effectors are no longer aligned with any objects unless it is grasping an object
effect(AL, ROBO, configure(ARM, retracted), configuration(END), aligned_with(nothing), t) :- holds(AL, grasping(END), nothing, t - 1),
                                                                                             action(AL, _, _, ROBO, configure(ARM, retracted)),
                                                                                             fluent(AL, inertial, configuration(END), aligned_with(nothing)),
                                                                                             desce_of(AL, ROBO, ARM),
                                                                                             desce_of(AL, ROBO, END),
                                                                                             siblings(AL, ARM, END),
                                                                                             insta_of(AL, armed_robot, ROBO),
                                                                                             insta_of(AL, extensible, ARM),
                                                                                             insta_of(AL, manipulator_end, END),
                                                                                             state_of(AL, END, aligned_with(nothing)),
                                                                                             AL < reduc, pl(AL).

%% When a robot puts an object on a surface the object is on that surface
effect(AL, ROBO, put(DESC, OBJ, SURFACE), on(OBJ), SURFACE) :- action(AL, _, _, ROBO, put(DESC, OBJ, SURFACE)),
                                                               fluent(AL, inertial, on(OBJ), SURFACE),
                                                               OBJ != SURFACE, SURFACE != nothing,
                                                               desce_of(AL, ROBO, DESC),
                                                               insta_of(AL, armed_robot, ROBO),
                                                               insta_of(AL, grasper, DESC),
                                                               insta_of(AL, placeable, OBJ),
                                                               insta_of(AL, surface, SURFACE),
                                                               pl(AL).

%% When a robot lifts an object the object is on nothing
effect(AL, ROBO, lift(DESC, OBJ), on(OBJ), nothing) :- action(AL, _, _, ROBO, lift(DESC, OBJ)),
                                                       fluent(AL, inertial, on(OBJ), nothing),
                                                       desce_of(AL, ROBO, DESC),
                                                       insta_of(AL, armed_robot, ROBO),
                                                       insta_of(AL, grasper, DESC),
                                                       insta_of(AL, placeable, OBJ),
                                                       pl(AL).

%% When a robot actuates a door its configuration is flipped, if it was closed it becomes open and vice versa
effect(AL, ROBO, actuate(DESC, D), configuration(D), STATE_1, t) :- holds(AL, configuration(D), STATE_2, t - 1),
                                                                    action(AL, _, _, ROBO, actuate(DESC, D)),
                                                                    fluent(AL, inertial, configuration(D), STATE_1),
                                                                    fluent(AL, inertial, configuration(D), STATE_2),
                                                                    insta_of(AL, armed_robot, ROBO),
                                                                    insta_of(AL, grasper, DESC),
                                                                    insta_of(AL, door, D),
                                                                    state_of(AL, D, STATE_1),
                                                                    state_of(AL, D, STATE_2),
                                                                    STATE_1 != STATE_2,
                                                                    AL < reduc, pl(AL).

%% In the redefined model, when a robot collects an object it is transporting that object
effect(AL, ROBO, collect(OBJ), transporting(ROBO, OBJ), true) :- action(AL, _, _, ROBO, collect(OBJ)),
                                                                 fluent(AL, inertial, transporting(ROBO, OBJ), true),
                                                                 insta_of(AL, armed_robot, ROBO),
                                                                 insta_of(AL, graspable, OBJ),
                                                                 AL >= TL, model_type(TL, tasking), pl(AL).

%% In the redefined model, when a robot collects an object the object is on nothing
effect(AL, ROBO, collect(OBJ), on(OBJ), nothing) :- action(AL, _, _, ROBO, collect(OBJ)),
                                                    fluent(AL, inertial, on(OBJ), nothing),
                                                    insta_of(AL, armed_robot, ROBO),
                                                    insta_of(AL, graspable, OBJ),
                                                    AL >= TL, model_type(TL, tasking), pl(AL).

%% In the redefined model, when a robot deposits an object it is no longer transporting that object
effect(AL, ROBO, deposit(OBJ), transporting(ROBO, OBJ), false) :- action(AL, _, _, ROBO, deposit(OBJ)),
                                                                  fluent(AL, inertial, transporting(ROBO, OBJ), false),
                                                                  insta_of(RL, armed_robot, ROBO),
                                                                  insta_of(RL, graspable, OBJ),
                                                                  AL >= TL, model_type(TL, tasking), pl(AL).

%% In the redefined model, when a robot places an object upon a surface the object is on that surface
effect(AL, ROBO, place_upon(OBJ, SURFACE), on(OBJ), SURFACE) :- action(AL, _, _, ROBO, place_upon(OBJ, SURFACE)),
                                                                fluent(AL, inertial, on(OBJ), SURFACE),
                                                                OBJ != SURFACE, SURFACE != nothing,
                                                                insta_of(AL, armed_robot, ROBO),
                                                                insta_of(AL, placeable, OBJ),
                                                                insta_of(AL, surface, SURFACE),
                                                                AL >= TL, model_type(TL, tasking), pl(AL).

%% In the redefined model, when a robot places an object upon a surface
effect(AL, ROBO, place_upon(OBJ, SURFACE), transporting(ROBO, OBJ), false) :- action(AL, _, _, ROBO, place_upon(OBJ, SURFACE)),
                                                                              fluent(AL, inertial, transporting(ROBO, OBJ), false),
                                                                              OBJ != SURFACE, SURFACE != nothing,
                                                                              insta_of(AL, armed_robot, ROBO),
                                                                              insta_of(AL, placeable, OBJ),
                                                                              insta_of(AL, surface, SURFACE),
                                                                              AL >= TL, model_type(TL, tasking), pl(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Agent Action Preconditions - Conditions that must hold to plan certain actions
#program action_preconditions(t).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% A robot can only move between locations that are connected
precond(AL, ROBO, move(L_2), in(ROBO), L_1, false) :- not is(AL, connected(L_1, L_2)), L_1 != L_2,
                                                      action(AL, _, _, ROBO, move(L_2)),
                                                      fluent(AL, inertial, in(ROBO), L_1),
                                                      static(AL, connected(L_1, L_2)),
                                                      insta_of(AL, robot, ROBO),
                                                      insta_of(AL, location, L_1),
                                                      insta_of(AL, location, L_2),
                                                      AL < reduc, pl(AL).

%% Alternative encoding of the previous rule using disjunctive precondition
% { precond(AL, ROBO, move(P_2), in(ROBO), P_1, true, t) : is(AL, connected(P_1, P_2)),
%                                                          fluent(AL, inertial, in(ROBO), P_1),
%                                                          static(AL, connected(P_1, P_2)), P_1 != P_2,
%                                                          insta_of(AL, location, P_1) } = 1 :- action(AL, _, _, ROBO, move(P_2)),
%                                                                                               insta_of(AL, robot, ROBO),
%                                                                                               insta_of(AL, location, P_2),
%                                                                                               AL < reduc, pl(AL).

%% A robot can only move between locations that are connected by a door if that door is open
precond(AL, ROBO, move(L_2), configuration(D), open, true, t) :- holds(AL, in(ROBO), L_1, t - 1),
                                                                 is(AL, connected_by_door(D, L_1, L_2)), L_1 != L_2,
                                                                 action(AL, _, _, ROBO, move(L_2)),
                                                                 fluent(AL, inertial, configuration(D), open),
                                                                 fluent(AL, _, in(ROBO), L_1),
                                                                 static(AL, connected_by_door(D, L_1, L_2)),
                                                                 insta_of(AL, robot, ROBO),
                                                                 insta_of(AL, location, L_1),
                                                                 insta_of(AL, location, L_2),
                                                                 insta_of(AL, door, D),
                                                                 AL < reduc, pl(AL).

%% An armed robot can only open a door if it is grasping at least one of its handles
{ precond(AL, ROBO, actuate(COMP, D), grasping(COMP), H, true, t) : desce_of(AL, D, H),
                                                                    insta_of(AL, handle, H),
                                                                    action(AL, _, _, ROBO, actuate(COMP, D)),
                                                                    fluent(AL, inertial, grasping(COMP), H) } = 1 :- desce_of(AL, ROBO, COMP),
                                                                                                                     insta_of(AL, armed_robot, ROBO),
                                                                                                                     insta_of(AL, grasper, COMP),
                                                                                                                     insta_of(AL, door, D),
                                                                                                                     AL < reduc, pl(AL).

% %% An armed robot can only open a door if it is grasping at least one of its handles
% not occurs(AL, ROBO, actuate(DESC, D), t) :- not { holds(AL, grasping(DESC), H, t - 1) : desce_of(AL, D, H),
%                                                                     insta_of(AL, handle, H),
%                                                                     action(AL, _, _, ROBO, actuate(DESC, D)),
%                                                                     fluent(AL, inertial, grasping(DESC), H) } = 1, desce_of(AL, ROBO, DESC),
%                                                                                                                      insta_of(AL, robot, ROBO),
%                                                                                                                      insta_of(AL, grasper, DESC),
%                                                                                                                      insta_of(AL, door, D),
%                                                                                                                      AL < reduc, pl(AL).

%% An armed robot can only grasp a grounded object when it is in the same location as it
precond(AL, ROBO, grasp(DESC, OBJ), in(ROBO), L, true) :- is(AL, in(OBJ, L)),
                                                          action(AL, _, _, ROBO, grasp(DESC, OBJ)),
                                                          fluent(AL, inertial, in(ROBO), L),
                                                          static(AL, in(OBJ, L)),
                                                          desce_of(AL, ROBO, DESC),
                                                          insta_of(AL, armed_robot, ROBO),
                                                          insta_of(AL, grasper, DESC),
                                                          insta_of(AL, graspable, OBJ), OBJ != nothing,
                                                          insta_of(AL, grounded, OBJ),
                                                          insta_of(AL, location, L),
                                                          AL < reduc, pl(AL).

%% An armed robot can only move if all of its extensible components are retracted
precond(AL, ROBO, move(L), configuration(DESC), retracted, true) :- action(AL, _, _, ROBO, move(L)),
                                                                    fluent(AL, inertial, configuration(DESC), retracted),
                                                                    desce_of(AL, ROBO, DESC),
                                                                    insta_of(AL, armed_robot, ROBO),
                                                                    insta_of(AL, extensible, DESC),
                                                                    insta_of(AL, location, L),
                                                                    state_of(AL, DESC, retracted),
                                                                    AL < reduc, pl(AL).

%% An armed robot can only grasp an object when it is in the same location as that object
precond(AL, ROBO, grasp(DESC, OBJ), in(OBJ), L, true, t) :- holds(AL, in(ROBO), L, t - 1), OBJ != nothing,
                                                            action(AL, _, _, ROBO, grasp(DESC, OBJ)),
                                                            fluent(AL, inertial, in(OBJ), L),
                                                            fluent(AL, inertial, in(ROBO), L),
                                                            desce_of(AL, ROBO, DESC),
                                                            insta_of(AL, armed_robot, ROBO),
                                                            insta_of(AL, grasper, DESC),
                                                            insta_of(AL, graspable, OBJ),
                                                            insta_of(AL, location, L),
                                                            pl(AL).

precond(AL, ROBO, grasp(DESC, OBJ), in(ROBO), L, true, t) :- holds(AL, in(OBJ), L, t - 1), OBJ != nothing,
                                                             action(AL, _, _, ROBO, grasp(DESC, OBJ)),
                                                             fluent(AL, inertial, in(OBJ), L),
                                                             fluent(AL, inertial, in(ROBO), L),
                                                             desce_of(AL, ROBO, DESC),
                                                             insta_of(AL, armed_robot, ROBO),
                                                             insta_of(AL, grasper, DESC),
                                                             insta_of(AL, graspable, OBJ),
                                                             insta_of(AL, location, L),
                                                             pl(AL).

%% An armed robot can only grasp an object with a given grasper if it is not currently grasping another object with that grasper
precond(AL, ROBO, grasp(DESC, OBJ), grasping(DESC), nothing, true) :- OBJ != nothing,
                                                                      action(AL, _, _, ROBO, grasp(DESC, OBJ)),
                                                                      fluent(AL, inertial, grasping(DESC), nothing),
                                                                      desce_of(AL, ROBO, DESC),
                                                                      insta_of(AL, armed_robot, ROBO),
                                                                      insta_of(AL, grasper, DESC),
                                                                      insta_of(AL, graspable, OBJ),
                                                                      pl(AL).

%% An armed robot cannot grasp an object that has another object on top of it
precond(AL, ROBO, grasp(DESC, OBJ_1), on(OBJ_2), OBJ_1, false) :- OBJ_1 != OBJ_2, OBJ_1 != nothing,
                                                                  action(AL, _, _, ROBO, grasp(DESC, OBJ_1)),
                                                                  fluent(AL, inertial, on(OBJ_2), OBJ_1),
                                                                  desce_of(AL, ROBO, DESC),
                                                                  insta_of(AL, armed_robot, ROBO),
                                                                  insta_of(AL, grasper, DESC),
                                                                  insta_of(AL, graspable, OBJ_1),
                                                                  insta_of(AL, surface, OBJ_1),
                                                                  insta_of(AL, placeable, OBJ_2),
                                                                  pl(AL).

%% An armed robot can only grasp an object with a grasper that is extensible if it is extended
precond(AL, ROBO, grasp(DESC, OBJ), configuration(DESC), extended, true) :- OBJ != nothing,
                                                                            action(AL, _, _, ROBO, grasp(DESC, OBJ)),
                                                                            fluent(AL, inertial, configuration(DESC), extended),
                                                                            desce_of(AL, ROBO, DESC),
                                                                            insta_of(AL, armed_robot, ROBO),
                                                                            insta_of(AL, grasper, DESC),
                                                                            insta_of(AL, extensible, DESC),
                                                                            insta_of(AL, graspable, OBJ),
                                                                            AL < reduc, pl(AL).

%% An armed robot can only grasp an object with a grasper that is not extensible but has extensible siblings is those siblings are extended
precond(AL, ROBO, grasp(DESC_1, OBJ), configuration(DESC_2), extended, true) :- OBJ != nothing,
                                                                                action(AL, _, _, ROBO, grasp(DESC_1, OBJ)),
                                                                                fluent(AL, inertial, configuration(DESC_2), extended),
                                                                                desce_of(AL, ROBO, DESC_1),              
                                                                                desce_of(AL, ROBO, DESC_2),
                                                                                siblings(AL, DESC_1, DESC_2),
                                                                                insta_of(AL, armed_robot, ROBO),
                                                                                insta_of(AL, grasper, DESC_1),
                                                                                not insta_of(AL, extensible, DESC_1),
                                                                                insta_of(AL, extensible, DESC_2),
                                                                                insta_of(AL, graspable, OBJ),
                                                                                AL < reduc, pl(AL).

%% An armed robot can only grasp an object with an end effector if that end effector is aligned with that object
precond(AL, ROBO, grasp(END, OBJ), configuration(END), aligned_with(OBJ), true) :- OBJ != nothing,
                                                                                   action(AL, _, _, ROBO, grasp(END, OBJ)),
                                                                                   fluent(AL, inertial, configuration(END), aligned_with(OBJ)),
                                                                                   desce_of(AL, ROBO, END),
                                                                                   insta_of(AL, armed_robot, ROBO),
                                                                                   insta_of(AL, manipulator_end, END),
                                                                                   insta_of(AL, graspable, OBJ),
                                                                                   state_of(AL, END, aligned_with(OBJ)),
                                                                                   AL < reduc, pl(AL).

%% An armed robot can only release an object when it is grasping that object
precond(AL, ROBO, release(DESC, OBJ), grasping(DESC), OBJ, true) :- OBJ != nothing,
                                                                    action(AL, _, _, ROBO, release(DESC, OBJ)),
                                                                    fluent(AL, inertial, grasping(DESC), OBJ),
                                                                    desce_of(AL, ROBO, DESC),
                                                                    insta_of(AL, armed_robot, ROBO),
                                                                    insta_of(AL, grasper, DESC),
                                                                    insta_of(AL, graspable, OBJ),
                                                                    pl(AL).

%% An armed robot can only release an object from a grasper that is extensible if it is extended
precond(AL, ROBO, release(DESC, OBJ), configuration(DESC), extended, true) :- OBJ != nothing,
                                                                              action(AL, _, _, ROBO, release(DESC, OBJ)),
                                                                              fluent(AL, inertial, configuration(DESC), extended),
                                                                              desce_of(AL, ROBO, DESC),
                                                                              insta_of(AL, armed_robot, ROBO),
                                                                              insta_of(AL, grasper, DESC),
                                                                              insta_of(AL, graspable, OBJ),
                                                                              state_of(AL, DESC, extended),
                                                                              pl(AL), t > 0.

%% An armed robot can only release an object from a grasper that is not extensible but has extensible siblings is those siblings are extended
precond(AL, ROBO, release(DESC_1, OBJ), configuration(DESC_2), extended, true) :- OBJ != nothing,
                                                                                  action(AL, _, _, ROBO, release(DESC_1, OBJ)),
                                                                                  fluent(AL, inertial, configuration(DESC_2), extended),
                                                                                  desce_of(AL, ROBO, DESC_1),
                                                                                  desce_of(AL, ROBO, DESC_2),
                                                                                  siblings(AL, DESC_1, DESC_2),
                                                                                  insta_of(AL, armed_robot, ROBO),
                                                                                  insta_of(AL, grasper, DESC_1),
                                                                                  not insta_of(AL, extensible, DESC_1),
                                                                                  insta_of(AL, extensible, DESC_2),
                                                                                  insta_of(AL, graspable, OBJ),
                                                                                  pl(AL).

%% An armed robot can only change the configuration of the descendant components of its manipulator arms if the extensible descendant sibling components of that arm are extended
precond(AL, ROBO, configure(DESC_1, STATE), configuration(DESC_2), extended, true) :- action(AL, _, _, ROBO, configure(DESC_1, STATE)),
                                                                                      fluent(AL, inertial, configuration(DESC_2), extended),
                                                                                      desce_of(AL, ROBO, DESC_1),
                                                                                      desce_of(AL, ROBO, DESC_2),
                                                                                      desce_of(AL, ROBO, ARM),
                                                                                      child_of(AL, ARM, DESC_1),
                                                                                      child_of(AL, ARM, DESC_2),
                                                                                      siblings(AL, DESC_1, DESC_2),
                                                                                      insta_of(AL, armed_robot, ROBO),
                                                                                      insta_of(AL, configurable, DESC_1),
                                                                                      insta_of(AL, extensible, DESC_2),
                                                                                      insta_of(AL, manipulator_arm, ARM),
                                                                                      state_of(AL, DESC_1, STATE),
                                                                                      AL < reduc, pl(AL).

%% An armed robot can only align an end effector with an object if it is grasping nothing with that end effector
precond(AL, ROBO, configure(DESC, aligned_with(OBJ)), grasping(DESC), nothing, true) :- action(AL, _, _, ROBO, configure(DESC, aligned_with(OBJ))),
                                                                                        fluent(AL, inertial, grasping(DESC), nothing),
                                                                                        desce_of(AL, ROBO, DESC),
                                                                                        insta_of(AL, armed_robot, ROBO),
                                                                                        insta_of(AL, manipulator_end, DESC),
                                                                                        insta_of(AL, graspable, OBJ), OBJ != nothing,
                                                                                        state_of(AL, DESC, aligned_with(OBJ)),
                                                                                        AL < reduc, pl(AL).

% An armed robot can only put an object on a surface when it is in the same location as that surface
precond(AL, ROBO, put(DESC, OBJ, SURFACE), in(SURFACE), L, true, t) :- holds(AL, in(ROBO), L, t - 1),
                                                                       action(AL, _, _, ROBO, put(DESC, OBJ, SURFACE)),
                                                                       fluent(AL, inertial, in(SURFACE), L),
                                                                       fluent(AL, inertial, in(ROBO), L),
                                                                       desce_of(AL, ROBO, DESC),
                                                                       insta_of(AL, armed_robot, ROBO),
                                                                       insta_of(AL, grasper, DESC),
                                                                       insta_of(AL, placeable, OBJ),
                                                                       insta_of(AL, surface, SURFACE),
                                                                       insta_of(AL, location, L),
                                                                       pl(AL).

precond(AL, ROBO, put(DESC, OBJ, SURFACE), in(ROBO), L, true, t) :- holds(AL, in(SURFACE), L, t - 1),
                                                                    action(AL, _, _, ROBO, put(DESC, OBJ, SURFACE)),
                                                                    fluent(AL, inertial, in(SURFACE), L),
                                                                    fluent(AL, inertial, in(ROBO), L),
                                                                    desce_of(AL, ROBO, DESC),
                                                                    insta_of(AL, armed_robot, ROBO),
                                                                    insta_of(AL, grasper, DESC),
                                                                    insta_of(AL, placeable, OBJ),
                                                                    insta_of(AL, surface, SURFACE),
                                                                    insta_of(AL, location, L),
                                                                    pl(AL).

%% An armed robot cannot put an object on top of another object that already has some other object on top of it
precond(AL, ROBO, put(DESC, OBJ_1, BLOCK), on(OBJ_2), BLOCK, false) :- OBJ_1 != OBJ_2,
                                                                       action(AL, _, _, ROBO, put(DESC, OBJ_1, SURFACE)),
                                                                       fluent(AL, inertial, on(OBJ_2), BLOCK),
                                                                       desce_of(AL, ROBO, DESC),
                                                                       insta_of(AL, armed_robot, ROBO),
                                                                       insta_of(AL, grasper, DESC),
                                                                       insta_of(AL, placeable, OBJ_1),
                                                                       insta_of(AL, placeable, OBJ_2),
                                                                       insta_of(AL, block, BLOCK),
                                                                       pl(AL).

%% An armed robot can only put an object on a surface when it is grasping that object
precond(AL, ROBO, put(DESC, OBJ, SURFACE), grasping(DESC), OBJ, true) :- OBJ != SURFACE, OBJ != nothing, SURFACE != nothing,
                                                                         action(AL, _, _, ROBO, put(DESC, OBJ, SURFACE)),
                                                                         fluent(AL, inertial, grasping(DESC), OBJ),
                                                                         desce_of(AL, ROBO, DESC),
                                                                         insta_of(AL, armed_robot, ROBO),
                                                                         insta_of(AL, grasper, DESC),
                                                                         insta_of(AL, placeable, OBJ),
                                                                         insta_of(AL, surface, SURFACE),
                                                                         pl(AL).

%% An armed robot cannot put an object on another object if any robot is grasping the other object with any of its graspers
precond(AL, ROBO_1, put(DESC_1, OBJ_1, OBJ_2), grasping(DESC_2), OBJ_2, false) :- OBJ_1 != OBJ_2, DESC_1 != DESC_2,
                                                                                  action(AL, _, _, ROBO, put(DESC_1, OBJ_1, OBJ_2)),
                                                                                  fluent(AL, inertial, grasping(DESC_2), OBJ_2),
                                                                                  desce_of(AL, ROBO_1, DESC_1),
                                                                                  desce_of(AL, ROBO_2, DESC_2),
                                                                                  insta_of(AL, armed_robot, ROBO_1),
                                                                                  insta_of(AL, armed_robot, ROBO_2),
                                                                                  insta_of(AL, grasper, DESC_1),
                                                                                  insta_of(AL, grasper, DESC_2),
                                                                                  insta_of(AL, placeable, OBJ_1),
                                                                                  insta_of(AL, surface, OBJ_2),
                                                                                  insta_of(AL, graspable, OBJ_2),
                                                                                  pl(AL).

%% An armed robot can only put an object on a surface with a grasper that is extensible if it is extended
precond(AL, ROBO, put(DESC, OBJ, SURFACE), configuration(DESC), extended, true) :- action(AL, _, _, ROBO, put(DESC, OBJ, SURFACE)),
                                                                                   fluent(AL, inertial, configuration(DESC), extended),
                                                                                   desce_of(AL, ROBO, DESC),
                                                                                   insta_of(AL, armed_robot, ROBO),
                                                                                   insta_of(AL, grasper, DESC),
                                                                                   insta_of(AL, extensible, DESC),
                                                                                   insta_of(AL, graspable, OBJ),
                                                                                   insta_of(AL, surface, SURFACE),
                                                                                   AL < reduc, pl(AL).

%% An armed robot can only put an object on aa surface with a grasper that is not extensible but has extensible siblings is those siblings are extended
precond(AL, ROBO, put(DESC_1, OBJ, SURFACE), configuration(DESC_2), extended, true) :- action(AL, _, _, ROBO, put(DESC_1, OBJ, SURFACE)),
                                                                                       fluent(AL, inertial, configuration(DESC_2), extended),
                                                                                       desce_of(AL, ROBO, DESC_1),
                                                                                       desce_of(AL, ROBO, DESC_2),
                                                                                       siblings(AL, DESC_1, DESC_2),
                                                                                       insta_of(AL, armed_robot, ROBO),
                                                                                       insta_of(AL, grasper, DESC_1),
                                                                                       not insta_of(AL, extensible, DESC_1),
                                                                                       insta_of(AL, extensible, DESC_2),
                                                                                       insta_of(AL, graspable, OBJ),
                                                                                       insta_of(AL, surface, SURFACE),
                                                                                       AL < reduc, pl(AL).

%% An armed robot cannot lift a grasped object that has another object on top of it
precond(AL, ROBO, lift(DESC, OBJ_1), on(OBJ_2), OBJ_1, false) :- OBJ_1 != OBJ_2,
                                                                 action(AL, _, _, ROBO, lift(DESC, OBJ_1)),
                                                                 fluent(AL, inertial, on(OBJ_2), OBJ_1),
                                                                 desce_of(AL, ROBO, DESC),
                                                                 insta_of(AL, armed_robot, ROBO),
                                                                 insta_of(AL, grasper, DESC),
                                                                 insta_of(AL, placeable, OBJ_1),
                                                                 insta_of(AL, surface, OBJ_1),
                                                                 insta_of(AL, placeable, OBJ_2),
                                                                 pl(AL).

%% An armed robot can only lift a block that it is grasping
precond(AL, ROBO, lift(DESC, OBJ), grasping(DESC), OBJ, true) :- OBJ != SURFACE,
                                                                 action(AL, _, _, ROBO, lift(DESC, OBJ)),
                                                                 fluent(AL, inertial, grasping(DESC), OBJ),
                                                                 desce_of(AL, ROBO, DESC),
                                                                 insta_of(AL, armed_robot, ROBO),
                                                                 insta_of(AL, grasper, DESC),
                                                                 insta_of(AL, graspable, OBJ), OBJ != nothing,
                                                                 insta_of(AL, surface, SURFACE),
                                                                 pl(AL).

%% An armed robot can only collect an object when it is in the same location as that object
precond(AL, ROBO, collect(OBJ), in(OBJ), L, true, t) :- holds(AL, in(ROBO), L, t - 1), OBJ != nothing,
                                                        action(AL, _, _, ROBO, collect(OBJ)),
                                                        fluent(AL, inertial, in(ROBO), L),
                                                        fluent(AL, inertial, in(OBJ), L),
                                                        insta_of(AL, armed_robot, ROBO),
                                                        insta_of(AL, graspable, OBJ),
                                                        insta_of(AL, location, L),
                                                        AL >= TL, model_type(TL, tasking), pl(AL).

precond(AL, ROBO, collect(OBJ), in(ROBO), L, true, t) :- holds(AL, in(OBJ), L, t - 1), OBJ != nothing,
                                                         action(AL, _, _, ROBO, collect(OBJ)),
                                                         fluent(AL, inertial, in(ROBO), L),
                                                         fluent(AL, inertial, in(OBJ), L),
                                                         insta_of(AL, armed_robot, ROBO),
                                                         insta_of(AL, graspable, OBJ),
                                                         insta_of(AL, location, L),
                                                         AL >= TL, model_type(TL, tasking), pl(AL).

%% An armed robot cannot collect an object that has another object on top of it
precond(AL, ROBO, collect(OBJ_1), on(OBJ_2), OBJ_1, false) :- OBJ_1 != OBJ_2,
                                                              action(AL, _, _, ROBO, collect(OBJ_1)),
                                                              fluent(AL, inertial, on(OBJ_2), OBJ_1),
                                                              insta_of(AL, armed_robot, ROBO),
                                                              insta_of(AL, placeable, OBJ_1),
                                                              insta_of(AL, surface, OBJ_1),
                                                              insta_of(AL, surface, SURFACE),
                                                              insta_of(AL, placeable, OBJ_2),
                                                              AL >= TL, model_type(TL, tasking), pl(AL).

%% An armed robot can only deposit an object when it is transporting that object
precond(AL, ROBO, deposit(OBJ), transporting(ROBO, OBJ), true, true) :- OBJ != nothing,
                                                                        action(AL, _, _, ROBO, deposit(OBJ)),
                                                                        fluent(AL, inertial, transporting(ROBO, OBJ), true),
                                                                        insta_of(AL, armed_robot, ROBO),
                                                                        insta_of(AL, graspable, OBJ),
                                                                        AL >= TL, model_type(TL, tasking), pl(AL).

%% An armed robot can only place an object upon a surface when it is in the same location as both the object and the surface
precond(AL, ROBO, place_upon(OBJ, SURFACE), in(SURFACE), L, true, t) :- holds(AL, in(ROBO), L, t - 1),
                                                                        holds(AL, in(OBJ), L, t - 1),
                                                                        action(AL, _, _, ROBO, place_upon(OBJ, SURFACE)),
                                                                        fluent(AL, inertial, in(SURFACE), L),
                                                                        fluent(AL, inertial, in(ROBO), L),
                                                                        fluent(AL, inertial, in(OBJ), L),
                                                                        insta_of(AL, armed_robot, ROBO),
                                                                        insta_of(AL, placeable, OBJ),
                                                                        insta_of(AL, surface, SURFACE),
                                                                        insta_of(AL, location, L),
                                                                        AL >= TL, model_type(TL, tasking), pl(AL).

precond(AL, ROBO, place_upon(OBJ, SURFACE), in(OBJ), L, true, t) :- holds(AL, in(ROBO), L, t - 1),
                                                                    holds(AL, in(SURFACE), L, t - 1),
                                                                    action(AL, _, _, ROBO, place_upon(OBJ, SURFACE)),
                                                                    fluent(AL, inertial, in(SURFACE), L),
                                                                    fluent(AL, inertial, in(ROBO), L),
                                                                    fluent(AL, inertial, in(OBJ), L),
                                                                    insta_of(AL, armed_robot, ROBO),
                                                                    insta_of(AL, placeable, OBJ),
                                                                    insta_of(AL, surface, SURFACE),
                                                                    insta_of(AL, location, L),
                                                                    AL >= TL, model_type(TL, tasking), pl(AL).

precond(AL, ROBO, place_upon(OBJ, SURFACE), in(ROBO), L, true, t) :- holds(AL, in(SURFACE), L, t - 1),
                                                                     holds(AL, in(OBJ), L, t - 1),
                                                                     action(AL, _, _, ROBO, place_upon(OBJ, SURFACE)),
                                                                     fluent(AL, inertial, in(SURFACE), L),
                                                                     fluent(AL, inertial, in(ROBO), L),
                                                                     fluent(AL, inertial, in(OBJ), L),
                                                                     insta_of(AL, armed_robot, ROBO),
                                                                     insta_of(AL, placeable, OBJ),
                                                                     insta_of(AL, surface, SURFACE),
                                                                     insta_of(AL, location, L),
                                                                     AL >= TL, model_type(TL, tasking), pl(AL).

%% An armed robot cannot place an object upon a surface if there is another object on top of it
precond(AL, ROBO, place_upon(OBJ_1, SURFACE), on(OBJ_2), OBJ_1, false) :- OBJ_1 != OBJ_2,
                                                                          action(AL, _, _, ROBO, place_upon(OBJ_1, SURFACE)),
                                                                          fluent(AL, inertial, on(OBJ_2), OBJ_1),
                                                                          insta_of(AL, armed_robot, ROBO),
                                                                          insta_of(AL, placeable, OBJ_1),
                                                                          insta_of(AL, surface, OBJ_1),
                                                                          insta_of(AL, surface, SURFACE),
                                                                          insta_of(AL, placeable, OBJ_2),
                                                                          AL >= TL, model_type(TL, tasking), pl(AL).

%% An armed robot cannot place an object upon another object that is on nothing
precond(AL, ROBO, place_upon(OBJ_1, OBJ_2), on(OBJ_2), nothing, false) :- OBJ_1 != OBJ_2,
                                                                          action(AL, _, _, ROBO, place_upon(OBJ, SURFACE)),
                                                                          fluent(AL, inertial, on(OBJ_2), nothing),
                                                                          insta_of(AL, armed_robot, ROBO),
                                                                          insta_of(AL, placeable, OBJ_1),
                                                                          insta_of(AL, placeable, OBJ_2),
                                                                          insta_of(AL, surface, OBJ_2),
                                                                          AL >= TL, model_type(TL, tasking), pl(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**% #program variable_relations(t).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% State Variable Relations and Validity Constraints - Rules constraining the range of valid states
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% If any of a robot is grasping an object then that object must share its location
:- holds(AL, grasping(DESC), OBJ, t),
   holds(AL, in(ROBO), L, t),
   not holds(AL, in(OBJ), L, t) : not insta_of(AL, grounded, OBJ),
                                  fluent(AL, inertial, in(OBJ), L);
   not is(AL, in(OBJ, L)) : insta_of(AL, grounded, OBJ),
                            static(AL, in(OBJ, L));
   fluent(AL, inertial, grasping(DESC), OBJ),
   fluent(AL, inertial, in(ROBO), L),
   desce_of(AL, ROBO, DESC),
   insta_of(AL, robot, ROBO),
   insta_of(AL, grasper, DESC),
   insta_of(AL, graspable, OBJ), OBJ != nothing,
   insta_of(AL, location, L),
   sl(AL).

%% If any of a robot's end effectors are aligned with an object then that object must share its location
:- holds(AL, configuration(END), aligned_with(OBJ), t),
   holds(AL, in(ROBO), L, t),
   not holds(AL, in(OBJ), L, t) : not insta_of(AL, grounded, OBJ),
                                  fluent(AL, inertial, in(OBJ), L);
   not is(AL, in(OBJ, L)) : insta_of(AL, grounded, OBJ),
                            static(AL, in(OBJ, L));
   fluent(AL, inertial, configuration(END), aligned_with(OBJ)),
   fluent(AL, inertial, in(ROBO), L),
   desce_of(AL, ROBO, END),
   insta_of(AL, robot, ROBO),
   insta_of(AL, manipulator_end, END),
   insta_of(AL, graspable, OBJ), OBJ != nothing,
   insta_of(AL, location, L),
   state_of(AL, END, aligned_with(OBJ)),
   sl(AL).

%% If a robot is grasping an object with an end effector then that end effector must be aligned with the object
:- holds(AL, grasping(END), OBJ, t),
   not holds(AL, configuration(END), aligned_with(OBJ), t),
   fluent(AL, inertial, grasping(END), OBJ),
   fluent(AL, inertial, configuration(END), aligned_with(OBJ)),
   insta_of(AL, manipulator_end, END),
   insta_of(AL, graspable, OBJ), OBJ != nothing,
   state_of(AL, END, aligned_with(OBJ)),
   sl(AL).

%% A robot cannot be grasping an object that has another object on top of it
:- holds(AL, grasping(DESC), OBJ_1, t),
   holds(AL, on(OBJ_2), OBJ_1, t),
   fluent(AL, inertial, grasping(DESC), OBJ_1),
   fluent(AL, inertial, on(OBJ_2), OBJ_1),
   desce_of(AL, ROBO, DESC),
   insta_of(AL, robot, ROBO),
   insta_of(AL, grasper, DESC),
   insta_of(AL, graspable, OBJ_1), OBJ_1 != nothing,
   insta_of(AL, surface, OBJ_1),
   insta_of(AL, placeable, OBJ_2),
   sl(AL).

%% If an object is on a surface then that object shares the location of the surface
:- holds(AL, on(OBJ), SURFACE, t),
   holds(AL, in(SURFACE), L, t),
   not holds(AL, in(OBJ), L, t),
   fluent(AL, inertial, on(OBJ), SURFACE),
   fluent(AL, inertial, in(SURFACE), L),
   fluent(AL, inertial, in(OBJ), L),
   insta_of(AL, location, L),
   insta_of(AL, location, L),
   insta_of(AL, surface, SURFACE),
   insta_of(AL, object, OBJ),
   sl(AL).

%% A block cannot be on a block that is on nothing
:- holds(AL, on(BLOCK_1), BLOCK_2, t),
   holds(AL, on(BLOCK_2), nothing, t),
   fluent(AL, inertial, on(BLOCK_1), BLOCK_2),
   fluent(AL, inertial, on(BLOCK_2), nothing),
   insta_of(AL, block, BLOCK_1),
   insta_of(AL, block, BLOCK_2),
   sl(AL).

%% Two blocks cannot be on top of the same block
:- holds(AL, on(BLOCK_1), BLOCK_3, t),
   holds(AL, on(BLOCK_2), BLOCK_3, t),
   fluent(AL, inertial, on(BLOCK_1), BLOCK_3),
   fluent(AL, inertial, on(BLOCK_2), BLOCK_3),
   BLOCK_1 != BLOCK_2,
   insta_of(AL, block, BLOCK_1),
   insta_of(AL, block, BLOCK_2),
   insta_of(AL, block, BLOCK_3),
   sl(AL).

%% A robot cannot grasp an object with two graspers simultaneously
:- holds(AL, grasping(DESC_1), OBJ, t),
   holds(AL, grasping(DESC_2), OBJ, t),
   fluent(AL, inertial, grasping(DESC_1), OBJ),
   fluent(AL, inertial, grasping(DESC_2), OBJ),
   DESC_1 != DESC_2,
   desce_of(AL, ROBO, DESC_1),
   desce_of(AL, ROBO, DESC_2),
   insta_of(AL, robot, ROBO),
   insta_of(AL, grasper, DESC_1),
   insta_of(AL, grasper, DESC_2),
   insta_of(AL, graspable, OBJ), OBJ != nothing,
   sl(AL).

%% A robot cannot transport more objects than it has manipulator arms
:- N = #count { OBJ : holds(AL_1, transporting(ROBO, OBJ), true, t),
                      fluent(AL_1, inertial, transporting(ROBO, OBJ), true),
                      insta_of(AL_1, graspable, OBJ) },
   M = #count { ARM : desce_of(AL_2, ROBO, ARM),
                      insta_of(AL_2, manipulator_arm, ARM),
                      al(AL_2), AL_1 >= AL_2 },
   N > M, block_number(N), block_number(M),
   insta_of(AL_1, robot, ROBO),
   AL_1 >= TL, model_type(TL, tasking), sl(AL_1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fluent Definitions - Rules defining the truth of defined fluents
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% A block is the base of a tower if it is either;
%%      - On a table,
%%      - On a descenant surface of a table.
holds(AL, tower_base(BASE, TABLE), true, t) :- holds(AL, on(BASE), TABLE, t),
                                        fluent(AL, defined, tower_base(BASE, TABLE), true),
                                        fluent(AL, inertial, on(BASE), TABLE),
                                        insta_of(AL, block, BASE),
                                        insta_of(AL, table, TABLE),
                                        sl(AL).

holds(AL, tower_base(BASE, TABLE), true, t) :- holds(AL, on(BASE), SURFACE, t),
                                        fluent(AL, defined, tower_base(BASE, TABLE), true),
                                        fluent(AL, inertial, on(BASE), SURFACE),
                                        desce_of(AL, TABLE, SURFACE),
                                        insta_of(AL, block, BASE),
                                        insta_of(AL, table, TABLE),
                                        insta_of(AL, surface, SURFACE),
                                        sl(AL).

%% A block is in a tower if it is either;
%%      - The base of the tower,
%%      - On a block that is in the tower.
holds(AL, in_tower(BASE, BASE), true, t) :- holds(AL, tower_base(BASE, TABLE), true, t),
                                            fluent(AL, defined, in_tower(BASE, BASE), true),
                                            fluent(AL, defined, tower_base(BASE, TABLE), true),
                                            insta_of(AL, block, BASE),
                                            insta_of(AL, table, TABLE),
                                            sl(AL).

holds(AL, in_tower(BLOCK_1, BLOCK_2), true, t) :- holds(AL, on(BLOCK_2), BLOCK_3, t),
                                                  holds(AL, in_tower(BLOCK_1, BLOCK_3), true, t),
                                                  fluent(AL, defined, in_tower(BLOCK_1, BLOCK_2), true),
                                                  fluent(AL, defined, in_tower(BLOCK_1, BLOCK_3), true),
                                                  fluent(AL, inertial, on(BLOCK_2), BLOCK_3),
                                                  BLOCK_1 != BLOCK_2,
                                                  insta_of(AL, block, BLOCK_1),
                                                  insta_of(AL, block, BLOCK_2),
                                                  insta_of(AL, block, BLOCK_3),
                                                  sl(AL).

%% A tower that is not stacked in descending order from top to bottom is unordered
holds(AL, unordered_tower(BASE, TABLE), true, t) :- holds(AL, on(BLOCK_1), BLOCK_2, t),
                                                    holds(AL, in_tower(BASE, BLOCK_1), true, t),
                                                    holds(AL, in_tower(BASE, BLOCK_2), true, t),
                                                    holds(AL, tower_base(BASE, TABLE), true, t),
                                                    fluent(AL, defined, unordered_tower(BASE, TABLE), true),
                                                    fluent(AL, defined, in_tower(BASE, BLOCK_1), true),
                                                    fluent(AL, defined, in_tower(BASE, BLOCK_2), true),
                                                    fluent(AL, defined, tower_base(BASE, TABLE), true),
                                                    fluent(AL, inertial, on(BLOCK_1), BLOCK_2),
                                                    BLOCK_1 = (block, N_1), block_number(N_1),
                                                    BLOCK_2 = (block, N_2), block_number(N_2),
                                                    BLOCK_1 != BLOCK_2, N_1 > N_2,
                                                    insta_of(AL, block, BLOCK_1),
                                                    insta_of(AL, block, BLOCK_2),
                                                    insta_of(AL, block, BASE),
                                                    insta_of(AL, table, TABLE),
                                                    sl(AL).

%% A tower is complete if it contains of the blocks of a particular colour and they are stacked in descending order from top to bottom
holds(AL, complete_tower(COLOUR, TABLE), true, t) :- N = #count { BLOCK : holds(AL, in_tower(BASE, BLOCK), true, t),
                                                                          is(AL, colour_of(BLOCK, COLOUR)),
                                                                          fluent(AL, defined, in_tower(BASE, BLOCK), true),
                                                                          static(AL, colour_of(BLOCK, COLOUR)),
                                                                          insta_of(AL, block, BLOCK) },
                                                     blocks_per_tower(N),
                                                     holds(AL, unordered_tower(BASE, TABLE), false, t),
                                                     holds(AL, tower_base(BASE, TABLE), true, t),
                                                     fluent(AL, defined, complete_tower(COLOUR, TABLE), true),
                                                     fluent(AL, defined, unordered_tower(BASE, TABLE), true),
                                                     fluent(AL, defined, tower_base(BASE, TABLE), true),
                                                     is(AL, colour_of(BASE, COLOUR)),
                                                     static(AL, colour_of(BASE, COLOUR)),
                                                     insta_of(AL, colour, COLOUR),
                                                     insta_of(AL, block, BASE),
                                                     insta_of(AL, table, TABLE),
                                                     sl(AL).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Static Abstraction Mappings - The mappings between the static state represention at adjacent abstraction levels are defined here
#program abstraction_mappings(t).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Abstraction rule for connectedness of locations that have descendants
is(AL_1, connected(L_1, L_2)) :- is(AL_2, connected(L_1_C, L_2_C)),
                                 static(AL_1, connected(L_1, L_2)),
                                 static(AL_2, connected(L_1_C, L_2_C)),
                                 L_1 != L_2,
                                 desce_of(AL_2, L_1, L_1_C),
                                 desce_of(AL_2, L_2, L_2_C),
                                 insta_of(AL_1, location, L_1),
                                 insta_of(AL_1, location, L_2),
                                 insta_of(AL_2, location, L_1_C),
                                 insta_of(AL_2, location, L_2_C),
                                 al(AL_1), al(AL_2), AL_1 >= AL_2.

%% Abstraction rule for connectedness of locations that do not have descendants
is(AL_1, connected(L_1, L_2)) :- is(AL_2, connected(L_1, L_2)),
                                 static(AL_1, connected(L_1, L_2)),
                                 static(AL_2, connected(L_1, L_2)),
                                 L_1 != L_2,
                                 insta_of(AL_1, location, L_1),
                                 insta_of(AL_1, location, L_2),
                                 insta_of(AL_2, location, L_1),
                                 insta_of(AL_2, location, L_2),
                                 al(AL_1), al(AL_2), AL_1 >= AL_2.

%% Abstraction rule for locations that are connected by doors
is(AL_1, connected_by_door(D, L_1, L_2)) :- is(AL_2, connected_by_door(D, L_1_C, L_2_C)),
                                            static(AL_1, connected_by_door(D, L_1, L_2)),
                                            static(AL_2, connected_by_door(D, L_1_C, L_2_C)),
                                            L_1 != L_2,
                                            desce_of(AL_2, L_1, L_1_C),
                                            desce_of(AL_2, L_2, L_2_C),
                                            insta_of(AL_1, door, D),
                                            insta_of(AL_2, door, D),
                                            insta_of(AL_1, location, L_1),
                                            insta_of(AL_1, location, L_2),
                                            insta_of(AL_2, location, L_1_C),
                                            insta_of(AL_2, location, L_2_C),
                                            al(AL_1), al(AL_2), AL_1 >= AL_2.

%% Abstraction rule for the location of grounded objects
is(AL_1, in(OBJ, L_1)) :- is(AL_2, in(OBJ, L_2)),
                          static(AL_1, in(OBJ, L_1)),
                          static(AL_2, in(OBJ, L_2)),
                          desce_of(AL_2, L_1, L_2),
                          insta_of(AL_1, location, L_1),
                          insta_of(AL_1, object, OBJ),
                          insta_of(AL_1, grounded, OBJ),
                          insta_of(AL_1, location, L_1),
                          insta_of(AL_2, location, L_2),
                          al(AL_1), al(AL_2), AL_1 >= AL_2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fluent Abstraction Mappings - The mappings between the dynamic state represention at adjacent abstraction levels are defined here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% If an object is in a location that is a descendant of some ancester location then that
%% object is also located in the ancestor location at the current and higher abstractions
holds(AL_1, in(OBJ), L_1, t) :- holds(AL_2, in(OBJ), L_2, t),
                                fluent(AL_1, inertial, in(OBJ), L_1),
                                fluent(AL_2, inertial, in(OBJ), L_2),
                                desce_of(AL_2, L_1, L_2),
                                insta_of(AL_1, location, L_1),
                                insta_of(AL_2, location, L_2),
                                insta_of(AL_3, object, OBJ),
                                sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                                sl(AL_3), AL_3 >= AL_1.

%% If an object is in a location then it is in that location at the current and higher abstractions
holds(AL_1, in(OBJ), L, t) :- holds(AL_2, in(OBJ), L, t),
                              fluent(AL_1, inertial, in(OBJ), L),
                              fluent(AL_2, inertial, in(OBJ), L),
                              insta_of(AL_1, location, L),
                              insta_of(AL_2, location, L),
                              insta_of(AL_3, object, OBJ),
                              sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                              sl(AL_3), AL_3 >= AL_1.

%% If an object has any descendants then those descendants share the location of the object at the current and higher abstractions
holds(AL_1, in(DESC), L, t) :- holds(AL_2, in(OBJ), L, t),
                               fluent(AL_1, inertial, in(DESC), L),
                               fluent(AL_2, inertial, in(OBJ), L),
                               desce_of(AL_2, OBJ, DESC),
                               insta_of(AL_1, location, L),
                               insta_of(AL_2, location, L),
                               insta_of(AL_3, object, OBJ),
                               insta_of(AL_4, _, DESC),
                               sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                               sl(AL_3), AL_3 >= AL_2,
                               sl(AL_4), AL_4 >= AL_1.

%% If an object has any descendants then the object shares the location of its descendants at the current and higher abstractions
holds(AL_1, in(OBJ), L, t) :- holds(AL_2, in(DESC), L, t),
                              fluent(AL_1, inertial, in(OBJ), L),
                              fluent(AL_2, inertial, in(DESC), L),
                              desce_of(AL_2, OBJ, DESC),
                              insta_of(AL_1, location, L),
                              insta_of(AL_2, location, L),
                              insta_of(AL_3, object, OBJ),
                              insta_of(AL_4, _, DESC),
                              sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                              sl(AL_3), AL_3 >= AL_1,
                              sl(AL_4), AL_4 >= AL_2.

%% If a robot is grasping an object then that object shares the location of the robot at the current and higher abstractions
holds(AL_1, in(OBJ), L, t) :- holds(AL_2, in(ROBO), L, t),
                              holds(AL_3, grasping(DESC), OBJ, t),
                              fluent(AL_1, inertial, in(OBJ), L),
                              fluent(AL_2, inertial, in(ROBO), L),
                              fluent(AL_3, inertial, grasping(DESC), OBJ),
                              desce_of(AL_2, ROBO, DESC),
                              insta_of(AL_1, location, L),
                              insta_of(AL_2, location, L),
                              insta_of(AL_3, graspable, OBJ), OBJ != nothing,
                              insta_of(AL_4, grasper, DESC),
                              insta_of(AL_5, robot, ROBO),
                              sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                              sl(AL_3), AL_3 >= AL_1,
                              sl(AL_4), AL_4 >= AL_3,
                              sl(AL_5), AL_5 >= AL_3.

%% If a robot is grasping a graspble object with any of its graspers which is a descendant of some ancestor graspable object then
%% the robot is also grasping the ancestor object with that grasper at the current and higher abstractions
holds(AL_1, grasping(DESC), OBJ_1, t) :- holds(AL_2, grasping(DESC), OBJ_2, t),
                                         fluent(AL_1, inertial, grasping(DESC), OBJ_1),
                                         fluent(AL_2, inertial, grasping(DESC), OBJ_2),
                                         desce_of(AL_2, ROBO, DESC),
                                         desce_of(AL_2, OBJ_1, OBJ_2),
                                         insta_of(AL_1, graspable, OBJ_1),
                                         insta_of(AL_2, graspable, OBJ_2),
                                         insta_of(AL_3, grasper, DESC),
                                         insta_of(AL_4, robot, ROBO),
                                         sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                                         sl(AL_3), AL_3 >= AL_1,
                                         sl(AL_4), AL_4 >= AL_1.

%% If a robot is grasping a graspable object with any of its graspers which is itself a descendant of an ancestor grasper then
%% the robot is also grasping the same object with the ancestor grasper at the current and higher abstractions
holds(AL_1, grasping(DESC_1), OBJ, t) :- holds(AL_2, grasping(DESC_2), OBJ, t),
                                         fluent(AL_1, inertial, grasping(DESC_1), OBJ),
                                         fluent(AL_2, inertial, grasping(DESC_2), OBJ),
                                         desce_of(AL_2, ROBO, DESC_1),
                                         desce_of(AL_2, ROBO, DESC_2),
                                         desce_of(AL_2, DESC_1, DESC_2),
                                         insta_of(AL_1, graspable, OBJ),
                                         insta_of(AL_2, graspable, OBJ),
                                         insta_of(AL_3, grasper, DESC_1),
                                         insta_of(AL_4, grasper, DESC_2),
                                         insta_of(AL_5, robot, ROBO),
                                         sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                                         sl(AL_3), AL_3 >= AL_1,
                                         sl(AL_4), AL_4 >= AL_2,
                                         sl(AL_5), AL_5 >= AL_1.

%% If a robot is grasping a graspable object with any of its graspers then the
%% robot is grasping that object with that grasper at the current and higher abstractions
holds(AL_1, grasping(DESC), OBJ, t) :- holds(AL_2, grasping(DESC), OBJ, t),
                                       fluent(AL_1, inertial, grasping(DESC), OBJ),
                                       fluent(AL_2, inertial, grasping(DESC), OBJ),
                                       desce_of(AL_2, ROBO, DESC),
                                       insta_of(AL_1, graspable, OBJ),
                                       insta_of(AL_2, graspable, OBJ),
                                       insta_of(AL_3, grasper, DESC),
                                       insta_of(AL_4, robot, ROBO),
                                       sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                                       sl(AL_3), AL_3 >= AL_1,
                                       sl(AL_4), AL_4 >= AL_1,
                                       boolean(B).

%% If a configurable entity has configurable descendants which share the same configuration states then the
%% ancestors have the same configuration state as their descendants at the current and higher abstractions
holds(AL_1, configuration(DESC_1), STATE, t) :- holds(AL_2, configuration(DESC_2), STATE, t),
                                                      fluent(AL_1, inertial, configuration(DESC_1), STATE),
                                                      fluent(AL_2, inertial, configuration(DESC_2), STATE),
                                                      desce_of(AL_2, DESC_1, DESC_2),
                                                      state_of(AL_1, DESC_1, STATE),
                                                      state_of(AL_2, DESC_2, STATE),
                                                      insta_of(AL_1, configurable, DESC_1),
                                                      insta_of(AL_2, configurable, DESC_2),
                                                      sl(AL_1), sl(AL_2), AL_1 >= AL_2.

%% If a configurable entity has the same configuration state at the current and higher abstractions
holds(AL_1, configuration(DESC), STATE, t) :- holds(AL_2, configuration(DESC), STATE, t),
                                              fluent(AL_1, inertial, configuration(DESC), STATE),
                                              fluent(AL_2, inertial, configuration(DESC), STATE),
                                              state_of(AL_1, DESC, STATE),
                                              state_of(AL_2, DESC, STATE),
                                              insta_of(AL_1, configurable, DESC),
                                              insta_of(AL_2, configurable, DESC),
                                              sl(AL_1), sl(AL_2), AL_1 >= AL_2.

%% If an object is on a surface that is a descendant of some ancester surface then that
%% object is also on the ancestor surface at the current and higher abstractions
holds(AL_1, on(OBJ), SURFACE_1, t) :- holds(AL_2, on(OBJ), SURFACE_2, t),
                                      fluent(AL_1, inertial, on(OBJ), SURFACE_1),
                                      fluent(AL_2, inertial, on(OBJ), SURFACE_2),
                                      desce_of(AL_2, SURFACE_1, SURFACE_2),
                                      insta_of(AL_1, surface, SURFACE_1),
                                      insta_of(AL_2, surface, SURFACE_2),
                                      insta_of(AL_3, placeable, OBJ),
                                      sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                                      sl(AL_3), AL_3 >= AL_1.

%% If an object is on a surface then it is on that surface at the current and higher abstractions
holds(AL_1, on(OBJ), SURFACE, t) :- holds(AL_2, on(OBJ), SURFACE, t),
                                    fluent(AL_1, inertial, on(OBJ), SURFACE),
                                    fluent(AL_2, inertial, on(OBJ), SURFACE),
                                    insta_of(AL_1, surface, SURFACE),
                                    insta_of(AL_2, surface, SURFACE),
                                    insta_of(AL_3, placeable, OBJ),
                                    sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                                    sl(AL_3), AL_3 >= AL_1.

%% If a robot is transporting an object then that object shares the location of the robot at the current and higher abstractions
holds(AL_1, in(OBJ), L, t) :- holds(AL_2, in(ROBO), L, t),
                              holds(AL_3, transporting(ROBO, OBJ), true, t),
                              fluent(AL_1, inertial, in(OBJ), L),
                              fluent(AL_2, inertial, in(ROBO), L),
                              fluent(AL_3, inertial, transporting(ROBO, OBJ), true),
                              insta_of(AL_1, location, L),
                              insta_of(AL_2, location, L),
                              insta_of(AL_3, graspable, OBJ),
                              insta_of(AL_4, robot, ROBO),
                              sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                              sl(AL_3), AL_3 >= AL_1,
                              sl(AL_4), AL_4 >= AL_3.

%% If a robot is grasping an object with any of its graspers than it is transporting that object at the current and higher abstractions
holds(AL_1, transporting(ROBO, OBJ), true, t) :- holds(AL_2, grasping(COMP), OBJ, t),
                                                 fluent(AL_1, inertial, transporting(ROBO, OBJ), true),
                                                 fluent(AL_2, inertial, grasping(COMP), OBJ),
                                                 desce_of(AL_2, ROBO, COMP),
                                                 insta_of(AL_1, graspable, OBJ),
                                                 insta_of(AL_2, graspable, OBJ), OBJ != nothing,
                                                 insta_of(AL_3, grasper, COMP),
                                                 insta_of(AL_4, robot, ROBO),
                                                 AL_1 >= TL, model_type(TL, tasking),
                                                 sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                                                 sl(AL_3), AL_3 >= AL_2,
                                                 sl(AL_4), AL_4 >= AL_1.

%% If a robot is not grasping an object with any of its manipulator arms then it is not transporting that object at the current and higher abstractions
holds(AL_1, transporting(ROBO, OBJ), false, t) :- 0 = #count { COMP : holds(AL_2, grasping(COMP), OBJ, t),
                                                                      fluent(AL_2, inertial, grasping(COMP), OBJ),
                                                                      desce_of(AL_2, ROBO, COMP),
                                                                      insta_of(AL_3, grasper, COMP),
                                                                      sl(AL_3), AL_3 >= AL_2 },
                                                  fluent(AL_1, inertial, transporting(ROBO, OBJ), false),
                                                  has_desce(AL_2, ROBO, grasper),
                                                  insta_of(AL_1, graspable, OBJ),
                                                  insta_of(AL_2, graspable, OBJ), OBJ != nothing,
                                                  insta_of(AL_4, robot, ROBO),
                                                  AL_1 >= TL, model_type(TL, tasking),
                                                  sl(AL_1), sl(AL_2), AL_1 >= AL_2,
                                                  sl(AL_4), AL_4 >= AL_1.