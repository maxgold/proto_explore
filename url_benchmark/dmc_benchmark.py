DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]
POINT_MASS_MAZE_TASKS = [
        'point_mass_maze_reach_custom_goal',
        'point_mass_maze_reach_custom_goal_room',
        'point_mass_maze_reach_top_left',
        'point_mass_maze_reach_top_right',
        'point_mass_maze_reach_bottom_left',
        'point_mass_maze_reach_bottom_right',
        'point_mass_maze_reach_no_goal',
        'point_mass_maze_reach_vertical',
        'point_mass_maze_reach_horizontal',
        'point_mass_maze_reach_vertical_no_goal',
        'point_mass_maze_reach_horizontal_no_goal',
        'point_mass_maze_reach_hard_no_goal',
        'point_mass_maze_reach_room_no_goal',
        'point_mass_maze_reach_hard2_no_goal']

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + POINT_MASS_MAZE_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'point_mass_maze': 'point_mass_maze_reach_top_left',
    }
