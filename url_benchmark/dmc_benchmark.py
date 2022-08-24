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
        'point_mass_maze_reach_top_left',
        'point_mass_maze_reach_top_right',
        'point_mass_maze_reach_bottom_left',
        'point_mass_maze_reach_bottom_right']

POINT_MASS_TASKS = [
        'point_mass_reach_hs',
        'point_mass_reach_ud_hs']
TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + POINT_MASS_MAZE_TASKS + POINT_MASS_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'point_mass': 'point_mass_reach_ud_hs',
    'point_mass_maze': 'point_mass_maze_reach_custom_goal'}
