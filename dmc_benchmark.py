DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
    "cheetah",
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

CHEETAH_TASKS = [
    "cheetah_run",
    "cheetah_run_backwards",
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

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + CHEETAH_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    "cheetah": "cheetah_run",
}
