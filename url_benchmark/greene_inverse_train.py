

for offset in 10 20 30 50 100
do
    for offline_model_step in 200000 400000 600000 800000 1000000
    do
	    ./greene_inverse_wrap.sh $offset $offline_model_step
    done
done
#for task_no_goal in point_mass_maze_reach_hard_no_goal
#do
#    for task in point_mass_maze_reach_custom_hard_room
#    do
#	for seed in 1 2 3 4 
#        do
#            ./pph_general_wrap.sh $task_no_goal $task $seed
#        done
#    done
#done
#for task_no_goal in point_mass_maze_reach_room_no_goal
#do
#    for task in point_mass_maze_reach_custom_goal_room
#    do
#	for seed in 1 2 3 4 
#        do
#            ./pph_general_wrap.sh $task_no_goal $task $seed
#        done
#    done
#done