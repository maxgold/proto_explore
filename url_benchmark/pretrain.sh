#!/bin/bash


env=point_mass_maze_reach_top_right
agent=$1
num_protos=$2
pred_dim=$3
proj_dim=$4
hidden_dim=$5
seed=$6
for agent in 'proto_encoder1' 'proto_encoder2' 'proto_encoder3'
    do
    for num_protos in 16 32 64
	do
        for pred_dim in 16 32 64
	    do
            for proj_dim in 256 512
		do
                for hidden_dim in 64 128
		    do
                    for seed in 0 1
                        do
                            ./pretrain_wrap.sh $agent $num_protos $pred_dim $proj_dim $hidden_dim $seed
                        done		    
                done
            done
        done
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

#env=point_mass_maze_reach_hard2_no_goal_v1
#./finetune_wrap.sh $env $REPLAY_BUFFER_SIZE $NUM_TRAIN_FRAMES $seed $load_seed
#env=point_mass_maze_reach_hard2_no_goal_v2
#./finetune_wrap.sh $env $REPLAY_BUFFER_SIZE $NUM_TRAIN_FRAMES $seed $load_seed
#env=point_mass_maze_reach_room_no_goal_v1
#./finetune_wrap.sh $env $REPLAY_BUFFER_SIZE $NUM_TRAIN_FRAMES $seed $load_seed
#env=point_mass_maze_reach_room_no_goal_v2
#./finetune_wrap.sh $env $REPLAY_BUFFER_SIZE $NUM_TRAIN_FRAMES $seed $load_seed
