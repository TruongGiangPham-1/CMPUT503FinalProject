all:
	python3 ppo.py --total_timesteps 1000000 --num_steps 2048 --num_minibatches 4 --run_label 1
