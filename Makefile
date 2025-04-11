all:
	python3 ppo.py --total_timesteps 10 --num_steps 5 --batch_size 1 --num_minibatches 5
