all:
	python3 ppo.py --total_timesteps 1000000 --num_steps 2048  --run_label 1 --debug --video --car_velocity 0.7
