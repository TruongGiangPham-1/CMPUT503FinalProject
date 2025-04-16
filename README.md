# CMPUT503FinalProject


## Requirements
- `python 3.10`
## Install instruction
-   git clone https://github.com/MasWag/gym-duckietown/tree/daffy
-   virtualenv dt-sim && source dt-sim/bin/activate
-   pip3 install -e gym-duckietown/
-   pip3 install torch tyro

## Running instructions
1. Use the make file. `make all` to train. (locally)
2. `sbatch CC_script.sh $run_label` (compute canada)

Training will generate `data/...map_name_runlabel_x_returns.csv` and `data/...map_name_runlabel_x_duration.csv`.
`..returns.csv` is the undiscounted rewards we collect during the phase of training.
`..durations.csv` is the average timestep of the training

## Generating plots
- Given a directory containing `.csv` files we plot using `python3 plot.py --folder <path-to-folder>`
Only requirement is that the `.csv` file must contain the duckietown map name
