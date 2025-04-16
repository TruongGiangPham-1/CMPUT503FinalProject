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

## Generating plots
- Given a directory containing `.csv` files we plot using `python3 plot.py --folder <path-to-folder>`
