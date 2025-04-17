import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
CCID = "truonggi"

def main():
    #folder_name = "data/walker-1million"
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data", help="path to the folder containing csv")
    parser.add_argument("--duration", action="store_true", help="if true, plot the duration")
    args = parser.parse_args()
    folder_name = args.folder
    running_avg_range = 100
    duration = args.duration

    produce_plots_for_all_configs(folder_name, duration)
    return


def extract_config(filename_without_ext):
    configs = ['straight_road', 'small_loop']
    for configuration in configs:
        if configuration in filename_without_ext:
            return configuration
    return None

def produce_plots_for_all_configs(folder_name="data", duration=False):
    keyword = "duration" if duration else "returns"
    configs = ['straight_road', 'small_loop']
    data_dict = {}
    for configuration in configs:
        data_dict[configuration] = []
    files = os.listdir(folder_name)
    for file in files:
        full_path = os.path.join(folder_name, file)
        if os.path.isfile(full_path):
            if os.path.splitext(file)[-1] == '.csv' and keyword in file:
                config = extract_config(os.path.splitext(file)[0])
                assert config is not None, f"{file} is not in the required format."
                print(f"Reading {full_path}")
                df = pd.read_csv(full_path)
                data_dict[config].append(np.squeeze(df.values))

    for configuration in configs:
        if data_dict[configuration]:
            plot_alg_results(data_dict[configuration], f"PPO_{configuration}_{keyword}.png", label="Running average", duration=duration, ylabel=keyword)


def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return", eval_interval=20480, duration=False):
    """
    episode_returns_list: list of episode returns. If there is 3 seeds, then the list should have 3 lists.


    1 do 1 million steps. each sample collection is 2048 steps. so ~ 1mil / 2048 = 488 updates.
    I evaluate every 10 update iteration, so in total 48 evaluations

    each evaluation is after 2048*10 = 20480 steps
    so x axis is 0, 20480, 40960, ..., 20480*48 = 983040
    """
    # Compute running average
    print("len of return list ", len(episode_returns_list[0]))

    running_avg = np.mean(np.array(episode_returns_list), axis=0)  # Average over seeds. dim (1, num_episodes)
    new_running_avg = running_avg.copy()
    for i in range(len(running_avg)):
        new_running_avg[i] = np.mean(running_avg[max(0, i-10):min(len(running_avg), i)])  # each point is the average of itself and its neighbors (+/- 10*eval_interval)
    running_avg = new_running_avg

    # x axis goes by 20480
    x_coords = np.arange(0, len(running_avg) * eval_interval, eval_interval)
    assert len(x_coords) == len(running_avg), f"len of x_coords {len(x_coords)} != len of running_avg {len(running_avg)}"
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot individual seeds with light transparency
    for seed_returns in episode_returns_list:
        plt.plot(x_coords, seed_returns,  color='gray', alpha=0.5)
    # Plot the running average
    plt.plot(
        x_coords,
        running_avg,
        color='r',
        label=label
    )
    #plt.plot(x_coords, np.full(len(running_avg), 3500)   , color='b', label='threshold')

    # Adding labels and title
    keyword = "Duration" if duration else "Returns"
    if 'straight_road' in file:
        plt.title(f"({CCID}) PPO straight_road {keyword}")
    else:
        plt.title(f"({CCID}) PPO small_loop {keyword}")
    plt.xlabel("Training Steps")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


if __name__ == "__main__":
    main()