import re
import numpy as np
import matplotlib.pyplot as plt
import pathlib

def parse_log(path):
    times = []
    finished = []
    gates = []

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Look for Flight time line
        m = re.search(r"Flight time \(s\):\s*([0-9]+\.[0-9]+)", lines[i])
        if m:
            t = float(m.group(1))
            times.append(t)
            # Next lines: Finished and Gates passed
            if i+1 < len(lines):
                m2 = re.search(r"Finished:\s*(True|False)", lines[i+1])
                finished.append(m2.group(1) == "True" if m2 else False)
            else:
                finished.append(False)
            if i+2 < len(lines):
                m3 = re.search(r"Gates passed:\s*([0-9]+)", lines[i+2])
                gates.append(int(m3.group(1)) if m3 else 0)
            else:
                gates.append(0)
            i += 3
        else:
            i += 1

    return np.array(times), np.array(finished), np.array(gates)

def main():
    log_path = pathlib.Path(__file__).parent / "sim.log"
    times, finished, gates = parse_log(log_path)

    total_runs = len(times)
    success_runs = finished.sum()
    fail_runs = total_runs - success_runs
    avg_time = times[finished].mean() if success_runs > 0 else float('nan')

    print(f"Total runs        : {total_runs}")
    print(f"Successful runs   : {success_runs}")
    print(f"Failed runs       : {fail_runs}")
    print(f"Average lap time  : {avg_time:.2f} s")

    # Print all finished lap times with two-decimal precision
    finished_times = times[finished]
    print("Finished lap times:", [f"{t:.2f}" for t in finished_times])

    # Plot 1: Lap times per run (only finished)
    plt.figure(figsize=(8,4))
    idx = np.arange(total_runs)
    plt.plot(idx[finished], times[finished], 'o-', label="Finished runs")
    plt.plot(idx[~finished], [np.nan]*fail_runs, 'x', label="DNF", color="gray")
    plt.xlabel("Run index")
    plt.ylabel("Lap time (s)")
    plt.title("Lap times for finished runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lap_times_per_run.png", dpi=150)

    # Plot 2: Histogram of gates passed
    unique, counts = np.unique(gates, return_counts=True)
    plt.figure(figsize=(6,4))
    plt.bar(unique, counts, width=0.6, color='skyblue', edgecolor='k')
    plt.xlabel("Gates passed")
    plt.ylabel("Number of runs")
    plt.title("Distribution of Gates Passed")
    plt.xticks(unique)
    plt.tight_layout()
    plt.savefig("gates_passed_histogram.png", dpi=150)

    # Plot 3: Histogram of finished lap times
    if success_runs > 0:
        plt.figure(figsize=(6,4))
        data = times[finished]
        plt.hist(data, bins=20, color='teal', edgecolor='black')
        plt.xlabel("Lap time (s)")
        plt.ylabel("Count")
        plt.title("Distribution of Finished Lap Times")
        plt.tight_layout()
        plt.savefig("lap_times_histogram.png", dpi=150)

    print("Plots saved to lap_times_per_run.png, gates_passed_histogram.png, and lap_times_histogram.png")

if __name__ == "__main__":
    main()