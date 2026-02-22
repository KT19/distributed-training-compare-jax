import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    path_root = "outputs/"
    dp = path_root + "dp/log.csv"
    tp = path_root + "tp/log.csv"
    pp = path_root + "pp/log.csv"

    dp_df = pd.read_csv(dp)
    tp_df = pd.read_csv(tp)
    pp_df = pd.read_csv(pp)

    colors = ["red", "green", "blue"]

    # plot (loss)
    plt.plot(dp_df["step"], dp_df["loss"], label="dp", alpha=0.5, color=colors[0])
    plt.plot(tp_df["step"], tp_df["loss"], label="tp", alpha=0.5, color=colors[1])
    plt.plot(pp_df["step"], pp_df["loss"], label="pp", alpha=0.5, color=colors[2])
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("outputs/loss.png")
    plt.cla()

    # bar
    labels = ["dp", "tp", "pp"]
    values = [
        np.sum(dp_df["elapsed_time"]),
        np.sum(tp_df["elapsed_time"]),
        np.sum(pp_df["elapsed_time"]),
    ]
    plt.bar(labels, values, color=colors)
    plt.xlabel("method")
    plt.ylabel("time [sec]")
    plt.savefig("outputs/average_elapsed_time.png")


if __name__ == "__main__":
    main()
