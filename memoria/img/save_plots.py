"""
Generar y guardar los gráficos basado en los resultados del test.
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt



# CONFIG
RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "results"
PLOTS_PATH = Path(__file__).resolve().parent / "plots"


def gen_plots():
    """
    Generate and save the defined plots.
    """
    # METRICS:
    df_metrics = pd.read_csv(RESULTS_PATH / "metrics.csv")

    # 1) auc_pr vs auc_roc vs tpr_at_fpr1 of the best 3 models
    cols = ["auc_pr_test", "auc_roc_test", "tpr_at_fpr1_test"]
    df_top3 = df_metrics.sort_values(cols, ascending=False).iloc[:3, :]
    # Plot: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    fig, ax = plt.subplots()
    res = ax.grouped_bar(df_top3.loc[:, cols], tick_labels=tuple(df_top3["model"]))
    for container in res.bar_containers:
        ax.bar_label(container, padding=3)
    ax.set_xlabel("Models")
    ax.set_title("Comparativa de rendimiento - Top 3\n",
                 fontdict={"weight": "bold", "size": "x-large"})
    ax.legend(bbox_to_anchor=(0.5, 0, 0.5, 0.5))
    plt.savefig(PLOTS_PATH/"top3_models.png", pad_inches=0.5)



def main():
    os.makedirs(PLOTS_PATH, exist_ok=True)

    gen_plots()


if __name__ == "__main__":
    main()
