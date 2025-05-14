import json
import os

from argparse import ArgumentParser
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/all_benchmark_data.csv"))
VISUALIZATIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizations/"))


@dataclass
class VisualizationsConfig:
    """
    Configuration for the visualizations script.

    Args:
        kernel_name (str): Kernel name to benchmark. (Will run `scripts/benchmark_{kernel_name}.py`)
        metric_name (str): Metric name to visualize (speed/memory)
        kernel_operation_mode (str): Kernel operation mode to visualize (forward/backward/full). Defaults to "full"
        display (bool): Display the visualization. Defaults to False
        overwrite (bool): Overwrite existing visualization, if none exist this flag has no effect as ones are always created and saved. Defaults to False

    """

    kernel_name: str
    metric_name: str
    kernel_operation_mode: str = "full"
    display: bool = False
    overwrite: bool = False
    platforms: list[str] = None


def parse_args() -> VisualizationsConfig:
    """Parse command line arguments into a configuration object.

    Returns:
        VisualizationsConfig: Configuration object for the visualizations script.
    """
    parser = ArgumentParser()
    parser.add_argument("--kernel-name", type=str, required=True, help="Kernel name to benchmark")
    parser.add_argument(
        "--metric-name",
        type=str,
        required=True,
        help="Metric name to visualize (speed/memory)",
    )
    parser.add_argument(
        "--kernel-operation-mode",
        type=str,
        required=True,
        help="Kernel operation mode to visualize (forward/backward/full)",
    )
    parser.add_argument("--display", action="store_true", help="Display the visualization")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing visualization, if none exist this flag has no effect as one are always created",
    )
    parser.add_argument(
    "--platforms",
    type=str,
    required=True,
    help="Comma-separated list of hardware platforms (e.g., a100,intel)",
    )  

    args = parser.parse_args()
    platforms = args.platforms.split(",")
    return VisualizationsConfig(platforms=platforms, **{k: v for k, v in vars(args).items() if k != "platforms"})

def load_data(config: VisualizationsConfig) -> pd.DataFrame:
    dfs = []

    for platform in config.platforms:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"data/{platform}/all_benchmark_data.csv"))
        df = pd.read_csv(data_path)
        df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(json.loads)
        df["platform"] = platform  # ← 标记平台
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    filtered_df = combined_df[
        (combined_df["kernel_name"] == config.kernel_name)
        & (combined_df["metric_name"] == config.metric_name)
        & (combined_df["kernel_operation_mode"] == config.kernel_operation_mode)
    ]

    if filtered_df.empty:
        raise ValueError("No data found for the given filters")

    return filtered_df

'''
def plot_data(df: pd.DataFrame, config: VisualizationsConfig):
    """Plots the benchmark data as bar chart, saving the result if needed.

    Args:
        df (pd.DataFrame): Filtered benchmark dataframe.
        config (VisualizationsConfig): Configuration object for the visualizations script.
    """
    xlabel = df["x_label"].iloc[0]
    ylabel = f"{config.metric_name} ({df['metric_unit'].iloc[0]})"

    df["provider_platform"] = df["kernel_provider"] + " (" + df["platform"] + ")"
    df = df.sort_values(by=["x_value", "provider_platform"])

    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid")

    # Bar chart
    ax = sns.barplot(
        data=df,
        x="x_value",
        y="y_value_50",
        hue="provider_platform",
        palette="tab10",
        errorbar=None,  # 我们手动添加误差棒
        dodge=True,
    )

    # 手动添加误差棒
    grouped = df.groupby(["x_value", "provider_platform"])
    positions = {}

    # 获取每个 bar 的横轴位置
    #for bar, (_, row) in zip(ax.patches, grouped):
    for bar, ((x_val, provider_platform), group_data) in zip(ax.patches, grouped):
        x = bar.get_x() + bar.get_width() / 2
        #positions[(row[0], row[1])] = x
        #positions[(row["x_value"], row["provider_platform"])] = x
        positions[(x_val, provider_platform)] = x

    for (x_val, provider_platform), group_data in grouped:
        row = group_data.iloc[0]
        y = row["y_value_50"]
        yerr_lower = y - row["y_value_20"]
        yerr_upper = row["y_value_80"] - y
        x_pos = positions[(x_val, provider_platform)]
        plt.errorbar(
            x=x_pos,
            y=y,
            yerr=[[yerr_lower], [yerr_upper]],
            fmt="none",
            ecolor="black",
            capsize=5,
            linewidth=1,
        )

    plt.legend(title="Provider (Platform)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    filename = f"{config.kernel_name}_{config.metric_name}_{'_'.join(config.platforms)}.png"
    out_path = os.path.join(VISUALIZATIONS_PATH, filename)

    if config.display:
        plt.show()
    if config.overwrite or not os.path.exists(out_path):
        os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
        plt.savefig(out_path)
    plt.close()
'''

def plot_data(df: pd.DataFrame, config: VisualizationsConfig):
    """Plots the benchmark data as a bar chart, saving the result if needed.

    Args:
        df (pd.DataFrame): Filtered benchmark dataframe.
        config (VisualizationsConfig): Configuration object for the visualizations script.
    """
    df["provider_platform"] = df["kernel_provider"] + " (" + df["platform"] + ")"

    xlabel = df["x_label"].iloc[0]
    ylabel = f"{config.metric_name} ({df['metric_unit'].iloc[0]})"

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    ax = sns.barplot(
        data=df,
        x="x_value",
        y="y_value_50",
        hue="provider_platform",
        palette="tab10"
    )
    for p in ax.patches:
        height = p.get_height()
        if pd.notna(height):
            ax.text(
                x=p.get_x() + p.get_width() / 2,
                y=height,
                s=f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    print(df[["x_value", "y_value_50", "provider_platform"]])
    df = df.dropna(subset=["y_value_50"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{config.kernel_name} - {config.metric_name}")
    plt.legend(title="Provider (Platform)")
    plt.tight_layout()

    filename = f"{config.kernel_name}_{config.metric_name}_{'_'.join(config.platforms)}.png"
    out_path = os.path.join(VISUALIZATIONS_PATH, filename)

    if config.display:
        plt.show()
    if config.overwrite or not os.path.exists(out_path):
        os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
        plt.savefig(out_path)
    pivot_df = df.pivot_table(
        index="x_value",
        columns="platform",
        values="y_value_50"
    )
    print(pivot_df)    
    pivot_df = pivot_df[sorted(pivot_df.columns)]

    csv_out_path = os.path.join(
        VISUALIZATIONS_PATH,
        f"{config.kernel_name}_{config.metric_name}_{'_'.join(config.platforms)}.csv"
    )

    pivot_df.to_csv(csv_out_path)
    plt.close()



def main():
    config = parse_args()
    df = load_data(config)
    plot_data(df, config)


if __name__ == "__main__":
    main()
