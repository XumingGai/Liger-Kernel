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
        print(data_path)
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

def plot_data(df: pd.DataFrame, config: VisualizationsConfig):
    """Plot benchmark results and save both figure and summary CSV."""

    # Drop invalid entries
    df = df.dropna(subset=["y_value_50"])

    # Enrich label for hue
    df["provider_platform"] = df["kernel_provider"] + " (" + df["platform"] + ")"

    # Extract axis labels
    xlabel = df["x_label"].iloc[0]
    ylabel = f"{config.metric_name} ({df['metric_unit'].iloc[0]})"

    # Plot setup
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    palette_dict = {
        provider: "#FFA500" if "a100" in provider.lower() else "#1F77B4"
        for provider in df["provider_platform"].unique()
    } 
    ax = sns.barplot(
        data=df,
        x="x_value",
        y="y_value_50",
        hue="provider_platform",
        palette=palette_dict
    )
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height) or height < 1e-4:
            continue  # Skip NaN or near-zero bars
        if height >= 1000:
            text = f"{height:.2e}"  # scientific notation
        elif height < 0.1:
            text = f"{height:.3f}"
        else:
            text = f"{height:.2f}"
        ax.text(
            x=p.get_x() + p.get_width() / 2,
            y=height,
            s=text,
            ha="center",
            va="bottom",
            fontsize=9
        )
 
    # Finalize plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{config.kernel_name} - {config.metric_name} - {config.kernel_operation_mode}")
    plt.legend(title="Provider (Platform)")
    plt.tight_layout()

    # Save plot
    filename = f"{config.kernel_name}_{config.metric_name}_{config.kernel_operation_mode}_{'_'.join(config.platforms)}.png"
    out_path = os.path.join(VISUALIZATIONS_PATH, filename)

    if config.display:
        plt.show()
    if config.overwrite or not os.path.exists(out_path):
        os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
        plt.savefig(out_path)
    plt.close()

    # Pivot for CSV summary
    pivot_df = df.pivot_table(
        index="x_value",
        columns="platform",
        values="y_value_50"
    )

    # Sort platform columns
    pivot_df = pivot_df[sorted(pivot_df.columns)]

    # Add platform ratio if at least two platforms
    platforms = sorted(pivot_df.columns)
    if len(platforms) >= 2:
        base, compare = platforms[0], platforms[1]
        ratio_name = f"{compare}/{base}"
        pivot_df[ratio_name] = pivot_df[compare] / pivot_df[base]

    # Save CSV
    csv_out_path = os.path.join(
        VISUALIZATIONS_PATH,
        f"{config.kernel_name}_{config.metric_name}_{config.kernel_operation_mode}_{'_'.join(config.platforms)}.csv"
    )
    pivot_df.to_csv(csv_out_path)


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
    pivot_df = pivot_df[sorted(pivot_df.columns)]

    csv_out_path = os.path.join(
        VISUALIZATIONS_PATH,
        f"{config.kernel_name}_{config.metric_name}_{'_'.join(config.platforms)}.csv"
    )

    pivot_df.to_csv(csv_out_path)
    plt.close()
'''


def main():
    config = parse_args()
    df = load_data(config)
    plot_data(df, config)


if __name__ == "__main__":
    main()
