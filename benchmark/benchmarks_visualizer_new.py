import json
import os

from argparse import ArgumentParser
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/all_benchmark_data.csv"))
VISUALIZATIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizations/"))


@dataclass
class VisualizationsConfig:
    kernel_names: list[str]
    metric_name: str
    kernel_operation_mode: str = "full"
    display: bool = False
    overwrite: bool = False
    platforms: list[str] = None

def parse_args() -> VisualizationsConfig:
    parser = ArgumentParser()
    parser.add_argument("--kernel-name", type=str, required=True, help="Comma-separated kernel names to benchmark")
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
    kernel_names = args.kernel_name.split(",")
    return VisualizationsConfig(platforms=platforms, kernel_names=kernel_names, **{k: v for k, v in vars(args).items() if k not in ["platforms", "kernel_name"]})

def load_data(config: VisualizationsConfig) -> pd.DataFrame:
    dfs = []
    for platform in config.platforms:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"data/{platform}/all_benchmark_data.csv"))
        print(data_path)
        df = pd.read_csv(data_path)
        df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
        df["platform"] = platform
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    filtered_df = combined_df[
        (combined_df["kernel_name"].isin(config.kernel_names))
        & (combined_df["metric_name"] == config.metric_name)
        & (combined_df["kernel_operation_mode"] == config.kernel_operation_mode)
    ]

    return filtered_df

def plot_data(df: pd.DataFrame, config: VisualizationsConfig):
    df = df.dropna(subset=["y_value_50"])
    df["provider_platform"] = df["kernel_provider"] + " (" + df["platform"] + ")"

    for kernel in config.kernel_names:
        df_kernel = df[df["kernel_name"] == kernel]
        if df_kernel.empty:
            print(f"Warning: No data for kernel {kernel} with operation mode '{config.kernel_operation_mode}'")
            continue

        xlabel = df_kernel["x_label"].iloc[0]
        ylabel = f"{config.metric_name} ({df_kernel['metric_unit'].iloc[0]})"

        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid")
        palette_dict = {
            provider: "#FFA500" if "a100" in provider.lower() else "#1F77B4"
            for provider in df_kernel["provider_platform"].unique()
        }
        ax = sns.barplot(
            data=df_kernel,
            x="x_value",
            y="y_value_50",
            hue="provider_platform",
            palette=palette_dict
        )
        for p in ax.patches:
            height = p.get_height()
            if pd.isna(height) or height < 1e-4:
                continue
            text = f"{height:.2f}"
            ax.text(
                x=p.get_x() + p.get_width() / 2,
                y=height,
                s=text,
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{kernel} - {config.metric_name} - {config.kernel_operation_mode}")
        plt.legend(title="Provider (Platform)")
        plt.tight_layout()

        filename = f"{kernel}_{config.metric_name}_{config.kernel_operation_mode}_{'_'.join(config.platforms)}.png"
        out_path = os.path.join(VISUALIZATIONS_PATH, filename)

        if config.display:
            plt.show()
        if config.overwrite or not os.path.exists(out_path):
            os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
            plt.savefig(out_path)
        plt.close()

        pivot_df = df_kernel.pivot_table(
            index="x_value",
            columns="platform",
            values="y_value_50"
        )
        pivot_df = pivot_df[sorted(pivot_df.columns)]

        platforms = sorted(pivot_df.columns)
        if len(platforms) >= 2:
            base, compare = platforms[0], platforms[1]
            ratio_name = f"{compare}/{base}"
            pivot_df[ratio_name] = pivot_df[compare] / pivot_df[base]

        csv_out_path = os.path.join(
            VISUALIZATIONS_PATH,
            f"{kernel}_{config.metric_name}_{config.kernel_operation_mode}_{'_'.join(config.platforms)}.csv"
        )
        pivot_df.to_csv(csv_out_path)

def main():
    config = parse_args()
    df = load_data(config)
    if df.empty:
        print("Warning: No data found for the specified filters. Nothing to visualize.")
        return
    plot_data(df, config)

if __name__ == "__main__":
    main()

