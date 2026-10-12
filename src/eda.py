"""Generate EDA figures into reports/figures/.

Usage: python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data import ROOT, TARGET, load_clean  # noqa: E402
from src.features import CATEGORICAL, NUMERIC  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    sns.set_theme(style="whitegrid")

    # 1. Class balance
    ax = df[TARGET].map({0: "No disease", 1: "Disease"}).value_counts().plot.bar(
        color=["#4C9F70", "#D1495B"], rot=0)
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance.png", dpi=120)
    plt.close()

    # 2. Histograms of numeric features split by target
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, NUMERIC):
        sns.histplot(df, x=col, hue=TARGET, kde=True, ax=ax, palette=["#4C9F70", "#D1495B"])
        ax.set_title(col)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "numeric_histograms.png", dpi=120)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(11, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature correlation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()

    # 4. Disease rate per category level
    fig, axes = plt.subplots(1, len(CATEGORICAL), figsize=(18, 4))
    for ax, col in zip(axes, CATEGORICAL):
        df.groupby(col)[TARGET].mean().plot.bar(ax=ax, color="#30638E", rot=0)
        ax.set_title(f"Disease rate by {col}")
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "categorical_disease_rate.png", dpi=120)
    plt.close()

    print(f"Saved figures to {FIG_DIR}")
    print(df.describe().T)


if __name__ == "__main__":
    main()
