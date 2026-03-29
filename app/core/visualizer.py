"""
Visualizer – creates Seaborn charts from pandas DataFrames.

Returns matplotlib Figure objects so the caller can embed them in Streamlit
with st.pyplot(fig) or save them as PNG bytes.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe in Streamlit threads
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

SUPPORTED_CHARTS = ("bar", "line", "scatter", "hist", "box", "heatmap")


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if df[c].dtype == object and df[c].nunique() <= 30
    ]


def auto_detect_params(
    df: pd.DataFrame,
    hint: str = "none",
    x_hint: str = "",
    y_hint: str = "",
) -> tuple[str, str, str]:
    """
    Return (chart_type, x_col, y_col) by inspecting df + LLM hints.
    Falls back gracefully if hints don't match actual columns.
    """
    num = _numeric_cols(df)
    cat = _categorical_cols(df)
    cols = df.columns.tolist()

    chart_type = hint if hint in SUPPORTED_CHARTS else "bar"

    # Resolve x_col
    if x_hint and x_hint in cols:
        x_col = x_hint
    elif cat:
        x_col = cat[0]
    elif cols:
        x_col = cols[0]
    else:
        x_col = ""

    # Resolve y_col
    if y_hint and y_hint in cols:
        y_col = y_hint
    elif num:
        # pick a numeric column that isn't x_col
        candidates = [c for c in num if c != x_col]
        y_col = candidates[0] if candidates else (num[0] if num else "")
    elif len(cols) > 1:
        y_col = cols[1] if cols[1] != x_col else (cols[0] if cols[0] != x_col else "")
    else:
        y_col = ""

    # Auto-upgrade chart type based on data shape
    if chart_type == "bar" and not cat and num:
        chart_type = "hist"
    if chart_type == "heatmap" and len(num) < 2:
        chart_type = "bar"

    return chart_type, x_col, y_col


# ---------------------------------------------------------------------------
# Chart renderer
# ---------------------------------------------------------------------------

def render_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str = "",
    title: str = "",
    hue_col: Optional[str] = None,
    figsize: tuple[int, int] = (9, 5),
) -> plt.Figure:
    """
    Create and return a matplotlib Figure with a Seaborn chart.
    Never raises – falls back to a simple table-info plot on any error.
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)

        # Trim very long string labels for readability
        plot_df = df.copy()
        if x_col in plot_df.columns and plot_df[x_col].dtype == object:
            plot_df[x_col] = plot_df[x_col].astype(str).str[:30]

        chart_type = chart_type.lower()

        if chart_type == "bar":
            if y_col and y_col in plot_df.columns:
                order = (
                    plot_df.groupby(x_col)[y_col].sum()
                    .sort_values(ascending=False).index.tolist()
                    if x_col in plot_df.columns else None
                )
                sns.barplot(
                    data=plot_df, x=x_col, y=y_col,
                    hue=hue_col, order=order[:20] if order else None, ax=ax,
                    errorbar=None,
                )
            else:
                counts = plot_df[x_col].value_counts().head(20)
                sns.barplot(x=counts.index, y=counts.values, ax=ax, errorbar=None)
                ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=35)

        elif chart_type == "line":
            sns.lineplot(
                data=plot_df, x=x_col, y=y_col or (plot_df.columns[1] if len(plot_df.columns) > 1 else x_col),
                hue=hue_col, ax=ax, markers=True,
            )
            ax.tick_params(axis="x", rotation=35)

        elif chart_type == "scatter":
            sns.scatterplot(
                data=plot_df, x=x_col, y=y_col,
                hue=hue_col, ax=ax, alpha=0.7,
            )

        elif chart_type == "hist":
            col = y_col if y_col and y_col in plot_df.columns else x_col
            sns.histplot(data=plot_df, x=col, hue=hue_col, ax=ax, bins=20, kde=True)

        elif chart_type == "box":
            if y_col and y_col in plot_df.columns:
                sns.boxplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            else:
                sns.boxplot(data=plot_df[[x_col]], ax=ax)
            ax.tick_params(axis="x", rotation=35)

        elif chart_type == "heatmap":
            num_df = plot_df.select_dtypes(include="number")
            if num_df.shape[1] >= 2:
                sns.heatmap(num_df.corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
            else:
                ax.text(0.5, 0.5, "Not enough numeric columns for heatmap",
                        ha="center", va="center", transform=ax.transAxes)

        else:
            ax.text(0.5, 0.5, f"Unknown chart type: {chart_type}",
                    ha="center", va="center", transform=ax.transAxes)

        if title:
            ax.set_title(title, fontsize=13, pad=12)
        fig.tight_layout()
        return fig

    except Exception as exc:
        # Fallback: empty figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Chart error: {exc}", ha="center", va="center",
                transform=ax.transAxes, wrap=True)
        ax.axis("off")
        return fig
