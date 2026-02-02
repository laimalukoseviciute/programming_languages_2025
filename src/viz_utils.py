import os
import itertools

import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


def custom_palette(n_colors):
    """
    Returns n_colors sampled from the custom colormap.


    :param n_colors: Number of discrete colors needed.
    :type n_colors: int
    :return: List of RGBA tuples.
    :rtype: list
    """
    oxylabs_cmap = LinearSegmentedColormap.from_list(
        "oxylabs_cmap", ["#130f35", "#52A8F8", "#23E6A8"]
    )
    return [oxylabs_cmap(x) for x in np.linspace(0, 1, n_colors)]


def grouped_barplot_matrix(df, col_names, palette=custom_palette):
    for target_col in col_names:
        group_cols = [col for col in col_names if col != target_col]
        if not group_cols:
            continue

        group_combinations = list(itertools.combinations(group_cols, 1))
        n_plots = len(group_combinations)

        n_cols = min(2, n_plots)
        n_rows = math.ceil(n_plots / n_cols)
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6.5 * n_cols, 4.6 * n_rows),
            constrained_layout=True,
        )

        if n_plots == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        for ax, group_by_col in zip(axs, group_combinations):
            hue_levels = df[group_by_col[0]].nunique()
            sns.countplot(
                data=df,
                x=target_col,
                hue=group_by_col[0],
                palette=palette(hue_levels),
                ax=ax,
            )
            ax.set_title(
                f"{target_col.title().replace("_", " ")} grouped by {group_by_col[0].title().replace("_", " ")}",
                fontsize=14,
            )
            ax.set_xlabel(target_col.title().replace("_", " "), fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.tick_params(axis="x", rotation=90, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.legend(
                title=group_by_col[0].title().replace("_", " "),
                title_fontsize=10,
                fontsize=8,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
            )

        for empty_ax in axs[n_plots:]:
            empty_ax.axis("off")

        plt.suptitle(
            f"Grouped Bar Plots for {target_col.title().replace("_", " ")}",
            fontsize=14,
            y=1.02,
        )
        plt.show()


def plot_countplots(df, cols, plots_per_row=2):
    """
    Plots countplots for multiple categorical columns with percentages above bars.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        cols (list): List of column names to plot.
        plots_per_row (int, optional): Number of plots per row. Defaults to 2.

    Returns:
        None
    """

    n_plots = len(cols)
    n_rows = math.ceil(n_plots / plots_per_row)

    plt.figure(figsize=(15, 5 * n_rows))

    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, plots_per_row, i)

        if col == "year":
            order = sorted(df["year"].dropna().unique())
        else:
            order = df[col].value_counts().index

        ax = sns.countplot(data=df, x=col, order=order, color="#23E6A8")

        plt.title(f"Countplot of {col.title().replace("_", " ")}")
        plt.xlabel(col.title().replace("_", " "), fontsize=10)
        plt.xticks(fontsize=8.5)
        plt.ylabel("Count")

        total = len(df[col].dropna())
        for p in ax.patches:
            height = p.get_height()
            percentage = 100 * height / total if total > 0 else 0
            ax.annotate(
                f"{percentage:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.show()


def plot_language_distribution_by_role(df, roles=None, languages=None):
    """
    Analyzes and plots the distribution of programming languages across job categories
    using binary role columns.


    :param df: The DataFrame containing binary flags for roles and languages.
    :type df: pandas.DataFrame
    :param output_path: The file path where the resulting visualization will be saved.
    :type output_path: str
    :param roles: List of column names representing job roles (binary 0/1).
    :type roles: list, optional
    :param languages: List of column names representing programming languages (binary 0/1).
    :type languages: list, optional
    :return: None
    """

    distribution_data = {}

    for role in roles:
        role_df = df[df[role] == 1]
        if not role_df.empty:
            distribution_data[role.replace("_", " ").capitalize()] = role_df[
                languages
            ].sum()
        else:
            distribution_data[role.replace("_", " ").capitalize()] = pd.Series(
                0, index=languages
            )

    summary_df = pd.DataFrame(distribution_data)
    summary_df.index = [lang.capitalize() for lang in languages]

    plt.figure(figsize=(14, 10))
    summary_norm = summary_df.div(summary_df.sum(axis=0), axis=1) * 100
    oxylabs_cmap = LinearSegmentedColormap.from_list(
        "oxylabs_cmap", ["#130f35", "#52A8F8", "#23E6A8"]
    )

    ax = sns.heatmap(
        summary_norm,
        annot=True,
        fmt=".1f",
        cmap=oxylabs_cmap,
        cbar_kws={"label": "Percentage of Role Mentions (%)"},
    )

    # Aesthetics and Labels
    plt.title(
        "Programming Language Distribution by Professional Role (2025)",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Job Category", fontsize=12)
    plt.ylabel("Programming Language", fontsize=12)

    plt.show()


def plot_language_distribution_by_role(df, roles=None, languages=None):
    """
    Analyzes and plots the distribution of programming languages across job categories
    using binary role columns.


    :param df: The DataFrame containing binary flags for roles and languages.
    :type df: pandas.DataFrame
    :param output_path: The file path where the resulting visualization will be saved.
    :type output_path: str
    :param roles: List of column names representing job roles (binary 0/1).
    :type roles: list, optional
    :param languages: List of column names representing programming languages (binary 0/1).
    :type languages: list, optional
    :return: None
    """

    distribution_data = {}

    for role in roles:
        role_df = df[df[role] == 1]

        if not role_df.empty:
            distribution_data[role.replace("_", " ").capitalize()] = role_df[
                languages
            ].sum()
        else:
            distribution_data[role.replace("_", " ").capitalize()] = pd.Series(
                0, index=languages
            )

    summary_df = pd.DataFrame(distribution_data)
    summary_df.index = [lang.capitalize() for lang in languages]
    plt.figure(figsize=(14, 10))
    summary_norm = summary_df.div(summary_df.sum(axis=0), axis=1) * 100

    oxylabs_cmap = LinearSegmentedColormap.from_list(
        "oxylabs_cmap", ["#130f35", "#52A8F8", "#23E6A8"]
    )

    ax = sns.heatmap(
        summary_norm,
        annot=True,
        fmt=".0f",
        cmap=oxylabs_cmap,
        cbar_kws={"label": "Percentage of Role Mentions (%)"},
    )

    plt.title(
        "Programming Language Distribution by Professional Role (2025)",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Job Category", fontsize=12)
    plt.ylabel("Programming Language", fontsize=12)

    plt.show()


def plot_language_distribution_by_language(df, languages=None, threshold=1.0):
    """
    Analyzes and plots the co-occurrence of programming languages.
    Filters out relationships below the specified threshold.
    """
    lang_subset = df[languages]
    co_occurrence = lang_subset.T.dot(lang_subset)

    total_postings = len(df)
    co_occurrence_pct = (co_occurrence / total_postings) * 100
    co_occurrence_pct = co_occurrence_pct.where(co_occurrence_pct >= threshold)

    mask = np.triu(np.ones_like(co_occurrence_pct, dtype=bool))

    oxylabs_cmap = LinearSegmentedColormap.from_list(
        "oxylabs_cmap", ["#130f35", "#52A8F8", "#23E6A8"]
    )

    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(
        co_occurrence_pct,
        mask=mask,
        annot=True,
        fmt=".0f",
        cmap=oxylabs_cmap,
        square=True,
        cbar_kws={"label": "Co-occurrence Frequency (%)", "shrink": 0.8},
        linecolor="#f0f0f0",
    )

    plt.title("Programming Language Co-occurrence in 2025 Job Postings", pad=20)
    plt.yticks(rotation=0)
    ax.collections[0].colorbar.outline.set_visible(False)
    
    plt.show()

def plot_language_market_share(
    df,
    prog_languages,
    precision_map,
    top_n=10,
    title="In-Demand Programming Languages: 2025 Market Share",
):
    """
    Calculates market share for languages, applies precision adjustments, and plots results.


    :param df: The raw DataFrame containing binary columns for each language.
    :type df: pandas.DataFrame
    :param prog_languages: List of column names representing programming languages.
    :type prog_languages: list
    :param precision_map: Dictionary mapping language names to their precision multipliers.
    :type precision_map: dict
    :return: None
    """

    lang_counts = df[prog_languages].sum().astype(float)

    for lang, multiplier in precision_map.items():
        if lang in lang_counts.index:
            lang_counts[lang] = lang_counts[lang] * multiplier

    total_postings = len(df)
    lang_percentages = (lang_counts / total_postings) * 100

    lang_percentages = lang_percentages.sort_values(ascending=False).head(top_n)
    lang_percentages.index = [
        str(col) if len(str(col)) > 1 else str(col) for col in lang_percentages.index
    ]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="none")
    ax.set_facecolor("none")
    ax = lang_percentages.plot(kind="bar", color="#23E6A8")

    for container in ax.containers:
        ax.bar_label(container, fmt="$%.0f\\%%$", padding=3)

    ax.set_title(title, pad=15)
    ax.set_ylabel("Percentage of Total Job Postings (%)")
    ax.set_xlabel("Programming Language")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=0)

    ax.set_ylim(0, lang_percentages.max() * 1.15)

    plt.tight_layout()
    return fig, ax


def plot_us_hiring_heatmap(
    df_map_data, total_denominator, output_filename="us_hiring_map"
):
    """
    Creates an interactive US choropleth map with percentages aligned to the global dataset.

    :param df_map_data: DataFrame with "state_code" and raw "counts"
    :param total_denominator: The total number of records (including NaNs) from the main DF
    :param output_filename: The name of the HTML file to save.
    """
    output_dir = "../outputs/figures"
    file_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    df_map_data["percentage"] = (df_map_data["counts"] / total_denominator) * 100

    custom_scale = ["#130f35", "#52A8F8", "#23E6A8"]

    fig = px.choropleth(
        df_map_data,
        locations="state_code",
        locationmode="USA-states",
        color="percentage",
        scope="usa",
        color_continuous_scale=custom_scale,
        labels={"percentage": "Market Share (%)", "state_code": "State"},
        title="<b>2025 US Job Market: Regional Demand Distribution</b>",
        hover_data={"state_code": True, "percentage": ":.2f"},
    )

    fig.update_layout(
        title_font_size=22,
        title_x=0.5,
        margin={"r": 0, "t": 80, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            bgcolor="rgba(0,0,0,0)", showlakes=False, showland=True, landcolor="#f9f9f9"
        ),
    )

    fig.write_html(file_path + ".html")
    fig.write_image(file_path + ".png", format="png", scale=3, width=1200, height=800)
    fig.show()


def plot_us_language_dominance(df_dominant, output_filename="us_lang_dominance", color_map={"Python": "#130f35", "SQL": "#23E6A8"}):
    """
    Plots a map where each state is colored by its most popular programming language.


    :param df_dominant: DataFrame with "state_code" and "top_language".
    :type df_dominant: pandas.DataFrame
    :param output_filename: Filename for the exports.
    :type output_filename: str
    :return: None
    """
    output_dir = "../outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_filename)
    color_map = color_map

    fig = px.choropleth(
        df_dominant,
        locations="state_code",
        locationmode="USA-states",
        color="top_language",
        scope="usa",
        color_discrete_map=color_map,
        labels={"top_language": "Dominant Language"},
        title="2025 Most Mentioned Language by State",
    )

    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", 
        geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
        legend_title_text="Programming Language",
    )

    fig.write_html(file_path + ".html")
    fig.write_image(file_path + ".png", scale=3)
    fig.show()
