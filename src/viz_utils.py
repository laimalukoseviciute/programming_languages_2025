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


oxy_violet = "#130f35"
oxy_dark_blue = "#3563a0"
oxy_light_blue = "#52A8F8"
oxy_light_teal = "#3ac7cd"
oxy_teal = "#23E6A8"


plt.style.use("fast")
plt.rcParams.update(
    {
        "font.family": "Avenir",
        "text.color": oxy_violet,
        "axes.labelcolor": oxy_violet,
        "xtick.color": oxy_violet,
        "ytick.color": oxy_violet,
        "figure.facecolor": "none",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlepad": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.transparent": True,
        "figure.autolayout": True,
    }
)

def custom_palette(n_colors):
    """
    Returns n_colors sampled from the custom colormap.


    :param n_colors: Number of discrete colors needed.
    :type n_colors: int
    :return: List of RGBA tuples.
    :rtype: list
    """
    oxylabs_cmap = LinearSegmentedColormap.from_list(
        "oxylabs_cmap", [oxy_violet, oxy_light_blue, oxy_teal]
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

        ax = sns.countplot(data=df, x=col, order=order, color=oxy_teal)

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


def plot_tools_distribution_by_tools(df, tools=None, threshold=1.0):
    """
    Analyzes and plots a heatmap showing how often specific tech tools appear together in the same job postings.


    :param df: The raw DataFrame where each column is a tool with binary (1/0) indicators of presence.
    :type df: pandas.DataFrame
    :param tools: A list of column names representing the tech tools to be analyzed.
    :type tools: list, optional
    :param threshold: The minimum percentage of co-occurrence required to display a relationship in the heatmap.
    :type threshold: float
    :return: None
    :rtype: None
    """

    tool_subset = df[tools]
    co_occurrence = tool_subset.T.dot(tool_subset)

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

    plt.title("Tech Tool Co-occurrence in 2025 Job Postings", pad=20)
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
    ax = lang_percentages.plot(kind="bar", color=oxy_teal)

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


def plot_tools_market_share(
    df,
    tech_tools,
    top_n=10,
    title="In-Demand Tech Tools: 2025 Market Share",
):
    """
    Calculates market share for tech tools, and plots results.


    :param df: The raw DataFrame containing binary columns for each language.
    :type df: pandas.DataFrame
    :param tech_tools: List of column names representing tech tools.
    :type tech_tools: list
    :return: None
    """

    tool_counts = df[tech_tools].sum().astype(float)

    total_postings = len(df)
    tool_percentages = (tool_counts / total_postings) * 100

    tool_percentages = tool_percentages.sort_values(ascending=False).head(top_n)
    tool_percentages.index = [
        str(col) if len(str(col)) > 1 else str(col) for col in tool_percentages.index
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = tool_percentages.plot(kind="bar", color=oxy_teal)

    for container in ax.containers:
        ax.bar_label(container, fmt="$%.0f\\%%$", padding=3)

    ax.set_title(title, pad=15)
    ax.set_ylabel("Percentage of Total Job Postings (%)")
    ax.set_xlabel("Tools")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=0)

    ax.set_ylim(0, tool_percentages.max() * 1.15)

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
    output_dir = "../outputs/figures/languages"
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


def plot_us_language_dominance(
    df_dominant,
    output_filename="us_lang_dominance",
    color_map={"Python": oxy_violet, "SQL": oxy_teal},
):
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


def plot_us_tools_dominance(
    df_dominant,
    output_filename="us_tools_dominance",
    color_map={"Excel": oxy_violet, "AWS": oxy_teal},
):
    """
    Plots a map where each state is colored by its most popular tech tool.

    :param df_dominant: DataFrame with "state_code" and "top_tool".
    :type df_dominant: pandas.DataFrame
    :param output_filename: Filename for the exports.
    :type output_filename: str
    :param color_map: Dictionary mapping tool names to colors.
    :type color_map: dict
    :return: None
    """

    output_dir = "../outputs/figures/tools"
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
        labels={"top_language": "Dominant CP"},
        title="2025 Most Mentioned Cloud Platforms by State",
    )

    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)"),
        legend_title_text="Dominant Tool",
    )

    fig.write_html(file_path + ".html")
    fig.write_image(file_path + ".png", scale=3)
    fig.show()


def plot_category_requirements(df_count, categories):
    """
    Aggregates tool counts into categories and plots a 5-bar summary chart.

    :param df_count: Dataframe containing "count" for each tool.
    :type df_count: pandas.DataFrame
    :param categories: Dictionary mapping category names to lists of tools.
    :type categories: dict
    :return: None
    :rtype: None
    """
    category_data = []

    for cat_name, tools in categories.items():
        valid_tools = df_count.index.intersection(tools)
        total_count = df_count.loc[valid_tools, "count"].sum()
        category_data.append({"category": cat_name, "total_count": total_count})

    df_categories = pd.DataFrame(category_data).sort_values(by="total_count", ascending=False)
    
    total_sum = df_categories["total_count"].sum()
    df_categories["percentage"] = (df_categories["total_count"] / total_sum) * 100

    plt.figure(figsize=(12, 7))

    palette = {
        "Data Storage & Infrastructure": oxy_teal,
        "Data Ingestion & Transformation": oxy_light_teal,
        "Business Intelligence (BI) & Analytics": oxy_light_blue,
        "Orchestration & Observability": oxy_dark_blue,
        "DevOps & Developer Experience": oxy_violet
    }

    ax = sns.barplot(
        data=df_categories,
        x="percentage",
        y="category",
        hue="category",
        palette=palette,
        legend=False
    )

    plt.title("Tech Tool Demand by Category (2025)", fontsize=18, pad=20)
    plt.xlabel("Percentage of Total Tool Mentions", fontsize=14)
    plt.ylabel("Category", fontsize=14)

    for i, (index, row) in enumerate(df_categories.iterrows()):
        ax.text(
            row["percentage"] + 0.5, 
            i, 
            f"{row['percentage']:.1f}%", 
            va="center", 
            fontweight="bold",
            fontsize=12
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.xlim(0, df_categories["percentage"].max() * 1.15)

    plt.tight_layout()
    plt.show()


def plot_tool_requirements(df_count):
    """
    Generates a color-coded bar chart of tech tool requirements for 2025.

    :param df_count: Dataframe containing 'count' and 'percentage' for each tool.
    :type df_count: pandas.DataFrame
    :return: None (displays the plot).
    :rtype: None
    """
    categories = {
        "Data Storage & Infrastructure": [
            "Snowflake", "BigQuery", "AWS", "Azure", "GCP", "Databricks", "Spark"
        ],
        "Data Ingestion & Transformation": [
            "Fivetran", "Airbyte", "dbt", "Kafka"
        ],
        "Business Intelligence (BI) & Analytics": [
            "Power_BI", "Tableau", "Looker", "Excel", "Google_Sheets"
        ],
        "Orchestration & Observability": [
            "Airflow", "Prefect", "Monte_Carlo"
        ],
        "DevOps & Developer Experience": [
            "Git", "Docker", "Kubernetes", "Terraform"
        ]
    }

    tool_to_cat = {tool: cat for cat, tools in categories.items() for tool in tools}
    df_count["category"] = df_count.index.map(tool_to_cat)
    df_plot = df_count.reset_index().rename(columns={"index": "tool"})
    df_plot = df_plot.sort_values("percentage", ascending=False)
    plt.figure(figsize=(14, 10))

    palette = {
        "Data Storage & Infrastructure": oxy_teal,
        "Data Ingestion & Transformation": oxy_light_teal,
        "Business Intelligence (BI) & Analytics": oxy_light_blue,
        "Orchestration & Observability": oxy_dark_blue,
        "DevOps & Developer Experience": oxy_violet
    }

    ax = sns.barplot(
        data=df_plot,
        x="percentage",
        y="tool",
        hue="category",
        palette=palette,
        dodge=False
    )

    plt.title("Most Required Tech Tools in IT Job Postings (2025)", fontsize=18, pad=20)
    plt.xlabel("Percentage of Mentions in Job Postings", fontsize=14)
    plt.ylabel("Technology Tool", fontsize=14)
    
    plt.legend(title="Tool Category", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if width > 0:
            ax.text(
                width + 0.5, 
                p.get_y() + p.get_height() / 2, 
                f"{width:.1f}%", 
                va="center",
                fontsize=11
            )

    plt.xlim(0, df_plot["percentage"].max() + 5)

    plt.show()