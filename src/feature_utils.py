import re
import os

import gc
import asyncio
import numpy as np
from ollama import AsyncClient
import pandas as pd
import polars as pl
import nest_asyncio
from tqdm.asyncio import tqdm as atqdm

nest_asyncio.apply()


async def verify_single_row(client, snippet, language):
    """
    Sends a single request to Ollama asynchronously.

    :param client: The Ollama client.
    :type client: ollama.AsyncClient
    :param snippet: The job description snippet.
    :type snippet: str
    :param language: The language to verify.
    :type language: str
    :return: The verification result.
    :rtype: int
    """

    prompt = f"""
    [INST]
    You are a technical recruiter. Analyze the job description snippet.
    Determine if {language} is a required programming language in a given 
    description or just a common word.

    Rules:
    - Ignore common phrases with that word or typos of the word {language}.
    - Ignore URLs containing "/{language}/".
    - Focus on technical stacks.
    - Return '1' if it is the programming language.
    - Return '0' if it is a common word or part of a URL.

    Snippet: "{snippet}"
    [/INST]
    Answer (1 or 0):"""

    response = await client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_ctx": 1024},
    )
    content = response["message"]["content"].strip()
    return 1 if "1" in content else 0


async def verify_single_row_tools(client, snippet, tool):
    """
    Verifies a single row of a tool using Ollama asynchronously.

    :param client: The Ollama client.
    :type client: ollama.AsyncClient
    :param snippet: The job description snippet.
    :type snippet: str
    :param tool: The tool to verify.
    :type tool: str
    :return: The verification result.
    :rtype: int
    """
    system_content = (
        "You are a professional data labeler. Your only job is to detect if a "
        "word is used as a technical software tool or a general English word."
    )

    user_content = (
        f"Analyze the word '{tool}' in the snippet below.\n\n"
        "Criteria for 1 (YES):\n"
        "- Mentioned as a skill, software, or part of a tech stack.\n"
        "Criteria for 0 (NO):\n"
        "- Used as a verb (to excel), adjective (excellent), or part of a URL.\n"
        "- Used in a non-technical context (building airflow).\n\n"
        "Example 1: 'Advanced Excel skills required' -> 1\n"
        "Example 2: 'Must excel at communication' -> 0\n\n"
        f'Snippet: "{snippet}"\n\n'
        "Final Answer (Respond with ONLY 1 or 0):"
    )

    response = await client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": 0.0, "num_ctx": 2048, "num_predict": 2},
    )

    content = response["message"]["content"].strip()
    return 1 if "1" in content else 0


async def process_batch(df_subset, description_col, language, pattern):
    """
    Processes a batch of rows concurrently.

    :param df_subset: The subset of the DataFrame to process.
    :type df_subset: pandas.DataFrame
    :param description_col: The column containing the job description text.
    :type description_col: str
    :param language: The language to verify.
    :type language: str
    :param pattern: The regex pattern to use for the verification.
    :type pattern: str
    :return: The verification results.
    :rtype: list
    """
    client = AsyncClient()
    tasks = []

    for text in df_subset[description_col]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start, end = match.span()
            snippet = text[max(0, start - 200) : min(len(text), end + 200)]
            tasks.append(verify_single_row(client, snippet, language))
        else:

            async def return_zero():
                return 0

            tasks.append(return_zero())

    return await asyncio.gather(*tasks)


async def process_batch_tools(df_subset, description_col, tool, pattern, pbar=None):
    """
    Processes rows concurrently and updates a shared progress bar for every request.

    :param df_subset: The subset of the DataFrame to process.
    :type df_subset: pandas.DataFrame
    :param description_col: The column containing the job description text.
    :type description_col: str
    :param tool: The tool to verify.
    :type tool: str
    :param pattern: The regex pattern to use for the verification.
    :type pattern: str
    :param pbar: The progress bar to update.
    :type pbar: tqdm.asyncio.tqdm
    :return: The verification results.
    :rtype: list
    """
    client = AsyncClient()
    semaphore = asyncio.Semaphore(20) 
    tasks = []

    async def verified_with_progress(snippet, tool):
        async with semaphore:
            result = await verify_single_row_tools(client, snippet, tool)
            if pbar:
                pbar.update(1) 
            return result

    for text in df_subset[description_col]:
        match = re.search(pattern, str(text), re.IGNORECASE)
        if match:
            start, end = match.span()
            snippet = text[max(0, start - 300) : min(len(text), end + 300)]
            tasks.append(verified_with_progress(snippet, tool))
        else:
            async def auto_zero():
                if pbar:
                    pbar.update(1)
                return 0

            tasks.append(auto_zero())

    return await asyncio.gather(*tasks)


def checking_go_lang(df, description_col, potential_col, golang_col):
    """
    Verifies Go language usage using asynchronous parallel processing.

    :param df: The DataFrame containing job postings.
    :type df: pandas.DataFrame
    :param description_col: Column with job descriptions.
    :type description_col: str
    :param potential_col: Column for potential regex matches.
    :type potential_col: str
    :param golang_col: Column for confirmed Golang.
    :type golang_col: str
    :return: DataFrame with verified Go column.
    :rtype: pandas.DataFrame
    """
    df_copy = df.copy()
    df_copy["Go_verified"] = 0

    mask = (df_copy[potential_col] == 1) & (df_copy[golang_col] == 0)
    indices = df_copy[mask].index

    if len(indices) > 0:
        verified_results = asyncio.run(
            process_batch(
                df_copy.loc[mask],
                description_col,
                "go",
                r"(?<!\bto\s)\bgo\b(?!\s*to|\-)",
            )
        )
        df_copy.loc[mask, "Go_verified"] = verified_results
    df_copy.loc[df_copy[golang_col] == 1, "Go_verified"] = 1

    return df_copy


def checking_a_lang(df, description_col, potential_col):
    """
    Verifies a language usage using asynchronous parallel processing.

    :param df: The DataFrame containing job postings.
    :type df: pandas.DataFrame
    :param description_col: Column with job descriptions.
    :type description_col: str
    :param potential_col: Column for potential regex matches.
    :type potential_col: str
    :param golang_col: Column for confirmed Golang.
    :type golang_col: str
    :return: DataFrame with verified Go column.
    :rtype: pandas.DataFrame
    """
    df_copy = df.copy()

    lang_patterns = {
        "Python": r"\b(Python|Pyton)\b",
        "SQL": r"\b(SQL|MySQL|PostgreSQL|Postgres|MS\s*SQL|T-SQL|SQLite)\b",
        "Java": r"\bJava\b(?!\s*Script)",
        "JavaScript": r"\b(Java\s*Script|JS)\b",
        "TypeScript": r"\bType\s*Script\b",
        "C++": r"\b(C\+\+|C plus plus)(?![a-zA-Z0-9])",
        "C#": r"\b(C#|C Sharp)(?![a-zA-Z0-9])",
        "Objective-C": r"\bObjective[- ]C(?=[ ,.]|$)",
        "C": r"(?<![Cc]lass\s)(?<!Objective[- ])\bC(?![#+\-&*]| Sharp| plus plus|plusplus)(?=[ ,.]|$)",
        "R": r"\bR(?=[ ,.]|$)",
        "Golang": r"\bGolang\b",
        "Potential_Go": r"(?<!\bto\s)\bgo\b(?!\s*to|\-)",
        "Swift": r"\bSwift\b",
        "PHP": r"(?<![.])\bPHP\b",
        "Ruby": r"\bRuby\b",
        "Kotlin": r"\bKotlin\b",
        "Rust": r"\bRust\b",
        "Matlab": r"\b(Matlab|Mat lab)\b",
        "Scala": r"\bScala\b",
        "Perl": r"\bPerl\b",
        "Dart": r"\bDart\b",
        "Bash": r"\b(Bash|Shell|PowerShell)\b",
        "Assembly": r"\bAssembly\b",
    }
    verified_col = f"{potential_col}_verified"
    df_copy[verified_col] = 0

    mask = df_copy[potential_col] == 1
    indices = df_copy[mask].index

    if len(indices) > 0:
        verified_results = asyncio.run(
            process_batch(
                df_copy.loc[mask],
                description_col,
                potential_col,
                lang_patterns[potential_col],
            )
        )
        df_copy.loc[mask, verified_col] = verified_results

    return df_copy


def checking_a_lang(df, description_col, potential_col):
    """
    Verifies a language usage using asynchronous parallel processing.

    :param df: The DataFrame containing job postings.
    :type df: pandas.DataFrame
    :param description_col: Column with job descriptions.
    :type description_col: str
    :param potential_col: Column for potential regex matches.
    :type potential_col: str
    :param golang_col: Column for confirmed Golang.
    :type golang_col: str
    :return: DataFrame with verified Go column.
    :rtype: pandas.DataFrame
    """
    df_copy = df.copy()

    lang_patterns = {
        "Python": r"\b(Python|Pyton)\b",
        "SQL": r"\b(SQL|MySQL|PostgreSQL|Postgres|MS\s*SQL|T-SQL|SQLite)\b",
        "Java": r"\bJava\b(?!\s*Script)",
        "JavaScript": r"\b(Java\s*Script|JS)\b",
        "TypeScript": r"\bType\s*Script\b",
        "C++": r"\b(C\+\+|C plus plus)(?![a-zA-Z0-9])",
        "C#": r"\b(C#|C Sharp)(?![a-zA-Z0-9])",
        "Objective-C": r"\bObjective[- ]C(?=[ ,.]|$)",
        "C": r"(?<![Cc]lass\s)(?<!Objective[- ])\bC(?![#+\-&*]| Sharp| plus plus|plusplus)(?=[ ,.]|$)",
        "R": r"\bR(?=[ ,.]|$)",
        "Golang": r"\bGolang\b",
        "Potential_Go": r"(?<!\bto\s)\bgo\b(?!\s*to|\-)",
        "Swift": r"\bSwift\b",
        "PHP": r"(?<![.])\bPHP\b",
        "Ruby": r"\bRuby\b",
        "Kotlin": r"\bKotlin\b",
        "Rust": r"\bRust\b",
        "Matlab": r"\b(Matlab|Mat lab)\b",
        "Scala": r"\bScala\b",
        "Perl": r"\bPerl\b",
        "Dart": r"\bDart\b",
        "Bash": r"\b(Bash|Shell|PowerShell)\b",
        "Assembly": r"\bAssembly\b",
    }
    verified_col = f"{potential_col}_verified"
    df_copy[verified_col] = 0

    mask = df_copy[potential_col] == 1
    indices = df_copy[mask].index

    if len(indices) > 0:
        verified_results = asyncio.run(
            process_batch(
                df_copy.loc[mask],
                description_col,
                potential_col,
                lang_patterns[potential_col],
            )
        )
        df_copy.loc[mask, verified_col] = verified_results

    return df_copy


async def checking_a_tool(df, description_col, potential_col, pbar=None):
    """
    Verifies if a specific tool mention is a technical tool using LLM context.
    Designed for large-scale job post analysis on Mac M4 Pro.

    :param df: The DataFrame containing job postings.
    :type df: pd.DataFrame
    :param description_col: Column name with raw job descriptions.
    :type description_col: str
    :param potential_col: Binary column (0/1) from initial regex pass.
    :type potential_col: str
    :param pattern: The regex pattern used for the initial match.
    :type pattern: str
    :return: DataFrame with an added '{tool}_verified' column.
    :rtype: pd.DataFrame
    """
    tool_patterns = {
        "Excel": r"\b(Excel|Spreadsheets)\b",
        "Google_Sheets": r"\b(Google Sheets|G-Sheets)\b",
        "Fivetran": r"\bFivetran\b",
        "Airbyte": r"\bAirbyte\b",
        "dbt": r"\bdbt\b",
        "Snowflake": r"\bSnowflake\b",
        "BigQuery": r"\b(BigQuery|Big Query)\b",
        "Airflow": r"\b(Airflow|Apache Airflow)\b",
        "Prefect": r"\bPrefect\b",
        "Power_BI": r"\b(Power BI|PowerBI)\b",
        "Tableau": r"\bTableau\b",
        "Looker": r"\bLooker\b",
        "Git": r"\b(Git|GitHub|GitLab|Version Control)\b",
        "Docker": r"\b(Docker|Containers)\b",
        "Kubernetes": r"\b(Kubernetes|K8s)\b",
        "Terraform": r"\bTerraform\b",
        "AWS": r"\b(AWS|Amazon Web Services)\b",
        "Azure": r"\b(Azure)\b",
        "GCP": r"\b(GCP|Google Cloud)\b",
        "Databricks": r"\bDatabricks\b",
        "Kafka": r"\b(Kafka|Apache Kafka)\b",
        "Spark": r"\b(Spark|PySpark|Apache Spark)\b",
        "Monte_Carlo": r"\bMonte Carlo\b",
    }
    pattern = tool_patterns[potential_col]
    print(f"Verifying {potential_col}...")
    df_copy = df.copy()
    verified_col = f"{potential_col}_verified"
    df_copy[verified_col] = 0

    mask = df_copy[potential_col] == 1

    if mask.any():
        verified_results = await process_batch_tools(
            df[mask],
            description_col,
            potential_col,
            pattern,
            pbar=pbar,  
        )
        df_copy.loc[mask, verified_col] = verified_results

    return df_copy


async def verify_tool_parallel(tool, num_files=11):
    """
    Parallelizes tool verification across all data shards.
    """
    tasks = []

    print(f"--- Starting Parallel Verification for: {tool} ---")
    for i in range(1, num_files + 1):
        file_path = f"../data/processed/tools/jobs_proc_all_USA_2025_{i}.csv"

        if not os.path.exists(file_path):
            continue
        df_shard = pd.read_csv(file_path)
        df_matches = df_shard[df_shard[tool] == 1]

        if len(df_matches) > 0:
            tasks.append(checking_a_tool(df_matches, "description", tool))

    print(f"Gathering {len(tasks)} file-shards for concurrent processing...")
    results = await asyncio.gather(*tasks)
    if results:
        df_final = pd.concat(results, ignore_index=True)
        out_path = (
            f"../data/processed/tools/verification/jobs_proc_all_USA_2025_{tool}.csv"
        )
        df_final.to_csv(out_path, index=False)
        print(f"Done! Saved {len(df_final)} verified rows to {out_path}")
    else:
        print("No matches found to verify.")


def extract_programming_languages(df, description_col="description"):
    """
    Scrapes a given dataframe column and extracts top 20 programming languages using regex.
    Adds new columns to the dataframe for each language with binary values (1 or 0).

    :param df: The pandas DataFrame containing job postings.
    :type df: pd.DataFrame
    :param description_col: The name of the column containing the job description text.
    :type description_col: str
    :return: The original DataFrame with additional binary columns for each programming language.
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()
    language_patterns = {
        "Python": r"\b(Python|Pyton)\b",
        "SQL": r"\b(SQL|MySQL|PostgreSQL|Postgres|MS\s*SQL|T-SQL|SQLite)\b",
        "Java": r"\bJava\b(?!\s*Script)",
        "JavaScript": r"\b(Java\s*Script|JS)\b",
        "TypeScript": r"\bType\s*Script\b",
        "C++": r"\b(C\+\+|C plus plus)(?![a-zA-Z0-9])",
        "C#": r"\b(C#|C Sharp)(?![a-zA-Z0-9])",
        "Objective-C": r"\bObjective[- ]C(?=[ ,.]|$)",
        "C": r"(?<![Cc]lass\s)(?<!Objective[- ])\bC(?![#+\-&*]| Sharp| plus plus|plusplus)(?=[ ,.]|$)",
        "R": r"\bR(?=[ ,.]|$)",
        "Golang": r"\bGolang\b",
        "Potential_Go": r"(?<!\bto\s)\bgo\b(?!\s*to|\-)",
        "Swift": r"\bSwift\b",
        "PHP": r"(?<![.])\bPHP\b",
        "Ruby": r"\bRuby\b",
        "Kotlin": r"\bKotlin\b",
        "Rust": r"\bRust\b",
        "Matlab": r"\b(Matlab|Mat lab)\b",
        "Scala": r"\bScala\b",
        "Perl": r"\bPerl\b",
        "Dart": r"\bDart\b",
        "Bash": r"\b(Bash|Shell|PowerShell)\b",
        "Assembly": r"\bAssembly\b",
    }

    for lang, pattern in language_patterns.items():
        df_copy[lang] = df_copy[description_col].apply(
            lambda text: 1 if re.search(pattern, str(text), re.IGNORECASE) else 0
        )

    df_copy = checking_go_lang(df_copy, "description", "Potential_Go", "Golang")
    df_copy.rename(columns={"Go_Verified": "Go"}, inplace=True)

    return df_copy


def extract_job_titles(df, title_col="title"):
    """
    Standardizes job titles and extracts six binary indicator columns for key roles.


    :param df: DataFrame containing the job title column.
    :type df: pd.DataFrame
    :param title_col: Name of the column with raw job titles.
    :type title_col: str
    :return: DataFrame with 6 new binary columns added after the title column.
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()

    cleaned_titles = (
        df_copy[title_col]
        .astype("str")
        .str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True)
        .str.lower()
    )
    job_groups = {
        "manager": r"manager(?:s|ing)?",
        "engineer": r"engineer(?:s|ing)?",
        "analyst": r"analyst(?:s)?",
        "scientist": r"scientist(?:s)?|researcher(?:s)?",
        "developer": r"developer(?:s)?|developing|programmer(?:s|ing)?",
    }

    insert_pos = df_copy.columns.get_loc(title_col) + 1

    for i, (col_name, pattern) in enumerate(job_groups.items()):
        df_copy.insert(
            insert_pos + i,
            col_name,
            cleaned_titles.str.contains(pattern, regex=True)
            .fillna(False)
            .astype("int"),
        )
    return df_copy


def clean_us_states(state_series):
    """
    Standardizes US state names by mapping full names, abbreviations, and foreign translations.


    :param state_series: A pandas Series containing the raw state strings.
    :type state_series: pandas.Series
    :return: A pandas Series with standardized full state names.
    :rtype: pandas.Series
    """

    state_mapping = {
        "alabama": "Alabama",
        "al": "Alabama",
        "阿拉巴马州": "Alabama",
        "alaska": "Alaska",
        "ak": "Alaska",
        "arizona": "Arizona",
        "az": "Arizona",
        "Аризона": "Arizona",
        "arkansas": "Arkansas",
        "ar": "Arkansas",
        "california": "California",
        "ca": "California",
        "รัฐแคลิฟอร์เนีย": "California",
        "كاليفورنيا": "California",
        "colorado": "Colorado",
        "co": "Colorado",
        "connecticut": "Connecticut",
        "ct": "Connecticut",
        "delaware": "Delaware",
        "de": "Delaware",
        "district of columbia": "District of Columbia",
        "dc": "District of Columbia",
        "washington d.c.": "District of Columbia",
        "florida": "Florida",
        "fl": "Florida",
        "floride": "Florida",
        "รัฐฟلอริดา": "Florida",
        "georgia": "Georgia",
        "ga": "Georgia",
        "hawaii": "Hawaii",
        "hi": "Hawaii",
        "idaho": "Idaho",
        "id": "Idaho",
        "illinois": "Illinois",
        "il": "Illinois",
        "chicago": "Illinois",
        "indiana": "Indiana",
        "in": "Indiana",
        "iowa": "Iowa",
        "ia": "Iowa",
        "kansas": "Kansas",
        "ks": "Kansas",
        "kentucky": "Kentucky",
        "ky": "Kentucky",
        "louisiana": "Louisiana",
        "la": "Louisiana",
        "maine": "Maine",
        "me": "Maine",
        "maryland": "Maryland",
        "md": "Maryland",
        "मेरीलैंड": "Maryland",
        "massachusetts": "Massachusetts",
        "ma": "Massachusetts",
        "michigan": "Michigan",
        "mi": "Michigan",
        "मिशिगन": "Michigan",
        "รัฐมิชิแกน": "Michigan",
        "minnesota": "Minnesota",
        "mn": "Minnesota",
        "รัฐมินนิโซตา": "Minnesota",
        "mississippi": "Mississippi",
        "ms": "Mississippi",
        "missouri": "Missouri",
        "mo": "Missouri",
        "Миссури": "Missouri",
        "मिज़ूरी": "Missouri",
        "ميسوري": "Missouri",
        "montana": "Montana",
        "mt": "Montana",
        "nebraska": "Nebraska",
        "ne": "Nebraska",
        "nevada": "Nevada",
        "nv": "Nevada",
        "new hampshire": "New Hampshire",
        "nh": "New Hampshire",
        "new jersey": "New Jersey",
        "nj": "New Jersey",
        "new mexico": "New Mexico",
        "nm": "New Mexico",
        "new york": "New York",
        "ny": "New York",
        "nowy jork": "New York",
        "nueva york": "New York",
        "north carolina": "North Carolina",
        "nc": "North Carolina",
        "north dakota": "North Dakota",
        "nd": "North Dakota",
        "ohio": "Ohio",
        "oh": "Ohio",
        "ओहायो": "Ohio",
        "オハイオ": "Ohio",
        "oklahoma": "Oklahoma",
        "ok": "Oklahoma",
        "oregon": "Oregon",
        "or": "Oregon",
        "pennsylvania": "Pennsylvania",
        "pa": "Pennsylvania",
        "rhode island": "Rhode Island",
        "ri": "Rhode Island",
        "รัฐโรดไอแลนด์": "Rhode Island",
        "south carolina": "South Carolina",
        "sc": "South Carolina",
        "south dakota": "South Dakota",
        "sd": "South Dakota",
        "tennessee": "Tennessee",
        "tn": "Tennessee",
        "texas": "Texas",
        "tx": "Texas",
        "تكساس": "Texas",
        "utah": "Utah",
        "ut": "Utah",
        "vermont": "Vermont",
        "vt": "Vermont",
        "virginia": "Virginia",
        "va": "Virginia",
        "वर्जिनिया": "Virginia",
        "washington": "Washington",
        "wa": "Washington",
        "west virginia": "West Virginia",
        "wv": "West Virginia",
        "wisconsin": "Wisconsin",
        "wi": "Wisconsin",
        "wyoming": "Wyoming",
        "wy": "Wyoming",
        "puerto rico": "Puerto Rico",
        "pr": "Puerto Rico",
        "guam": "Guam",
        "gu": "Guam",
        "virgin islands": "Virgin Islands",
        "vi": "Virgin Islands",
        "u.s. virgin islands": "Virgin Islands",
    }

    def transform_value(value):
        if pd.isna(value):
            return np.nan
        lookup = str(value).lower().strip()
        return state_mapping.get(lookup, np.nan)

    return state_series.apply(transform_value)


def categorize_industry(name):
    """
    Maps 300+ detailed industries into 10 high-level strategic categories.
    """
    name = str(name).lower()

    if any(
        k in name
        for k in [
            "software",
            "it services",
            "information",
            "internet",
            "computer",
            "data",
            "web",
            "cyber",
            "artificial intelligence",
            "semiconductor",
            "telecom",
            "networking",
            "digital",
            "tech",
            "blockchain",
            "bi platform",
            "wireless",
        ]
    ):
        return "Tech, Data & Telecom"

    if any(
        k in name
        for k in [
            "manufacturing",
            "machinery",
            "automotive",
            "industrial",
            "defense",
            "aerospace",
            "chemical",
            "textile",
            "paper",
            "wood",
            "space",
            "automation",
            "equipment",
            "robotics",
            "shipbuilding",
            "metal",
            "packaging",
        ]
    ):
        return "Manufacturing, Industrial & Defense"

    if any(
        k in name
        for k in [
            "financial",
            "insurance",
            "banking",
            "investment",
            "venture",
            "accounting",
            "tax",
            "fintech",
            "capital",
            "credit",
            "real estate",
            "property",
            "holding companies",
            "pension",
        ]
    ):
        return "Finance, Insurance & Real Estate"

    if any(
        k in name
        for k in [
            "medical",
            "health",
            "biotechnology",
            "pharma",
            "clinical",
            "hospital",
            "nursing",
            "outpatient",
            "wellness",
            "medicine",
            "veterinary",
            "therapists",
        ]
    ):
        return "Healthcare, Pharma & Wellness"

    if any(
        k in name
        for k in [
            "retail",
            "wholesale",
            "consumer",
            "food",
            "beverage",
            "furniture",
            "apparel",
            "luxury",
            "cosmetics",
            "e-commerce",
            "grocery",
            "marketplace",
            "fashion",
            "farming",
            "agriculture",
            "fisheries",
            "dairy",
            "distilleries",
            "breweries",
        ]
    ):
        return "Consumer, Retail & Agriculture"

    if any(
        k in name
        for k in [
            "consulting",
            "staffing",
            "recruiting",
            "legal",
            "human resources",
            "outsourcing",
            "strategic",
            "management",
            "professional services",
            "research",
            "advertising",
            "marketing",
            "law",
            "security",
            "facilities",
            "translation",
        ]
    ):
        return "Professional, Legal & Business Services"

    if any(
        k in name
        for k in [
            "energy",
            "utility",
            "utilities",
            "oil",
            "gas",
            "renewable",
            "power",
            "mining",
            "environmental",
            "water",
            "waste",
            "solar",
            "sustainability",
        ]
    ):
        return "Energy, Utilities & Environment"

    if any(
        k in name
        for k in [
            "media",
            "publishing",
            "broadcast",
            "design",
            "arts",
            "entertainment",
            "museum",
            "musicians",
            "creative",
            "writing",
            "photography",
            "games",
            "video",
            "news",
            "sports",
            "blogs",
        ]
    ):
        return "Media, Entertainment & Arts"

    if any(
        k in name
        for k in [
            "education",
            "government",
            "public",
            "non-profit",
            "ngo",
            "charity",
            "philanthropy",
            "military",
            "international affairs",
            "political",
            "civic",
            "social services",
            "fundraising",
            "e-learning",
        ]
    ):
        return "Education, Government & Non-profit"

    if any(
        k in name
        for k in [
            "transportation",
            "logistics",
            "travel",
            "hospitality",
            "restaurant",
            "hotel",
            "warehouse",
            "shipping",
            "aviation",
            "airlines",
            "construction",
            "delivery",
            "building",
            "landscaping",
            "civil engineering",
        ]
    ):
        return "Logistics, Travel & Construction"

    return "Miscellaneous"


def create_faang_df(df):
    """
    Extracts high-precision FAANG data using non-capturing regex groups.


    :param df: The main DataFrame containing job postings.
    :type df: pandas.DataFrame
    :return: A DataFrame filtered strictly for the core Big Tech entities.
    :rtype: pandas.DataFrame
    """
    patterns = {
        "Meta": r"\bMeta\b(?!\s+(?:Care|Sensing|Resources|Group))\b|\b(?:Facebook|Instagram|WhatsApp|Oculus)\b",
        "Google": r"\b(?:Google|Alphabet|YouTube|DeepMind)\b",
        "Amazon": r"\b(?:Amazon|AWS)\b",
        "Apple": r"\bApple\b",
        "Netflix": r"\bNetflix\b",
    }

    df["faang_category"] = None

    for official_name, regex_pattern in patterns.items():
        mask = df["company_name"].str.contains(
            regex_pattern, case=False, na=False, regex=True
        )
        df.loc[mask, "faang_category"] = official_name

    df_faang = df[df["faang_category"].notna()].copy()

    return df_faang


def create_mango_df(df):
    """
    Extracts high-precision MANGO data (Microsoft, Apple, Nvidia, Google, OpenAI) using regex.


    :param df: The main DataFrame containing job postings.
    :type df: pandas.DataFrame
    :return: A DataFrame filtered strictly for the defined MANGO and AI-pioneer entities.
    :rtype: pandas.DataFrame
    """
    patterns = {
        "Microsoft": r"\b(?:Microsoft|LinkedIn|GitHub|Azure)\b",
        "Meta": r"\bMeta\b(?!\s+(?:Care|Sensing|Resources|Group))\b|\b(?:Facebook|Instagram|WhatsApp)\b",
        "Apple": r"\bApple\b",
        "Anthropic": r"\b(?:Anthropic)\b",
        "Nvidia": r"\b(?:Nvidia|GeForce|Mellanox)\b",
        "Google/DeepMind": r"\b(?:Google|Alphabet|YouTube|DeepMind|Waymo)\b",
        "OpenAI": r"\bOpenAI\b",
    }

    df["mango_category"] = None

    for official_name, regex_pattern in patterns.items():
        mask = df["company_name"].str.contains(
            regex_pattern, case=False, na=False, regex=True
        )
        df.loc[mask, "mango_category"] = official_name

    df_mango = df[df["mango_category"].notna()].copy()

    return df_mango


def get_dominant_language_by_state(df, languages):
    """
    Identifies the most frequent programming language required in each state.


    :param df: The main DataFrame containing state codes and language data.
    :type df: pandas.DataFrame
    :param languages: List of columns representing the languages to check.
    :type languages: list
    :return: A DataFrame with state_code and the top_language.
    :rtype: pandas.DataFrame
    """
    state_lang_totals = df.groupby("state_code")[languages].sum()

    dominant_langs = state_lang_totals.idxmax(axis=1).reset_index()
    dominant_langs.columns = ["state_code", "top_language"]

    return dominant_langs


async def run_full_verification(tools_list, shard_count=11, batch_size=5000):
    """
    Analyzes the ENTIRE dataset. Processes shards sequentially to save RAM,
    but batches rows within shards to keep the LLM busy without crashing.
    """
    for tool in tools_list:
        print(f"\nFull Verification: {tool}")

        for i in range(1, shard_count + 1):
            path = f"../data/processed/tools/jobs_proc_all_USA_2025_{i}.csv"
            if not os.path.exists(path):
                continue

            lazy_shard = pl.scan_csv(path).filter(pl.col(tool) == 1)
            shard_matches_count = lazy_shard.select(pl.len()).collect().item()
            print(f"Shard {i}: Found {shard_matches_count:,} total matches for {tool}")

            shard_results = []

            with atqdm(
                total=shard_matches_count, desc=f"   Shard {i} Progress", unit="row"
            ) as pbar:
                for offset in range(0, shard_matches_count, batch_size):
                    df_batch = (
                        lazy_shard.slice(offset, batch_size).collect().to_pandas()
                    )

                    verified_batch_pd = await checking_a_tool(
                        df_batch, "description", tool, pbar=pbar
                    )

                    shard_results.append(pl.from_pandas(verified_batch_pd))
                    gc.collect()
            if shard_results:
                df_shard_final = pl.concat(shard_results)
                out_path = f"../data/processed/tools/verification/verified_{tool}_shard_{i}.csv"
                df_shard_final.write_csv(out_path)
                print(f"Saved verified shard to: {out_path}")

            del shard_results
            gc.collect()

    print(f"\n All verification tasks for {tools_list} completed.")


def extract_tech_tools(df, description_col="description"):
    """
    Extracts high-priority tech tools across Data Engineering, Analytics,
    and Infrastructure layers for a 2026 job market analysis.

    :param df: The pandas DataFrame containing job postings.
    :type df: pd.DataFrame
    :param description_col: The name of the column containing the job description text.
    :type description_col: str
    :return: The original DataFrame with binary columns for foundational and modern tools.
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()

    tool_patterns = {
        "Excel": r"\b(Excel|Spreadsheets|MS Excel)\b",
        "Google_Sheets": r"\b(Google Sheets|G-Sheets)\b",
        "Fivetran": r"\bFivetran\b",
        "Airbyte": r"\bAirbyte\b",
        "dbt": r"\bdbt\b",
        "Snowflake": r"\bSnowflake\b",
        "BigQuery": r"\b(BigQuery|Big Query)\b",
        "Airflow": r"\b(Airflow)\b",
        "Prefect": r"\bPrefect\b",
        "Power_BI": r"\b(Power BI|PowerBI|MS Power BI)\b",
        "Tableau": r"\bTableau\b",
        "Looker": r"\bLooker\b",
        "Git": r"\b(Git|GitHub|GitLab|Version Control)\b",
        "Docker": r"\b(Docker|Containers)\b",
        "Kubernetes": r"\b(Kubernetes|K8s)\b",
        "Terraform": r"\bTerraform\b",
        "AWS": r"\b(AWS|Amazon Web Services)\b",
        "Azure": r"\b(Azure)\b",
        "GCP": r"\b(GCP|Google Cloud)\b",
        "Databricks": r"\bDatabricks\b",
        "Kafka": r"\b(Kafka)\b",
        "Spark": r"\b(Spark|PySpark)\b",
        "Monte_Carlo": r"\bMonte Carlo\b",
    }

    descriptions = df_copy[description_col].astype("str").fillna("")

    for tool, pattern in tool_patterns.items():
        df_copy[tool] = descriptions.apply(
            lambda text: 1 if re.search(pattern, text, re.IGNORECASE) else 0
        )

    return df_copy
