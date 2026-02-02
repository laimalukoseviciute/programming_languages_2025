import re
import asyncio
import pandas as pd
import numpy as np
import re
from ollama import AsyncClient
import nest_asyncio

nest_asyncio.apply()


async def verify_single_row(client, snippet, language):
    """
    Sends a single request to Ollama asynchronously.
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
    Asynchronously classifies if a term in a snippet is a technical tool using Llama 3.2.

    :param client: The asynchronous Ollama client instance.
    :type client: ollama.AsyncClient
    :param snippet: The job description text fragment.
    :type snippet: str
    :param tool: The specific tool to verify.
    :type tool: str
    :return: 1 if the tool is used in a technical context, 0 otherwise.
    :rtype: int
    """
    
    system_content = (
        "You are a technical data annotator. Your task is to identify if a term refers to a "
        "software tool, framework, or library within a job description.\n\n"
        "Rules:\n"
        "- Score 1: Used as a tech tool (e.g., 'Experience in Go', 'dbt developer').\n"
        "- Score 0: Used as a common verb/noun, URL, or typo (e.g., 'Must go to', '/git/repo').\n"
        "Respond with exactly one character: '1' or '0'."
    )

    user_content = (
        f"Target Tool: \"{tool}\"\n"
        f"Snippet: \"{snippet}\"\n"
        "Answer (1 or 0):"
    )

    response = await client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        options={
            "temperature": 0.0,  
            "num_ctx": 1024,   
            "num_predict": 2   
        },
    )
    
    content = response["message"]["content"].strip()
    
    return 1 if "1" in content else 0


async def process_batch(df_subset, description_col, language, pattern):
    """
    Processes a batch of rows concurrently.
    """
    client = AsyncClient()
    tasks = []

    for text in df_subset[description_col]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start, end = match.span()
            snippet = text[max(0, start - 100) : min(len(text), end + 100)]
            tasks.append(verify_single_row(client, snippet, language))
        else:

            async def return_zero():
                return 0

            tasks.append(return_zero())

    return await asyncio.gather(*tasks)


async def process_batch_tools(df_subset, description_col, tool, pattern):
    """
    Processes a batch of rows concurrently.
    """
    client = AsyncClient()
    tasks = []

    for text in df_subset[description_col]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start, end = match.span()
            snippet = text[max(0, start - 100) : min(len(text), end + 100)]
            tasks.append(verify_single_row_tools(client, snippet, tool))
        else:

            async def return_zero():
                return 0

            tasks.append(return_zero())

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




async def checking_a_tool(df, description_col, potential_col):
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
        # --- Foundational & Spreadsheets ---
        "Excel": r"\b(Excel|Spreadsheets)\b",
        "Google_Sheets": r"\b(Google Sheets|G-Sheets)\b",
        
        # --- Modern Data Stack (Core) ---
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
        
        # --- Engineering & Infrastructure ---
        "Git": r"\b(Git|GitHub|GitLab|Version Control)\b",
        "Docker": r"\b(Docker|Containers)\b",
        "Kubernetes": r"\b(Kubernetes|K8s)\b",
        "Terraform": r"\bTerraform\b",
        
        # --- Cloud Platforms ---
        "AWS": r"\b(AWS|Amazon Web Services)\b",
        "Azure": r"\b(Azure)\b",
        "GCP": r"\b(GCP|Google Cloud)\b",
        
        # --- Emerging 2026 Trends ---
        "Databricks": r"\bDatabricks\b",
        "Kafka": r"\b(Kafka|Apache Kafka)\b",
        "Spark": r"\b(Spark|PySpark|Apache Spark)\b",
        "Monte_Carlo": r"\bMonte Carlo\b"
    }
    pattern = tool_patterns[potential_col]
    print(f"Verifying {potential_col}...")
    df_copy = df.copy()
    verified_col = f"{potential_col}_verified"
    df_copy[verified_col] = 0

    # Filter only for rows where regex initially found a potential match
    mask = df_copy[potential_col] == 1
    
    if mask.any():
        # process_batch_tools is your async function that calls verify_single_row_tools
        verified_results = await process_batch_tools(
            df_copy[mask], 
            description_col, 
            potential_col, 
            pattern
        )
        df_copy.loc[mask, verified_col] = verified_results

    return df_copy


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
        "Amazon": r"\b(?:Amazon|AWS|Whole Foods)\b",
        "Apple": r"\bApple\b",
        "Microsoft": r"\b(?:Microsoft|LinkedIn|GitHub|Azure)\b",
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
        # --- Foundational & Spreadsheets ---
        "Excel": r"\b(Excel|Spreadsheets|MS Excel)\b",
        "Google_Sheets": r"\b(Google Sheets|G-Sheets)\b",
        
        # --- Modern Data Stack (Core) ---
        "Fivetran": r"\bFivetran\b",
        "Airbyte": r"\bAirbyte\b",
        "dbt": r"\bdbt\b",
        "Snowflake": r"\bSnowflake\b",
        "BigQuery": r"\b(BigQuery|Big Query)\b",
        "Airflow": r"\b(Airflow|Apache Airflow)\b",
        "Prefect": r"\bPrefect\b",
        "Power_BI": r"\b(Power BI|PowerBI|MS Power BI)\b",
        "Tableau": r"\bTableau\b",
        "Looker": r"\bLooker\b",
        
        # --- Engineering & Infrastructure ---
        "Git": r"\b(Git|GitHub|GitLab|Version Control)\b",
        "Docker": r"\b(Docker|Containers)\b",
        "Kubernetes": r"\b(Kubernetes|K8s)\b",
        "Terraform": r"\bTerraform\b",
        
        # --- Cloud Platforms ---
        "AWS": r"\b(AWS|Amazon Web Services)\b",
        "Azure": r"\b(Azure|Microsoft Azure)\b",
        "GCP": r"\b(GCP|Google Cloud Platform|Google Cloud)\b",
        
        # --- Emerging 2026 Trends ---
        "Databricks": r"\bDatabricks\b",
        "Kafka": r"\b(Kafka|Apache Kafka)\b",
        "Spark": r"\b(Spark|PySpark|Apache Spark)\b",
        "Monte_Carlo": r"\bMonte Carlo\b"
    }

    # Optimization: Convert to string and fill NaNs once
    descriptions = df_copy[description_col].astype("str").fillna("")

    for tool, pattern in tool_patterns.items():
        # Case-insensitive regex for maximum coverage
        df_copy[tool] = descriptions.apply(
            lambda text: 1 if re.search(pattern, text, re.IGNORECASE) else 0
        )
        
    return df_copy