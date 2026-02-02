import numpy as np
import pandas as pd
import re
import ollama
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import re
import ollama
from concurrent.futures import ThreadPoolExecutor

# 1. PRE-COMPILE REGEX FOR SPEED
# This pattern excludes the most common "noise" before the LLM is even touched.
GO_CLEAN_PATTERN = re.compile(
    r"(?i)(?<!to\s)(?<!on\s)(?<!the\s)\b(go)\b(?!\s+to)(?!\-to)(?!\s+further)(?!\s+live)"
)


def process_batch_worker(rows, description_col):
    """Processes a small chunk of rows to keep the LLM pipeline saturated."""
    results = []
    for idx, row in rows:
        text = str(row[description_col])
        match = GO_CLEAN_PATTERN.search(text)

        if not match:
            results.append((idx, 0))
            continue

        # Narrow context window to save tokens (speed)
        snippet = text[max(0, match.start() - 150) : min(len(text), match.end() + 150)]

        # Hyper-short prompt for faster token generation
        prompt = (
            f"Text: {snippet}\nIs 'Go' the Google programming language? Answer 1 or 0:"
        )

        try:
            # Using the 1B model is significantly faster for binary classification
            response = ollama.generate(model="llama3.2:1b", prompt=prompt)
            digit = re.search(r"[01]", response["response"])
            results.append((idx, int(digit.group(0)) if digit else 0))
        except:
            results.append((idx, 0))
    return results


def verify_go_turbo(df, description_col, go_col, chunk_size=20, max_workers=12):
    """
    The fastest implementation for M4 Pro.
    Splits the work into chunks and processes them in parallel.
    """
    df_potential = df[df[go_col] == 1].copy()
    if df_potential.empty:
        return pd.Series(0, index=df.index)

    rows = list(df_potential.iterrows())
    # Split into small chunks for the workers
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

    final_map = {}
    print(f"Turbo-charging verification for {len(df_potential)} rows...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_batch_worker, chunk, description_col)
            for chunk in chunks
        ]
        for future in futures:
            for idx, val in future.result():
                final_map[idx] = val

    return pd.Series(final_map).reindex(df.index, fill_value=0)


def process_single_go_row(row_tuple, description_col):
    """Worker function for a single LLM call."""
    idx, row = row_tuple
    text = str(row[description_col])

    # Quick regex to grab context window
    go_pattern = r"(?i)(?<!to\s)\b(go|golang)\b(?!\s+to)(?!\-to)"
    match = re.search(go_pattern, text)

    if match:
        start = max(0, match.start() - 250)
        end = min(len(text), match.end() + 250)
        snippet = text[start:end]
    else:
        snippet = text[:500]

    prompt = f"""
        [INST]
        You are a technical recruiter. Analyze the job description snippet provided.
        Determine if there are any mentions of "Go" (or "Golang") 
        that are listed as a required technical skill/programming language.

        Rules:
        - Ignore common phrases like "go-to", "on the go", "go through", "to go", or "go-live".
        - Ignore URLs/Links containing "/go/".
        - Focus on technical stacks (e.g., "Python, Go, Java").
        - Do not explain your answer. Just return '1' or '0'.
        
        Return '1' if "Go" is mentioned as the programming language.
        Return '0' if "Go" is not mentioned as the programming language or is common English word, phrase, or part of a URL.

        Snippet: "{snippet}"
        [/INST]
        Answer (1 or 0):"""

    try:
        # Use a smaller/faster model like llama3.2:1b for massive speedups if needed
        response = ollama.generate(model="llama3.2", prompt=prompt)
        digit = re.search(r"\d", response["response"])
        return idx, (int(digit.group(0)) if digit else 0)
    except:
        return idx, 0


def verify_go_parallel(df, description_col, go_col, max_workers=20):
    """Runs LLM verification in parallel threads."""
    df_potential = df[df[go_col] == 1].copy()
    if df_potential.empty:
        return pd.Series(0, index=df.index)

    print(
        f"Parallelizing LLM verification for {len(df_potential)} rows using {max_workers} workers..."
    )

    # Using ThreadPoolExecutor to handle concurrent Ollama calls
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Pass (index, row) tuples to the worker
        future_to_row = {
            executor.submit(process_single_go_row, (idx, row), description_col): idx
            for idx, row in df_potential.iterrows()
        }

        for future in future_to_row:
            idx, verified_val = future.result()
            results[idx] = verified_val

    # Map results back to the original dataframe index
    return pd.Series(results).reindex(df.index, fill_value=0)


def extract_programming_languages_fast(df, description_col="description"):
    df_copy = df.copy()

    # Use non-capturing groups (?:...) to avoid UserWarnings and save memory
    language_patterns = {
        "Python": r"\b(?:Python|Pyton)\b",
        "SQL": r"\b(?:SQL|MySQL|PostgreSQL|Postgres|MS\s*SQL|T-SQL|SQLite)\b",
        "Java": r"\bJava\b(?!\s*Script)",
        "JavaScript": r"\b(?:Java\s*Script|JS)\b",
        "TypeScript": r"\bType\s*Script\b",
        "C++": r"\b(?:C\+\+|C plus plus)(?![a-zA-Z0-9])",
        "C#": r"\b(?:C#|C Sharp)(?![a-zA-Z0-9])",
        "Objective-C": r"\bObjective[- ]C(?=[ ,.]|$)",
        "C": r"(?<!Objective[- ])\bC(?![#+]| Sharp| plus plus)(?=[ ,.]|$)",
        "R": r"\bR(?=[ ,.]|$)",
        "Golang": r"\bGolang\b",
        "Potential_Go": r"(?<!\bto\s)\bgo\b(?!\s*to|\-)",
        "Swift": r"\bSwift\b",
        "PHP": r"(?<![.])\bPHP\b",
        "Ruby": r"\bRuby\b",
        "Kotlin": r"\bKotlin\b",
        "Rust": r"\bRust\b",
        "Bash": r"\b(?:Bash|Shell|PowerShell)\b",
    }

    for lang, pattern in language_patterns.items():
        df_copy[lang] = (
            df_copy[description_col]
            .str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
            .astype(int)
        )

    df_copy["Go"] = verify_go_turbo(
        df_copy, description_col, "Potential_Go", max_workers=20
    )

    df_copy.drop(columns=["Potential_Go"], inplace=True)
    return df_copy


def verify_go_with_llm(df, description_col="description", go_col="Potential_Go"):
    """
    Validates rows where 'Go' == 1 using a local LLM to filter out linguistic noise.

    :param df: DataFrame containing potential 'Go' matches.
    :param description_col: Column name for job descriptions.
    :return: DataFrame with an added 'Go_Verified' column.
    """

    # 1. Filter to only rows that Regex flagged as 1
    df_potential_go = df[df[go_col] == 1].copy()

    if df_potential_go.empty:
        print("No 'Go' matches found to verify.")
        return df_potential_go

    verified_results = []

    print(f"Starting LLM verification for {len(df_potential_go)} rows...")

    for index, row in df_potential_go.iterrows():
        # HELPER: Get context window around the word 'Go' instead of just the start
        text = str(row[description_col])
        match = re.search(r"\b(go|golang)\b", text, re.IGNORECASE)

        if match:
            start = max(0, match.start() - 250)
            end = min(len(text), match.end() + 250)
            snippet = text[start:end]
        else:
            snippet = text[:500]  # Fallback

        prompt = f"""
        [INST]
        You are a technical recruiter. Analyze the job description snippet provided.
        Determine if there are any mentions of "Go" (or "Golang") 
        that are listed as a required technical skill/programming language.

        Rules:
        - Ignore common phrases like "go-to", "on the go", "go through", "to go", or "go-live".
        - Ignore URLs/Links containing "/go/".
        - Focus on technical stacks (e.g., "Python, Go, Java").
        - Do not explain your answer. Just return '1' or '0'.
        
        Return '1' if "Go" is mentioned as the programming language.
        Return '0' if "Go" is not mentioned as the programming language or is common English word, phrase, or part of a URL.

        Snippet: "{snippet}"
        [/INST]
        Answer (1 or 0):"""

        try:
            response = ollama.generate(model="llama3.2", prompt=prompt)
            raw_response = response["response"].strip()

            # Robust extraction: find the first digit in the response
            digit_match = re.search(r"\d", raw_response)
            if digit_match:
                verified_results.append(int(digit_match.group(0)))
            else:
                verified_results.append(0)

        except Exception as e:
            print(f"Error at index {index}: {e}")
            verified_results.append(0)

    df_potential_go["Go_Verified"] = verified_results
    return df_potential_go


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
        "C": r"(?<!Objective[- ])\bC(?![#+]| Sharp| plus plus)(?=[ ,.]|$)",
        "R": r"\bR(?=[ ,.]|$)",
        "Potential_Go": r"\b(Go|Golang)\b",
        "Swift": r"\bSwift\b",
        "PHP": r"(?<![.])\bPHP\b",
        "Ruby": r"\bRuby\b",
        "Kotlin": r"\bKotlin\b",
        "Rust": r"\bRust\b",
        "Matlab": r"\bMatlab\b",
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

    df_copy = verify_go_with_llm(df_copy, description_col="description")
    df_copy.drop(columns=["Potential_Go"], inplace=True)
    df_copy.rename(columns={"Go_Verified": "Go"}, inplace=True)

    return df_copy


def extract_technical_skills(df, description_col="description"):
    """
    Extracts programming languages, cloud platforms, and tools from job descriptions.
    Adds binary columns for each technical skill found.


    :param df: The pandas DataFrame containing job postings.
    :type df: pd.DataFrame
    :param description_col: The column containing the job description text.
    :type description_col: str
    :return: A copy of the DataFrame with additional binary skill columns.
    :rtype: pandas.DataFrame
    """
    df_copy = df.copy()

    # We organize patterns by category for better maintainability
    skill_patterns = {
        # --- Cloud Platforms ---
        "AWS": r"\b(AWS|Amazon\s*Web\s*Services)\b",
        "Google Cloud": r"\b(GCP|Google\s*Cloud|Google\s*Cloud\s*Platform|BigQuery|Big\s*Query)\b",
        "Azure": r"\bAzure\b",
        "Snowflake": r"\bSnowflake\b",
        "Databricks": r"\bDatabricks\b",
        # --- Data & Engineering Tools ---
        "dbt": r"\dbt\b",
        "Informatica": r"\Informatica\b",
        "Docker": r"\bDocker\b",
        "Kubernetes": r"\b(Kubernetes|K8s)\b",
        "Git": r"\b(Git|GitHub|GitLab)\b",
        "Airflow": r"\b(Airflow|Apache\s*Airflow)\b",
        "Spark": r"\b(Spark|PySpark)\b",
        "Terraform": r"\bTerraform\b",
        "Kafka": r"\bKafka\b",
        # --- Methodologies & Project Management ---
        "Agile": r"\b(Agile|Scrum|Kanban)\b",
        "DevOps": r"\bDevOps\b",
        "CI/CD": r"\b(CI/CD|Continuous\s*Integration|Continuous\s*Deployment)\b",
        "Jira": r"\bJira\b",
        "Confluence": r"\bConfluence\b",
    }

    # Iterate through all skills and create binary columns
    for skill, pattern in skill_patterns.items():
        # Using vectorized str.contains is faster than .apply(lambda) for larger datasets
        df_copy[skill] = (
            df_copy[description_col]
            .str.contains(pattern, case=False, na=False, regex=True)
            .astype(int)
        )

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

    # 1. Pre-cleaning: remove special characters and lowercase
    # This ensures "Data-Scientist" and "Data Scientist" match the same pattern
    cleaned_titles = (
        df_copy[title_col]
        .astype("str")
        .str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True)
        .str.lower()
    )

    # Using non-capturing groups (?:...) to silence UserWarnings and optimize speed
    job_groups = {
        "manager": r"manager(?:s|ing)?",
        "engineer": r"engineer(?:s|ing)?",
        "analyst": r"analyst(?:s)?",
        "scientist": r"scientist(?:s)?|researcher(?:s)?",
        "developer": r"developer(?:s)?|developing|programmer(?:s|ing)?",
    }

    insert_pos = df_copy.columns.get_loc(title_col) + 1

    for i, (col_name, pattern) in enumerate(job_groups.items()):
        # Applying the updated pattern to create binary indicators
        df_copy.insert(
            insert_pos + i,
            col_name,
            cleaned_titles.str.contains(pattern, regex=True)
            .fillna(False)
            .astype("int"),
        )

    return df_copy
