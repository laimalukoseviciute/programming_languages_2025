import pytest
import pandas as pd
from src.feature_utils import extract_programming_languages

def test_extract_programming_languages_accuracy():
    """
    Verifies that the extraction function correctly identifies languages 
    and avoids common false positives in job descriptions.


    :return: None
    """
    test_data = {
            "title": ["Data Scientist", "Objective programmer", "Java developer",
                    "Not PHP developer", "C Dev", "Data Analyst",
                    "Data Scientist", "Backend Dev", "Data Analyst",
                    "Data Engineer", "C Dev", "Programmer"],
            "description": [
                "We need a Python and SQL expert.",
                "We need an expert in java,sql, and python. Good python knowlage is an analytic plus php.",           
                "Must have experience with Java and node.js. Knowing Swift is an advantage.",
                "We (www.company.com/index.php) need experienced programmers with python,r, and MySQL.",   
                "Looking for a C++ and C# developer.",         
                "Expert in R for statistical research.",     
                "Go is our primary language for backend. But we also need an objective Rust \
                    pragrammer, with a good knowledge of C.",     
                "No languages mentioned here. We just need a person in our R&D department. With swifty \
                    hands to assemble our rusty furniture and be objective",
                "We are looking for a very universal programer with experience in Kotlin, Scala, Matlab,\
                    Perl, Dart. Knowing C is a plus, but not exotic plus plus.",
                "We need engineers with knowlage in Assembly, Objective-C, Bash, PHP, Ruby, TypeScript,\
                    and C sharp",
                "Objective C and C plus plus is a must. Golang and Java Script is a +.",
                "Type Script is a must. Knowlage in PowerShell,  would be an advantage.",
            ]
        }
    df_test = pd.DataFrame(test_data)


    result_df = extract_programming_languages(df_test, "description")

      
    assert result_df.loc[1, "Python"] == 1
    assert result_df["Python"].sum() == 3
    assert result_df.loc[0, "SQL"] == 1
    
    # Java vs JavaScript distinction
    assert result_df.loc[1, "Java"] == 1
    assert result_df.loc[1, "JavaScript"] == 1  # Matches node.js or JS
    
    # C++ and C# special characters
    assert result_df.loc[2, "C++"] == 1
    assert result_df.loc[2, "C#"] == 1
    
    # R boundary check (Should not match 'R' inside 'research')
    assert result_df.loc[3, "R"] == 1
    
    # Go language check
    assert result_df.loc[4, "Go"] == 1
    
    # Negative case (No languages should be 1)
    languages_to_check = ["Python", "Java", "Rust", "Swift"]
    for lang in languages_to_check:
        assert result_df.loc[5, lang] == 0


def test_dataframe_structure_integrity():
    """
    Ensures that the function returns the original columns plus 20 language columns.


    :return: None
    """
    data = {"description": ["Test text"], "other_col": [123]}
    df = pd.DataFrame(data)
    
    result_df = extract_programming_languages(df)
    
    # Check if original columns still exist
    assert "description" in result_df.columns
    assert "other_col" in result_df.columns
   
    assert len(result_df.columns) == 22