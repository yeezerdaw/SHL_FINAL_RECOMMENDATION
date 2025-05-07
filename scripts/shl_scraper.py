import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

def scrape_shl_assessment_details(url):
    """
    Scrape SHL assessment fields: Description, Job Levels, Assessment Length, Languages, and Test Types
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')

        # Helper to extract text from <p> following a given <h4>
        def extract_section(heading_text):
            section_div = soup.find('h4', text=heading_text)
            if section_div:
                next_p = section_div.find_next('p')
                if next_p:
                    return next_p.get_text(strip=True)
            return None

        # Extracting values
        description = extract_section("Description")
        job_levels = extract_section("Job levels")
        assessment_length = extract_section("Assessment length")
        languages = extract_section("Languages")

        # Extract test types (e.g., C, P, A, B)
        test_type_spans = soup.find_all('span', class_='product-catalogue__key')
        test_types = ', '.join([span.get_text(strip=True) for span in test_type_spans]) if test_type_spans else None

        return {
            "Description": description,
            "Job Levels": job_levels,
            "Assessment Length": assessment_length,
            "Languages": languages,
            "Test Types": test_types
        }

    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {str(e)}")
        return {key: None for key in ["Description", "Job Levels", "Assessment Length", "Languages", "Test Types"]}
    except Exception as e:
        print(f"Error parsing {url}: {str(e)}")
        return {key: None for key in ["Description", "Job Levels", "Assessment Length", "Languages", "Test Types"]}

def enrich_assessment_descriptions(input_csv, output_csv, delay=2.5):
    """
    Enrich SHL assessment data with scraped details
    """
    df = pd.read_csv("shl_complete_assessments.csv")

    # Ensure new columns exist
    columns_to_add = ["Description", "Job Levels", "Assessment Length", "Languages", "Test Types"]
    for col in columns_to_add:
        if col not in df.columns:
            df[col] = None

    total_urls = len(df[df['URL'].notna()])
    success_count = 0

    print(f"Starting to process {total_urls} assessment URLs...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(row['URL']) or not pd.isna(row['Description']):
            continue

        details = scrape_shl_assessment_details(row['URL'])

        for key in details:
            if details[key]:
                df.at[idx, key] = details[key]

        if details["Description"]:
            success_count += 1

        time.sleep(delay)

    df.to_csv(output_csv, index=False)

    print(f"\nProcessing complete! Results:")
    print(f"- Total assessments processed: {total_urls}")
    print(f"- Successfully extracted descriptions: {success_count}")
    print(f"- Success rate: {success_count / total_urls:.1%}")
    print(f"- Saved to: {output_csv}")
    
    return df

# Example usage
if __name__ == "__main__":
    enriched_data = enrich_assessment_descriptions(
        input_csv="shl_complete_assessments.csv",
        output_csv="shl_assessments_kms.csv",
        delay=2.5
    )
