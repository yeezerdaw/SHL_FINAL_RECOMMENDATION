import json
import os
import time
from dotenv import load_dotenv
from groq import Groq
# requests is not strictly needed if only using Groq, but kept for consistency with original imports
# import requests
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

ASSESSMENT_FILE = "data/assessment_structure_filled.json"
OUTPUT_FILE = "data/assessment_structure_filled_groq_v2.json" # Changed output file name for v2
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192" # Powerful option on Groq

MAX_RETRIES = 3
RETRY_DELAY = 5

PROMPT_TEMPLATE = """
You are an expert HR data extraction engine.
Analyze the following job assessment description and extract the requested information.
Return your response as a single, valid JSON object. Do not add any explanatory text, comments, or markdown formatting before or after the JSON object.
The entire response should be only the JSON object itself.

The JSON object should have the following keys:
- "Skills": A list of strings, where each string is a distinct skill relevant to the job. These can be technical skills, soft skills, or tools/methodologies.
- "Domain": A single string representing the primary business domain or industry for this role (e.g., "Client Services", "Software Engineering", "Sales Operations").
- "SpokenLanguages": A list of strings, where each string is a spoken language explicitly mentioned or strongly implied as a communication requirement for the job.

[BEGIN DESCRIPTION]
{description}
[END DESCRIPTION]

Ensure the output is only the JSON object. For example:
{
  "Skills": ["Client Communication", "Project Management", "CRM Software"],
  "Domain": "Account Management",
  "SpokenLanguages": ["English", "Spanish (preferred)"]
}

If no specific information is found for a field, return an empty list for "Skills" and "SpokenLanguages", and an empty string or a generic value like "Not specified" for "Domain".
"""

def call_groq_llm_with_retry(client, description):
    prompt = PROMPT_TEMPLATE.format(description=description.strip())
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an extraction engine that responds only in valid JSON. Do not add any text outside the JSON object."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.05, # Even lower temperature for stricter adherence to format
        "max_tokens": 1024,
        "stream": False,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(**payload)
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq API call attempt {attempt + 1} failed for description snippet '{description[:50]}...': {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached for this item.")
                raise

def parse_json_response(response_text):
    # First, strip any leading/trailing whitespace (includes newlines)
    cleaned_response_text = response_text.strip()

    # Check for and remove markdown code block fences if present
    if cleaned_response_text.startswith("```json"):
        cleaned_response_text = cleaned_response_text[7:] # Remove ```json
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3] # Remove ```
        cleaned_response_text = cleaned_response_text.strip() # Strip again after removing fences
    elif cleaned_response_text.startswith("```"): # More generic ``` ``` removal
        cleaned_response_text = cleaned_response_text[3:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        cleaned_response_text = cleaned_response_text.strip()


    try:
        # Now, attempt to parse the cleaned string
        data = json.loads(cleaned_response_text)

        skills = data.get("Skills", [])
        domain = data.get("Domain", "")
        raw_languages = data.get("SpokenLanguages", [])
        
        # Ensure extracted values are of expected types
        if not isinstance(skills, list): skills = [str(skills)] if skills else []
        else: skills = [str(s).strip() for s in skills if str(s).strip()]
            
        if not isinstance(domain, str): domain = str(domain).strip() if domain else ""
        else: domain = domain.strip()

        if not isinstance(raw_languages, list): languages = [str(raw_languages)] if raw_languages else []
        else: languages = [str(lang).strip() for lang in raw_languages if str(lang).strip()]

        return skills, domain, languages
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"Cleaned response text that failed to parse: >>>{cleaned_response_text}<<<")
        print(f"Original raw response: >>>{response_text}<<<")
        return [], "", []  # Return defaults
    except Exception as e: # Catch any other unexpected errors during parsing
        print(f"An unexpected error occurred during parsing LLM response: {e}")
        print(f"Original raw response: >>>{response_text}<<<")
        return [], "", []  # Return defaults

def main():
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in .env file or environment variables.")
        print("Please create a .env file with GROQ_API_KEY='your_key' or set the environment variable.")
        return

    try:
        client = Groq(api_key=GROQ_API_KEY)
        print(f"Groq client initialized successfully with model: {MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return

    try:
        with open(ASSESSMENT_FILE, "r", encoding="utf-8") as f:
            assessments = json.load(f)
        print(f"Successfully loaded {len(assessments)} assessments from {ASSESSMENT_FILE}")
    except FileNotFoundError:
        print(f"Error: Assessment file not found at {ASSESSMENT_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {ASSESSMENT_FILE}. Please ensure it's valid JSON.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading assessments: {e}")
        return


    updated_assessments_count = 0
    failed_assessments_count = 0

    for i, assessment in enumerate(tqdm(assessments, desc="Updating assessments")):
        description = assessment.get("description", "").strip()
        assessment_name = assessment.get('name', f'Unnamed Assessment {i+1}')
        
        # Clear previous errors if any before reprocessing
        if "processing_error" in assessment:
            del assessment["processing_error"]

        if not description:
            tqdm.write(f"Skipping {assessment_name} due to empty description.")
            continue

        try:
            response_text = call_groq_llm_with_retry(client, description)
            if response_text:
                skills, domain, languages = parse_json_response(response_text)

                assessment["skills"] = skills
                assessment["domain"] = domain
                assessment["languages"] = languages # Overwrites existing languages field as per previous logic
                updated_assessments_count += 1

                if i < 3 or (len(skills) == 0 and domain == "" and len(languages) == 0): # Print first few or if extraction seems empty
                    tqdm.write(f"\n--- Processed: {assessment_name} ---")
                    # tqdm.write(f"Description (snippet): {description[:100]}...")
                    tqdm.write(f"Extracted Skills: {skills}")
                    tqdm.write(f"Extracted Domain: {domain}")
                    tqdm.write(f"Extracted Spoken Languages: {languages}")
                    tqdm.write("-" * 50)
            else:
                tqdm.write(f"No response from LLM for {assessment_name}.")
                assessment["processing_error"] = "No response from LLM"
                failed_assessments_count +=1

        except Exception as e:
            tqdm.write(f"\n--- Error processing: {assessment_name} ---")
            tqdm.write(f"Error details: {e}")
            assessment["skills"] = assessment.get("skills", [])
            assessment["domain"] = assessment.get("domain", "")
            assessment["languages"] = assessment.get("languages", [])
            assessment["processing_error"] = str(e)
            failed_assessments_count +=1
            tqdm.write("-" * 50)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
        print(f"\nProcessing complete.")
        print(f"Successfully updated (or attempted to update) fields for {updated_assessments_count} assessments.")
        print(f"Failed to fully process {failed_assessments_count} assessments (check 'processing_error' field).")
        print(f"Updated assessments saved to {OUTPUT_FILE}")
    except IOError:
        print(f"Error: Could not write to output file {OUTPUT_FILE}")
    except Exception as e:
        print(f"An unexpected error occurred while saving assessments: {e}")

if __name__ == "__main__":
    main()