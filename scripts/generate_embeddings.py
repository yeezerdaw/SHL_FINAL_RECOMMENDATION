import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch

# --- Configuration ---
# Determine project root (assuming this script is in a subdirectory like 'scripts' or 'notebooks')
# If this script is at the root, project_root = Path(__file__).resolve().parent
project_root = Path(__file__).resolve().parent.parent # Adjust if script is not two levels down from root
sys.path.append(str(project_root)) # If you have utility modules in the project root

# Define file names and data subdirectory
DATA_SUBDIRECTORY = "data"
INPUT_FILE_NAME = "assessment_structure_filled_groq_v2.json"
OUTPUT_FILE_NAME = "assessment_embeddings_groq_v2_BERT.json"

# Construct full paths
# Ensure the DATA_SUBDIRECTORY exists at project_root level
data_dir_path = project_root / DATA_SUBDIRECTORY
INPUT_FILE = data_dir_path / INPUT_FILE_NAME
OUTPUT_FILE = data_dir_path / OUTPUT_FILE_NAME

# --- BERT Model and Tokenizer Setup ---
# Determine device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model once
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device) # Move model to the determined device
    model.eval() # Set model to evaluation mode
except Exception as e:
    print(f"Error loading BERT model or tokenizer: {e}")
    sys.exit(1) # Exit if model can't be loaded

def clean_assessment_fields(assessment: dict) -> dict:
    """
    Cleans specified fields in the assessment dictionary.
    Ensures 'skills', 'languages', and 'domain' are lists of unique, cleaned strings.
    Removes leading/trailing whitespace and asterisks from items.
    """
    cleaned_assessment = assessment.copy() # Work on a copy

    def _clean_field_to_list(field_value):
        if isinstance(field_value, str):
            # If it's a single string, make it a list of one item
            items_to_clean = [field_value]
        elif isinstance(field_value, list):
            items_to_clean = field_value
        else:
            # If it's neither string nor list (e.g., None, number), return empty list
            return []

        cleaned_items = set() # Use a set for uniqueness
        for item in items_to_clean:
            if isinstance(item, str) and item.strip():
                # Clean: strip whitespace -> remove leading asterisks -> strip again
                clean_item = item.strip().lstrip('*').strip()
                if clean_item: # Add only if not empty after cleaning
                    cleaned_items.add(clean_item)
        return sorted(list(cleaned_items)) # Return as a sorted list

    cleaned_assessment["skills"] = _clean_field_to_list(assessment.get("skills", []))
    cleaned_assessment["languages"] = _clean_field_to_list(assessment.get("languages", []))
    cleaned_assessment["domain"] = _clean_field_to_list(assessment.get("domain", "")) # Handles single string or list

    return cleaned_assessment

def build_embedding_text(assessment: dict) -> str:
    """
    Constructs a single text string from various assessment fields for embedding.
    """
    # Use .get with empty string/list defaults for safety
    name = assessment.get("name", "")
    description = assessment.get("description", "")
    
    # Ensure domain, skills, languages are lists for join, even if empty after cleaning
    domain_list = assessment.get("domain", [])
    skills_list = assessment.get("skills", [])
    languages_list = assessment.get("languages", [])

    parts = [
        name,
        description,
        f"Domain: {', '.join(domain_list)}" if domain_list else "",
        f"Skills: {', '.join(skills_list)}" if skills_list else "",
        f"Languages: {', '.join(languages_list)}" if languages_list else "",
    ]
    # Join non-empty parts with a space
    return " ".join(filter(None, parts)).strip()

def get_bert_embeddings(text: str) -> list | None:
    """
    Generates BERT embeddings for the given text using the CLS token.
    """
    if not text:
        # print("Warning: Empty text received for embedding.") # Optional warning
        return None
    try:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(device) # Move inputs to the same device as the model

        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model(**inputs)
        
        # Use the embedding of the [CLS] token as the sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        # Move to CPU before converting to numpy/list if it was on GPU
        return cls_embedding.cpu().numpy().tolist()
    except Exception as e:
        # Log the text that caused the error for easier debugging (first 100 chars)
        print(f"Error embedding text (snippet: '{text[:100]}...'): {e}")
        return None

def main():
    """
    Main function to load assessments, clean data, generate embeddings, and save results.
    """
    # Ensure data directory exists
    if not data_dir_path.exists():
        print(f"Data directory not found: {data_dir_path}")
        print("Please ensure the 'data' subdirectory exists at the project root and contains the input file.")
        sys.exit(1)
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            assessments_data = json.load(f)
        print(f"Successfully loaded {len(assessments_data)} assessments from {INPUT_FILE}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {INPUT_FILE}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading assessments: {e}")
        sys.exit(1)
    
    processed_assessments = []
    embedding_errors = 0

    for assessment_orig in tqdm(assessments_data, desc="Generating embeddings", unit="assessment"):
        # 1. Clean assessment fields
        assessment = clean_assessment_fields(assessment_orig)
        
        # 2. Build text for embedding
        text_to_embed = build_embedding_text(assessment)
        
        # 3. Generate embedding
        if text_to_embed: # Only proceed if there's text to embed
            embedding = get_bert_embeddings(text_to_embed)
            if embedding:
                assessment["embedding_text_used"] = text_to_embed # Store the text used for embedding
                assessment["embedding"] = embedding
            else:
                embedding_errors += 1
                assessment["embedding_error"] = "Failed to generate embedding"
        else:
            # Handle cases where no text could be built (e.g., all fields were empty)
            assessment["embedding_error"] = "No content to build embedding text"
            embedding_errors +=1
            
        processed_assessments.append(assessment)

    # Create output directory if it doesn't exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(processed_assessments, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Successfully processed {len(processed_assessments)} assessments.")
        if embedding_errors > 0:
            print(f"⚠️ Encountered errors for {embedding_errors} embeddings (see 'embedding_error' field in output).")
        print(f"Results saved to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing output to {OUTPUT_FILE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving results: {e}")

if __name__ == "__main__":
    main()