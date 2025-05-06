import json
import sys
from pathlib import Path
# from urllib.parse import urlparse # Not used in this version
import logging

# --- Path Setup ---
# Assume this script (evaluate.py) is in a directory (e.g., 'scripts' or 'evaluation')
# and the 'api' and 'data' directories are siblings to its parent directory.
# Project Root
#   |- api
#   |  |- models
#   |     |- recommender.py
#   |- data
#   |  |- test_set.json
#   |  |- assessment_embeddings_groq_v2_BERT.json (used by SHLRecommender)
#   |- scripts (or where evaluate.py is)
#      |- evaluate.py

try:
    # Path to the directory containing this script
    current_script_dir = Path(__file__).resolve().parent
    # Path to the project root (assuming script is in a subdirectory like 'scripts')
    project_root = current_script_dir.parent
except NameError:
    # Fallback if __file__ is not defined (e.g., in some interactive environments)
    project_root = Path(".").resolve() # Assumes script is run from project root

sys.path.insert(0, str(project_root)) # Prepend project root to sys.path

# Now import the SHLRecommender
try:
    from api.models.recommender import SHLRecommender
except ImportError as e:
    print(f"Error importing SHLRecommender: {e}")
    print(f"Ensure 'recommender.py' is in 'api/models/' relative to project root: {project_root}")
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- File Paths ---
TEST_SET_FILENAME = 'test_set.json'
DATA_DIR_NAME = 'data'
test_set_path = project_root / DATA_DIR_NAME / TEST_SET_FILENAME

# --- Normalization Function ---
def normalize_name(name: str) -> str:
    """
    Normalizes an assessment name for comparison.
    - Converts to lowercase.
    - Removes " | SHL" suffix if present.
    - Removes all spaces.
    - Strips leading/trailing whitespace.
    """
    if not isinstance(name, str):
        return ""
    # Remove " | SHL" suffix or similar patterns if they exist
    name_cleaned = name.split("|")[0].strip()
    return name_cleaned.lower().replace(" ", "").strip()

# --- Metrics Calculation Functions ---
def calculate_precision_at_k(predicted_normalized_names: list[str], relevant_normalized_names: set[str], k: int) -> float:
    """Calculates Precision@k."""
    if k == 0:
        return 0.0
    top_k_predicted = predicted_normalized_names[:k]
    num_relevant_in_top_k = sum(1 for item in top_k_predicted if item in relevant_normalized_names)
    return num_relevant_in_top_k / k

def calculate_average_precision_at_k(predicted_normalized_names: list[str], relevant_normalized_names: set[str], k: int) -> float:
    """Calculates Average Precision@k (AP@k)."""
    if not relevant_normalized_names or k == 0:
        return 0.0
    
    num_relevant_hits = 0
    sum_of_precisions = 0.0
    
    for i, p_name in enumerate(predicted_normalized_names[:k]):
        if p_name in relevant_normalized_names:
            num_relevant_hits += 1
            precision_at_i = num_relevant_hits / (i + 1)
            sum_of_precisions += precision_at_i
            
    if num_relevant_hits == 0:
        return 0.0
    # Standard AP definition divides by min(k, |Relevant Items|) or just |Relevant Items|
    # Here, we'll use the number of relevant items found up to k, or total relevant items if evaluating full recall.
    # For MAP, it's typically divided by the total number of relevant items for the query.
    # However, if we are strictly evaluating AP@k, it can be num_relevant_hits.
    # Let's stick to the common definition of AP which normalizes by total relevant items.
    return sum_of_precisions / len(relevant_normalized_names)


def calculate_recall_at_k(predicted_normalized_names: list[str], relevant_normalized_names: set[str], k: int) -> float:
    """Calculates Recall@k."""
    if not relevant_normalized_names or k == 0:
        return 0.0
    top_k_predicted = predicted_normalized_names[:k]
    num_relevant_in_top_k = sum(1 for item in top_k_predicted if item in relevant_normalized_names)
    return num_relevant_in_top_k / len(relevant_normalized_names)

# --- Main Evaluation Logic ---
# --- Main Evaluation Logic ---
def main():
    # Print header with version/date info
    print("\n" + "="*80)
    print("SHL Recommender System Evaluation".center(80))
    print(f"{Path(__file__).name} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80 + "\n")

    logger.info(f"Attempting to load test set from: {test_set_path}")
    if not test_set_path.exists():
        logger.error(f"Test set file not found: {test_set_path}")
        sys.exit(1)

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    logger.info(f"Loaded {len(test_set)} queries from test set.\n")

    logger.info("Initializing SHLRecommender...")
    recommender = SHLRecommender()
    if not recommender.assessments:
        logger.error("SHLRecommender failed to load assessments. Evaluation cannot proceed.")
        sys.exit(1)
    logger.info("SHLRecommender initialized successfully.\n")

    total_queries = len(test_set)
    if total_queries == 0:
        logger.warning("Test set is empty. No evaluation to perform.")
        return

    # Metrics configuration
    k_for_accuracy = 5
    k_for_map_recall = 3
    
    # Initialize metrics
    overall_accuracy_correct_count = 0
    cumulative_ap_at_3 = 0.0
    cumulative_recall_at_3 = 0.0

    # Print evaluation configuration
    print("-"*80)
    print(" EVALUATION CONFIGURATION ".center(80, "-"))
    print(f"• Total test queries: {total_queries}")
    print(f"• Accuracy measured @ top {k_for_accuracy} recommendations")
    print(f"• MAP and Recall measured @ top {k_for_map_recall} recommendations")
    print("-"*80 + "\n")

    for i, entry in enumerate(test_set, 1):
        query = entry.get("query")
        expected_assessments_raw = entry.get("assessments", [])

        if not query or not isinstance(query, str) or not query.strip():
            logger.warning(f"Skipping test entry {i} due to missing or invalid query.")
            continue
        if not expected_assessments_raw:
            logger.warning(f"Skipping test entry {i} (Query: {query[:50]}...) due to no expected assessments.")
            continue

        # Print query header
        print("\n" + "="*80)
        print(f" QUERY {i}/{total_queries} ".center(80, "="))
        print(f"Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        print("-"*80)

        # Get recommendations
        system_recommendations_raw = recommender.recommend(query, top_k=k_for_accuracy)

        # Normalize names for comparison
        expected_normalized_names = {normalize_name(e.get("name")) for e in expected_assessments_raw if e.get("name")}
        system_normalized_names_list = [normalize_name(r.get("name")) for r in system_recommendations_raw if r.get("name")]

        if not expected_normalized_names:
            logger.warning("No valid expected assessment names after normalization. Skipping...")
            continue

        # Print expected vs recommended
        print("\nEXPECTED ASSESSMENTS:")
        for idx, name in enumerate(expected_normalized_names, 1):
            print(f"  {idx}. {name}")
            
        print("\nSYSTEM RECOMMENDATIONS:")
        for idx, name in enumerate(system_normalized_names_list[:k_for_accuracy], 1):
            match_indicator = "✓" if name in expected_normalized_names else "✗"
            print(f"  {idx}. {name} {match_indicator}")

        # Calculate metrics
        found_match = any(rec_name in expected_normalized_names for rec_name in system_normalized_names_list[:k_for_accuracy])
        ap_at_k = calculate_average_precision_at_k(system_normalized_names_list, expected_normalized_names, k_for_map_recall)
        recall_at_k = calculate_recall_at_k(system_normalized_names_list, expected_normalized_names, k_for_map_recall)

        # Update cumulative metrics
        if found_match:
            overall_accuracy_correct_count += 1
        cumulative_ap_at_3 += ap_at_k
        cumulative_recall_at_3 += recall_at_k

        # Print query-level metrics
        print("\nQUERY METRICS:")
        print(f"• Accuracy @{k_for_accuracy}: {'HIT' if found_match else 'MISS'}")
        print(f"• AP@{k_for_map_recall}: {ap_at_k:.3f}")
        print(f"• Recall@{k_for_map_recall}: {recall_at_k:.3f}")
        print("="*80)

    # Calculate final metrics
    overall_accuracy = overall_accuracy_correct_count / total_queries if total_queries > 0 else 0.0
    mean_average_precision_at_3 = cumulative_ap_at_3 / total_queries if total_queries > 0 else 0.0
    mean_recall_at_3 = cumulative_recall_at_3 / total_queries if total_queries > 0 else 0.0

    # Print final summary with visual emphasis
    print("\n" + "="*80)
    print(" FINAL EVALUATION RESULTS ".center(80, "="))
    print("="*80)
    print(f"{'Total Queries:':<25} {total_queries}")
    print(f"{'Accuracy @5:':<25} {overall_accuracy * 100:.1f}% ({overall_accuracy_correct_count}/{total_queries})")
    print(f"{'MAP@3:':<25} {mean_average_precision_at_3:.3f}")
    print(f"{'Recall@3:':<25} {mean_recall_at_3:.3f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    from datetime import datetime  # Added for timestamp
    main()