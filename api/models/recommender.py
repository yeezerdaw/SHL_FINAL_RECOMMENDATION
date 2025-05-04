import json
import re
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
# Load environment variables (.env file should contain GEMINI_API_KEY=your_key)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Ensure API key is available
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please create a .env file or set the environment variable.")

# Constants
PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../data/assessments_structured.json"
)
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
GEMINI_MODEL_NAME = 'gemini-1.5-flash'
SIMILARITY_WEIGHT = 0.6  # Weight for semantic similarity score
NAME_MATCH_WEIGHT = 0.4  # Weight for name matching score

print(f"Resolved PATH: {PATH}")
class SHLRecommender:
    """
    Recommends SHL assessments based on natural language queries using filtering,
    semantic search, and hybrid scoring.
    """
    def __init__(self, data_path: str = PATH):
        """
        Initialize the recommender, load data, configure models, and pre-compute embeddings.

        Args:
            data_path (str): Path to the structured JSON data file.
        """
        print("Initializing SHLRecommender...")
        print(f"Provided data path: {data_path}")
        print(f"Resolved absolute path: {os.path.abspath(data_path)}")
        print(f"File exists: {os.path.exists(data_path)}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the file exists.")

        self.df = self._load_data(data_path)
        self.gemini = self._init_gemini()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # --- Pre-compute Embeddings ---
        print("Pre-computing embeddings for all assessments...")
        # Create a combined text field for richer embeddings
        self.df['CombinedText'] = self.df['Name'].fillna('') + ' | ' + self.df['Description'].fillna('')
        self.all_doc_embeddings = self.embedding_model.encode(
            self.df['CombinedText'].tolist(),
            show_progress_bar=True
        )
        print(f"✅ Embeddings computed for {len(self.df)} assessments.")
        print("✅ System ready. Gemini and embeddings loaded.")

    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Load assessment data from JSON, normalize, and preprocess.

        Args:
            path (str): Path to the JSON data file.

        Returns:
            pd.DataFrame: Processed DataFrame with assessment data.
        """
        print(f"Loading data from {path}...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {path}: {e}")
        except Exception as e:
            raise IOError(f"Could not read file {path}: {e}")

        df = pd.json_normalize(data)

        # Standardize column names (adjust if your JSON structure differs)
        column_mapping = {
            'name': 'Name',
            'description': 'Description',
            'length_minutes': 'Duration',
            'test_types': 'TestTypes',
            'job_levels': 'JobLevels',
            'languages': 'Languages',
            'url': 'URL',
            'remote': 'Remote',
            'adaptive': 'Adaptive'
        }
        # Only rename columns that exist in the DataFrame
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # --- Preprocessing ---
        # Ensure essential columns exist
        for col in ['Name', 'Description', 'TestTypes']:
             if col not in df.columns:
                 raise ValueError(f"Missing essential column '{col}' in the data.")

        # Fill missing text fields
        df['Name'] = df['Name'].fillna('Unknown Assessment')
        df['Description'] = df['Description'].fillna('')

        # Ensure list-like columns are actually lists and handle NaNs/nulls
        for col in ['TestTypes', 'JobLevels', 'Languages']:
            if col in df.columns:
                # Convert non-list entries (like NaN) to empty lists
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            else:
                 # If column is missing, add it with empty lists
                 df[col] = [[] for _ in range(len(df))]


        # Handle Duration: Convert to numeric, use -1 for missing/invalid
        if 'Duration' in df.columns:
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(-1).astype(int)
        else:
            df['Duration'] = -1 # Add duration column if missing

        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def _init_gemini(self):
        """Configure and initialize the Gemini API client."""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            # Try a simple generation to check connectivity (optional)
            # model.generate_content("test")
            print("Gemini initialized successfully.")
            return model
        except Exception as e:
            print(f"⚠️ Failed to initialize Gemini: {e}")
            print("⚠️ Query parsing will rely solely on regex fallback.")
            return None

    def _parse_query_gemini(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Parse the user query using Gemini LLM to extract structured information.

        Args:
            query (str): The natural language user query.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with parsed fields or None if failed.
        """
        if not self.gemini:
            return None # Gemini failed to initialize

        prompt = f"""
        Analyze the user query about SHL assessments and extract the following information into a valid JSON object.

        Query: "{query}"

        Extract JSON with EXACTLY these fields:
        {{
            "assessment_topic": str (The primary subject/topic of the desired assessment, e.g., "leadership", "python coding", "sales ability", "cognitive ability", "teamwork". Be specific. Return null if unclear or too generic.),
            "max_duration": int (Maximum duration in minutes, e.g., query "under 45 mins" -> 45, "less than 1 hour" -> 59. Null if no upper limit.),
            "min_duration": int (Minimum duration in minutes, e.g., query "over 30 mins" -> 31, "at least 1 hour" -> 60. Null if no lower limit.),
            "exact_duration": int (Exact duration in minutes, e.g., query "30 minute test" -> 30. Null if not exact.),
            "technical_skills": list[str] (List of specific technical skills mentioned, lowercase, e.g., ["java", "sql"]. Empty list if none.),
            "behavioral_traits": list[str] (List of specific behavioral traits mentioned, lowercase, e.g., ["collaboration", "adaptability"]. Empty list if none.),
            "job_role_hint": str (Any mentioned job role or level context, lowercase, e.g., "manager", "entry level", "developer". Null if none.),
            "required_languages": list[str] (List of required languages for the test, lowercase. Empty list if none.),
            "cleaned_query": str (The original query, possibly slightly cleaned or standardized.)
        }}

        Rules for Extraction:
        1.  Duration: Prioritize `exact_duration`. If a range is given (e.g., "under", "over", "less/more than"), use `max_duration` or `min_duration`. Convert hours to minutes.
        2.  Topic: Identify the core assessment subject. If multiple are plausible, pick the most prominent.
        3.  Skills/Traits: Extract specific skills/traits mentioned.
        4.  Cleaned Query: Return the original query for embedding purposes.
        5.  Return ONLY the valid JSON object, without any surrounding text or markdown formatting (like ```json).

        JSON Output:
        """
        try:
            print(f"Sending query to Gemini: '{query}'")
            response = self.gemini.generate_content(prompt)
            # Clean potential markdown backticks and surrounding whitespace
            raw_json = re.sub(r'^```json\s*|\s*```$', '', response.text.strip())
            parsed_data = json.loads(raw_json)

            # Basic validation of structure
            expected_keys = {"assessment_topic", "max_duration", "min_duration", "exact_duration",
                             "technical_skills", "behavioral_traits", "job_role_hint",
                             "required_languages", "cleaned_query"}
            if not expected_keys.issubset(parsed_data.keys()):
                 print(f"⚠️ Gemini output missing expected keys. Output: {raw_json}")
                 return None

            print(f"Gemini parsed result: {parsed_data}")
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"⚠️ Gemini returned invalid JSON: {e}. Raw response: '{response.text}'")
            return None
        except Exception as e:
            # Catch potential API errors, rate limits, etc.
            print(f"⚠️ Gemini API call failed: {e}")
            return None

    def _parse_query_regex_fallback(self, query: str) -> Dict[str, Any]:
        """Fallback query parser using simple regex (limited capability)."""
        print("Using regex fallback parser.")
        parsed = {
            "assessment_topic": None, "max_duration": None, "min_duration": None,
            "exact_duration": None, "technical_skills": [], "behavioral_traits": [],
            "job_role_hint": None, "required_languages": [], "cleaned_query": query
        }

        # Duration parsing (simple examples)
        if match := re.search(r'under\s+(\d+)\s*min', query, re.IGNORECASE):
            parsed['max_duration'] = int(match.group(1))
        elif match := re.search(r'less than\s+(\d+)\s*min', query, re.IGNORECASE):
            parsed['max_duration'] = int(match.group(1)) -1
        elif match := re.search(r'over\s+(\d+)\s*min', query, re.IGNORECASE):
            parsed['min_duration'] = int(match.group(1)) + 1
        elif match := re.search(r'(\d+)\s*min', query, re.IGNORECASE):
             parsed['exact_duration'] = int(match.group(1))
        # Add hour parsing if needed

        # Simple keyword extraction for topic/skills/traits (very basic)
        if 'leadership' in query.lower(): parsed['assessment_topic'] = 'leadership'; parsed['behavioral_traits'].append('leadership')
        if 'python' in query.lower(): parsed['assessment_topic'] = 'python coding'; parsed['technical_skills'].append('python')
        if 'java' in query.lower(): parsed['assessment_topic'] = 'java coding'; parsed['technical_skills'].append('java')
        if 'teamwork' in query.lower(): parsed['assessment_topic'] = 'teamwork'; parsed['behavioral_traits'].append('teamwork')
        # ... add more basic rules as needed ...

        print(f"Regex fallback result: {parsed}")
        return parsed

    def _filter_assessments(self, parsed_query: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter assessments based on parsed query criteria.

        Args:
            parsed_query (Dict[str, Any]): The structured query from the parser.

        Returns:
            pd.DataFrame: A DataFrame containing only the assessments that match the filters.
        """
        df_filtered = self.df.copy()
        print(f"Starting filtering with {len(df_filtered)} assessments.")

        # --- Duration Filter ---
        exact_duration = parsed_query.get('exact_duration')
        max_duration = parsed_query.get('max_duration')
        min_duration = parsed_query.get('min_duration')

        if exact_duration is not None:
            df_filtered = df_filtered[df_filtered['Duration'] == exact_duration]
            print(f"Applied exact duration filter ({exact_duration} mins). Remaining: {len(df_filtered)}")
        else:
            if max_duration is not None:
                 # Apply max duration, but keep assessments with unknown duration (-1) unless explicitly excluded
                 # df_filtered = df_filtered[(df_filtered['Duration'] <= max_duration) & (df_filtered['Duration'] != -1)] # Exclude unknown
                 df_filtered = df_filtered[df_filtered['Duration'] <= max_duration] # Include unknown (-1 matches <= max_duration)
                 print(f"Applied max duration filter (<= {max_duration} mins). Remaining: {len(df_filtered)}")
            if min_duration is not None:
                 df_filtered = df_filtered[df_filtered['Duration'] >= min_duration]
                 print(f"Applied min duration filter (>= {min_duration} mins). Remaining: {len(df_filtered)}")

        # --- Test Type Filter based on Skills/Traits ---
        # Define mappings (customize these)
        tech_test_types = {'K', 'T', 'S'} # Knowledge, Technical, Simulation
        behav_test_types = {'B', 'P', 'S'} # Behavioral, Personality, Simulation

        required_types = set()
        if parsed_query.get('technical_skills'):
            required_types.update(tech_test_types)
        if parsed_query.get('behavioral_traits'):
            required_types.update(behav_test_types)
        # If query mentions topic often linked to cognitive (e.g., "cognitive", "aptitude", "reasoning")
        if any(t in parsed_query.get('assessment_topic', '').lower() for t in ['cognitive', 'aptitude', 'reasoning']):
             required_types.add('A') # Aptitude

        if required_types:
             # Filter assessments that have AT LEAST ONE of the required test types
             def check_types(assessment_types):
                 return not required_types.isdisjoint(assessment_types)

             df_filtered = df_filtered[df_filtered['TestTypes'].apply(check_types)]
             print(f"Applied test type filter ({required_types}). Remaining: {len(df_filtered)}")


        # --- Language Filter ---
        required_languages = parsed_query.get('required_languages')
        if required_languages:
             req_langs_set = set(lang.lower() for lang in required_languages)
             def check_langs(available_langs):
                  # Assumes languages in df are lowercase strings in a list
                  available_set = set(lang.lower() for lang in available_langs)
                  return req_langs_set.issubset(available_set)

             df_filtered = df_filtered[df_filtered['Languages'].apply(check_langs)]
             print(f"Applied language filter ({required_languages}). Remaining: {len(df_filtered)}")

        # --- Job Role Hint Filter (Optional - simple substring match) ---
        job_hint = parsed_query.get('job_role_hint')
        if job_hint:
             def check_levels(levels):
                  return any(job_hint in level.lower() for level in levels)
             df_filtered = df_filtered[df_filtered['JobLevels'].apply(check_levels)]
             print(f"Applied job level hint filter ('{job_hint}'). Remaining: {len(df_filtered)}")

        print(f"Filtering complete. {len(df_filtered)} assessments remaining.")
        return df_filtered


    def recommend(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Main recommendation method. Parses query, filters assessments, ranks by hybrid score.

        Args:
            query (str): The natural language user query.
            top_k (int): The number of top recommendations to return.

        Returns:
            List[Dict]: A list of recommended assessment dictionaries.
        """
        # 1. Parse Query
        parsed_query = self._parse_query_gemini(query)
        if parsed_query is None:
            # Fallback to regex if Gemini failed
            parsed_query = self._parse_query_regex_fallback(query)

        # 2. Filter Assessments
        filtered_df = self._filter_assessments(parsed_query)

        if filtered_df.empty:
            print("No assessments found matching the filter criteria.")
            return []

        # 3. Rank Filtered Assessments
        print(f"Ranking {len(filtered_df)} filtered assessments...")

        # Get embeddings for the filtered assessments using their indices
        filtered_indices = filtered_df.index
        filtered_embeddings = self.all_doc_embeddings[filtered_indices]

        # Encode the user query (use cleaned_query from parser)
        query_embed = self.embedding_model.encode(parsed_query['cleaned_query'])

        # Calculate semantic similarity
        if filtered_embeddings.shape[0] > 0:
            similarities = util.cos_sim(query_embed, filtered_embeddings)[0].numpy()
            # Add similarity score to the filtered DataFrame
            # Use .loc to avoid SettingWithCopyWarning
            filtered_df.loc[:, 'similarity'] = similarities
        else:
             filtered_df.loc[:, 'similarity'] = np.nan


        # --- Calculate Hybrid Score ---
        assessment_topic = parsed_query.get('assessment_topic')
        if assessment_topic:
            # Simple name match boost: 1 if topic word(s) in name, 0 otherwise
            # Use regex word boundaries for more precise matching
            topic_pattern = r'\b' + re.escape(assessment_topic) + r'\b'
            filtered_df.loc[:, 'name_match_score'] = filtered_df['Name'].str.contains(topic_pattern, case=False, na=False, regex=True).astype(float)
            print(f"Calculated name match score for topic: '{assessment_topic}'")
        else:
            filtered_df.loc[:, 'name_match_score'] = 0.0 # No topic, no boost

        # Combine scores
        filtered_df.loc[:, 'final_score'] = (SIMILARITY_WEIGHT * filtered_df['similarity'].fillna(0)) + \
                                            (NAME_MATCH_WEIGHT * filtered_df['name_match_score'])

        # Sort by the final combined score
        ranked_df = filtered_df.sort_values('final_score', ascending=False)

        # 4. Format Top-K Results
        top_results = ranked_df.head(top_k).to_dict('records')

        # Clean up output fields
        output_list = []
        for item in top_results:
            duration_display = int(item['Duration']) if item['Duration'] != -1 else "N/A"
            output_list.append({
                'Name': item['Name'],
                'Description': item.get('Description', ''), # Use .get for safety
                'Duration': duration_display,
                'TestTypes': item.get('TestTypes', []), # Use .get for safety
                'URL': item.get('URL', 'N/A'), # Use .get for safety
                'similarity': round(float(item.get('similarity', 0.0)), 4),
                'name_match': round(float(item.get('name_match_score', 0.0)), 4),
                'final_score': round(float(item.get('final_score', 0.0)), 4)
            })

        print(f"Returning top {len(output_list)} recommendations.")
        return output_list

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # Create recommender instance (loads data, models, computes embeddings)
        recommender = SHLRecommender(data_path="../data/assessments_structured.json") # Adjust path if needed

        def print_recommendations(query: str):
            """Helper function to query the recommender and print results."""
            print("-" * 50)
            print(f"Recommendations for: '{query}'")
            print("-" * 50)
            try:
                results = recommender.recommend(query, top_k=3)

                if not results:
                    print("\nNo matching assessments found.")
                    return

                print("\nTop Recommendations:")
                for i, item in enumerate(results, 1):
                    print(f"{i}. {item['Name']}")
                    print(f"   Duration: {item['Duration']} mins" if item['Duration'] != "N/A" else "   Duration: N/A")
                    print(f"   Test Types: {' | '.join(item['TestTypes'])}") # Display types nicely
                    print(f"   Relevance Score: {item['final_score']:.3f} (Sim: {item['similarity']:.3f}, Name Match: {item['name_match']:.1f})")
                    print(f"   URL: {item['URL']}\n")

            except Exception as e:
                print(f"\nAn error occurred during recommendation for '{query}': {e}")
                import traceback
                traceback.print_exc() # Print detailed traceback for debugging

        # --- Example Queries ---
        print_recommendations("Leadership assessment under 45 mins")
        print_recommendations("Leadership assessment")
        print_recommendations("python coding test")
        print_recommendations("customer service assessment around 30 minutes")
        print_recommendations("cognitive ability test for managers")
        print_recommendations("sales assessment in French") # Example requiring language

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file exists at the specified path.")
    except ValueError as e:
         print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
