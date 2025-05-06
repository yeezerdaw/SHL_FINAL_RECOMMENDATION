# SHL Assessment Recommender System

This project is an intelligent recommendation system designed to help hiring managers find relevant SHL assessments based on natural language queries or job descriptions. It leverages a hybrid approach combining traditional keyword search with modern LLM capabilities for criteria extraction and re-ranking.

## Project Overview

Hiring managers often spend significant time searching for the right assessments. This system aims to simplify that process by:
1.  Accepting natural language input describing hiring needs.
2.  Understanding the core requirements (domain, skills, duration, experience, cultural context).
3.  Recommending a ranked list of the most relevant SHL assessments.

**Deployed Applications:**
* **Interactive Demo (Streamlit UI):** [Link to Your Deployed Streamlit App e.g., `https://shlfinalrecommendation-9n4zwkcwk2dbz9ocdld3an.streamlit.app/`
* **API Endpoint (FastAPI):** [Link to Your Deployed FastAPI App Base URL e.g., `https://shl-final-recommendation-api-yesh.onrender.com`
    * Health Check: `https://shl-final-recommendation-api-yesh.onrender.com/health`
    * Recommend: `https://shl-final-recommendation-api-yesh.onrender.com/recommend` (POST request)
    * API Docs (Swagger): `https://shl-final-recommendation-api-yesh.onrender.com/docs`

## Features
* Natural language query processing.
* Hybrid recommendation engine:
    * BM25 for initial candidate retrieval.
    * LLM (Groq Llama3-8B) for structured criteria extraction (domain, skills, duration, experience, cultural context).
    * Heuristic boosting based on extracted criteria and pre-defined technical skill mappings.
    * LLM (Groq Llama3-70B) for sophisticated re-ranking of candidates.
* FastAPI backend for robust API endpoints.
* Streamlit web interface for interactive demos.
* Automated evaluation pipeline using a test set.

## Tech Stack & Libraries
* **Python 3.10+**
* **Backend API:** FastAPI, Uvicorn
* **Frontend UI:** Streamlit
* **LLM Interaction:** Groq Python SDK (for Llama3 models)
* **Search & Ranking:** `rank_bm25`
* **NLP (Tokenization):** Hugging Face `transformers` (BertTokenizer)
* **Data Handling:** `pandas` (if used in scripts), `json`
* **Environment Management:** `python-dotenv`
* **Core Utilities:** `pathlib`, `logging`, `argparse`, `re`

## Project Structure

```
SHL-ASSESSMENT-RECOMMENDATION/
├── api/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       └── recommender.py # Core recommendation logic
├── data/
│   ├── assessment_embeddings_groq_v2_BERT.json # Main assessment data
│   └── test_set.json # Evaluation test set
├── scripts/
│   ├── evaluate.py # Script for evaluating the recommender
│   ├── generate_embeddings.py # Script for data preprocessing (if used)
│   └── update_assessments.py # Script for data preprocessing (if used)
├── main.py # FastAPI application
├── streamlit_ui.py # Streamlit application
├── requirements.txt # Python package dependencies
├── .env.example # Example environment variables
├── .gitignore
└── README.md # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yeezerdaw/SHL_FINAL_RECOMMENDATION.git
    cd SHL_FINAL_RECOMMENDATION
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate   # On Linux/macOS
    # .venv\Scripts\activate      # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    * Copy `.env.example` to a new file named `.env`:
        ```bash
        cp .env.example .env
        ```
    * Open the `.env` file and add your Groq API key:
        ```
        GROQ_API_KEY="your_actual_groq_api_key_here"
        ```

## Running the Applications

### 1. FastAPI API Server

From the project root directory:
```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

* Health Check: `http://127.0.0.1:8000/health`
* API Docs (Swagger UI): `http://127.0.0.1:8000/docs`

### 2. Streamlit Web UI

In a new terminal, from the project root directory:
```bash
streamlit run streamlit_ui.py
```

The Streamlit app will typically open in your browser at `http://localhost:8501`.

## Evaluation

The recommender system's performance is evaluated using a predefined test set (`data/test_set.json`) and standard information retrieval metrics.

**Metrics Used:**

* Mean Recall@3 (MR@3)
* Mean Average Precision@3 (MAP@3)
* Accuracy@5 (at least one correct recommendation in the top 5)

To run the evaluation:
From the project root directory:
```bash
python scripts/evaluate.py
```

**Final Evaluation Results (as of May 7, 2025):**

```
================================================================================
=========================== FINAL EVALUATION RESULTS ===========================
================================================================================
Total Queries:             7
Accuracy @5:               85.7% (6/7)
MAP@3:                      0.161
Recall@3:                   0.316
```

**Evaluation Strategy & "Tracing":**
Our evaluation involved both quantitative metrics (MR@3, MAP@3) generated by `evaluate.py` and qualitative "tracing." Since dedicated LLM observability tools were not integrated for this task, tracing involved:

* **Manual Review of LLM I/O:** Systematically examining the inputs (queries, candidate assessments) and outputs (extracted criteria, ranked lists) for both the criteria extraction LLM (Llama3-8B) and the re-ranking LLM (Llama3-70B).
* **Iterative Prompt Engineering:** Refining prompts based on observed LLM behavior to improve accuracy in criteria extraction (e.g., minimizing skill misattribution for non-technical roles, enhancing cultural context sensitivity) and relevance in re-ranking.
* **Intermediate Pipeline Output Analysis:** Logging and reviewing outputs at each stage (BM25 candidates, post-filtering/boosting candidates) to understand component contributions and identify areas for heuristic refinement (e.g., the `_is_technical_role` logic).

This iterative process of testing, observing, and refining was crucial for improving the system's performance and understanding its behavior.

## Approach and Design

The SHL Assessment Recommender employs a multi-stage hybrid approach:

1.  **Data Preparation (Offline):**
    * **Initial Data Source & Enrichment:** The primary assessment data, including names and URLs, was sourced from [https://www.shl.com/solutions/products/product-catalog/, e.g., "the SHL product catalog website" or "a provided list/CSV"]. To enrich this data with detailed descriptions, job levels, assessment lengths, supported languages, and test types, a custom web scraping script (`scripts/scrape_shl_details.py` - *adjust script name if different*) was developed. This script utilizes `requests` and `BeautifulSoup4` to parse individual assessment pages from the SHL catalog, respecting a polite delay between requests.
    * **LLM-based Feature Extraction:** The scraped descriptions were then processed by an LLM (Groq Llama3-70B via `scripts/update_assessments.py`) to further extract and structure key skills, primary business domains, and spoken languages, ensuring consistency and relevance.
    * **Embedding Generation:** For semantic understanding, relevant textual fields (name, description, extracted skills, domain) from each assessment were combined and converted into dense vector embeddings using a pre-trained BERT model (`bert-base-uncased` via `scripts/generate_embeddings.py`).
    * **Final Dataset:** The enriched and embedded assessment data is stored in `data/assessment_embeddings_groq_v2_BERT.json`, which serves as the primary knowledge base for the recommender.
    * **Technical Skill Mapping:** The `_enhance_technical_skills` method within the recommender pre-computes a mapping between specific technical skills/keywords (e.g., "Selenium," "Java," "SEO") and assessments that cover them, allowing for targeted boosting during the recommendation process.

2.  **Query Understanding (Online):**
    * **Cultural Context Detection:** The input query is first analyzed for explicit or implicit cultural or geographical contexts using a keyword-based approach (`_detect_cultural_focus`).
    * **LLM-Powered Criteria Extraction:** A fast LLM (Groq Llama3-8B) parses the natural language query to extract structured criteria:
        * Primary Job Domain
        * Key Skills (3-5)
        * Maximum Duration (if specified)
        * Experience Level (if specified)
        The detected cultural context is also fed into this LLM to potentially influence its extraction.

3.  **Candidate Retrieval & Initial Ranking (Online):**
    * **BM25 Search:** The original query, augmented with the extracted domain and skills, is tokenized and used to perform a BM25 search over the entire assessment corpus. This retrieves an initial broad set of keyword-relevant candidates.
    * **Heuristic Boosting:** Scores from BM25 are then boosted based on:
        * Match with the LLM-extracted primary domain.
        * Overlap with LLM-extracted key skills.
        * If the query is identified as a "technical role" (`_is_technical_role`), assessments matching pre-defined technical skill mappings (`tech_skill_map` and `skill_to_assessments`) receive an additional boost.

4.  **Candidate Filtering & Preparation for Re-ranking (Online):**
    * **Hard Filtering:** The boosted candidate list is filtered based on strict constraints, primarily the maximum duration extracted by the LLM.
    * **Soft Preference Sorting & Pooling:** Candidates that pass hard filters are then sorted based on their proximity to the desired duration (if specified). A pool of the top candidates (e.g., up to 50) is selected for the final re-ranking stage.

5.  **LLM-Powered Re-ranking (Online):**
    * A more powerful LLM (Groq Llama3-70B) re-ranks the selected candidate pool.
    * The LLM is provided with:
        * The original user query.
        * The structured criteria extracted in Step 2 (including domain, skills, duration, experience, and cultural context).
        * A summarized list of the candidate assessments (including their name, domain, key skills, duration, adaptive/remote flags, and a description snippet).
    * The LLM is prompted to holistically evaluate each candidate against all requirements and return a ranked list of the top N assessment IDs.

6.  **Result Presentation:**
    * The final ranked list is returned via the API or displayed in the Streamlit UI.

## Known Limitations & Future Work

* **Cultural Context Nuance:** While basic cultural context detection is implemented, its influence on the LLM extraction and re-ranking could be further enhanced for more subtle geographical or cultural role requirements (e.g., Query 3 for "COO in China" was a notable challenge).
* **LLM Misinterpretations:** As observed (e.g., Query 6 skill misattribution), criteria extraction LLMs can occasionally misinterpret queries. Continuous prompt engineering, more examples in prompts, or fine-tuning could mitigate this.
* **Ground Truth Subjectivity:** The evaluation scores are relative to the provided `test_set.json`. The system sometimes recommends valid alternatives not present in this specific ground truth.
* **Data Freshness:** The system relies on the `assessment_embeddings_groq_v2_BERT.json` data. A pipeline for regularly updating this data from the SHL catalog would be needed for a production system.
* **Advanced Filtering in UI:** The Streamlit UI could incorporate more direct post-search filtering based on attributes like `adaptive_support` or specific `test_types` from the recommended set.
* **URL Fetching:** Directly processing job description URLs as input (fetching and parsing their content) could be added.

## Author
*   YeezerDaw ([https://github.com/yeezerdaw](https://github.com/yeezerdaw))
```
