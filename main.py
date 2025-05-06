# main.py (Located at Project Root)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Path Setup (REMOVED) ---
# No longer needed if main.py is at the project root and uvicorn is run from there.
# Python's default module search path will handle finding the 'api' package.
# import sys
# from pathlib import Path
# PROJECT_ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(PROJECT_ROOT)) # REMOVE THIS

_SHLRecommender_CLASS = None
_INITIALIZATION_ERROR_MESSAGE = None

try:
    # Direct import assuming 'api' is a package (directory with __init__.py)
    # in the current working directory (which should be the project root when running uvicorn)
    from api.models.recommender import SHLRecommender
    _SHLRecommender_CLASS = SHLRecommender
except ImportError as e:
    _INITIALIZATION_ERROR_MESSAGE = f"CRITICAL: Could not import SHLRecommender. Ensure 'api/models/recommender.py' exists and project structure is correct. Error: {e}"
    print(_INITIALIZATION_ERROR_MESSAGE)
except Exception as e: # Catch other potential errors during import phase
    _INITIALIZATION_ERROR_MESSAGE = f"CRITICAL: An unexpected error occurred during SHLRecommender import. Error: {e}"
    print(_INITIALIZATION_ERROR_MESSAGE)


app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API to get SHL assessment recommendations based on a query.",
    version="1.0.2" # Incremented version
)

# --- Initialize recommender instance ---
recommender_instance = None
if _SHLRecommender_CLASS:
    try:
        # SHLRecommender's __init__ will call load_dotenv()
        # and use its own path logic to find data files
        recommender_instance = _SHLRecommender_CLASS()
        if not recommender_instance.assessments:
            # This message might be redundant if recommender.py already logs/errors heavily
            _INITIALIZATION_ERROR_MESSAGE = (_INITIALIZATION_ERROR_MESSAGE or "") + \
                                            " WARNING: SHLRecommender initialized, but failed to load assessments."
            print(_INITIALIZATION_ERROR_MESSAGE)
            recommender_instance = None # Treat as failed initialization
    except Exception as e:
        _INITIALIZATION_ERROR_MESSAGE = (_INITIALIZATION_ERROR_MESSAGE or "") + \
                                            f" CRITICAL: Failed to initialize SHLRecommender instance: {e}"
        print(_INITIALIZATION_ERROR_MESSAGE)
        recommender_instance = None
elif not _INITIALIZATION_ERROR_MESSAGE:
    _INITIALIZATION_ERROR_MESSAGE = "CRITICAL: SHLRecommender class could not be loaded. API will not function correctly."
    print(_INITIALIZATION_ERROR_MESSAGE)


# --- Pydantic Models for API Specification ---

class RecommendQueryRequest(BaseModel):
    query: str = Field(..., description="Job description or natural language query for assessment recommendations.")

class RecommendedAssessmentDetail(BaseModel):
    name: str = Field(..., description="Name of the SHL assessment.")
    url: Optional[str] = Field(None, description="Valid URL to the assessment resource.")
    adaptive_support: str = Field(..., description="Either 'Yes' or 'No' indicating if the assessment supports adaptive testing.")
    description: str = Field(..., description="Detailed description of the assessment.")
    duration: Optional[int] = Field(None, description="Duration of the assessment in minutes.")
    remote_support: str = Field(..., description="Either 'Yes' or 'No' indicating if the assessment can be taken remotely.")
    test_type: List[str] = Field(default_factory=list, description="Categories or types of the assessment.")

class RecommendationListResponse(BaseModel):
    recommended_assessments: List[RecommendedAssessmentDetail]


# --- API Endpoints ---

@app.get("/health", summary="API Health Check", response_model=dict)
async def health_check():
    """
    Provides a simple status check. Returns `{"status": "healthy"}` if the API
    is operational and the recommender is initialized with assessments.
    Otherwise, raises a 503 Service Unavailable error.
    """
    if recommender_instance and recommender_instance.assessments:
        return {"status": "healthy"}
    else:
        reason = _INITIALIZATION_ERROR_MESSAGE or "Recommender not initialized or no assessments loaded."
        # Log the reason for internal tracking
        app.logger.error(f"Health check failed: {reason}") # Requires FastAPI app logger setup if not default
        print(f"Health check failed: {reason}") # Fallback print
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "reason": reason }
        )

@app.post("/recommend", response_model=RecommendationListResponse, summary="Get Assessment Recommendations")
async def get_recommendations_api(request: RecommendQueryRequest):
    """
    Accepts a job description or natural language query and returns
    recommended relevant assessments (At most 10, minimum 1 if matches found)
    based on the input.
    """
    if not recommender_instance:
        # This check ensures that if recommender failed to init, we provide a clear error.
        reason = _INITIALIZATION_ERROR_MESSAGE or "Recommender service is not available due to an initialization error."
        app.logger.error(f"/recommend called but recommender not available: {reason}")
        print(f"/recommend called but recommender not available: {reason}")
        raise HTTPException(status_code=503, detail=reason)

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    internal_top_k = 10 # Fetch up to 10 candidates as per "At most 10"

    try:
        raw_recommendations = recommender_instance.recommend(
            query=request.query,
            top_k=internal_top_k
        )
    except Exception as e:
        # Log the full exception for debugging
        app.logger.error(f"Error during recommendation generation: {e}", exc_info=True)
        print(f"Error during recommendation generation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating recommendations.")

    if not raw_recommendations:
        return RecommendationListResponse(recommended_assessments=[])

    output_assessments: List[RecommendedAssessmentDetail] = []
    for r_dict in raw_recommendations:
        if not isinstance(r_dict, dict):
            print(f"Skipping non-dictionary item in raw_recommendations: {r_dict}")
            continue

        assessment_name = r_dict.get('name', "Unknown Assessment")
        desc_val = r_dict.get('description', "No description available.")

        duration_raw = r_dict.get('length_minutes')
        duration_int: Optional[int] = None
        if duration_raw is not None:
            try:
                duration_int = int(float(duration_raw))
            except (ValueError, TypeError):
                print(f"Warning: Could not convert duration '{duration_raw}' to int for assessment '{assessment_name}'.")

        adaptive_bool = r_dict.get('adaptive', False)
        adaptive_str = "Yes" if adaptive_bool else "No"

        remote_bool = r_dict.get('remote', False)
        remote_str = "Yes" if remote_bool else "No"

        test_types_list_raw = r_dict.get('test_types', [])
        if isinstance(test_types_list_raw, list):
            test_types_list = [str(tt) for tt in test_types_list_raw if str(tt).strip()] # Ensure non-empty strings
        elif test_types_list_raw and str(test_types_list_raw).strip(): # If it's a single non-empty value
            test_types_list = [str(test_types_list_raw)]
        else:
            test_types_list = [] # Default to empty list if raw is None or empty string

        try:
            assessment_detail = RecommendedAssessmentDetail(
                name=assessment_name,
                url=r_dict.get('url'), # Will be None if key missing or value is None
                adaptive_support=adaptive_str,
                description=desc_val,
                duration=duration_int, # Will be None if conversion failed or key missing
                remote_support=remote_str,
                test_type=test_types_list
            )
            output_assessments.append(assessment_detail)
        except Exception as pydantic_error: # Catch Pydantic validation errors specifically if possible
            # Or catch broader errors if needed
            print(f"Warning: Could not create RecommendedAssessmentDetail for '{assessment_name}'. Error: {pydantic_error}")
            # Log the problematic data for debugging
            debug_data = {
                "name": assessment_name, "url": r_dict.get('url'), "adaptive_support": adaptive_str,
                "description_len": len(desc_val) if desc_val else 0, "duration": duration_int,
                "remote_support": remote_str, "test_type": test_types_list
            }
            print(f"Data causing Pydantic error: {debug_data}")


        if len(output_assessments) >= 10: # Ensure at most 10 results are processed and returned
            break
    
    # Ensure minimum 1 if matches found, but if all failed Pydantic validation, list might be empty
    if raw_recommendations and not output_assessments:
        # This means items were recommended but none could be formatted to the API spec
        # This indicates an internal server issue with data consistency or mapping logic
        app.logger.error("Recommendations were generated but failed to map to the response model.")
        print("Recommendations were generated but failed to map to the response model.")
        raise HTTPException(status_code=500, detail="Failed to format recommendations. Data inconsistency or internal mapping issue suspected.")

    return RecommendationListResponse(recommended_assessments=output_assessments)


@app.get("/", summary="API Root Information", include_in_schema=False)
async def read_root():
    # Determine overall status based on recommender_instance and any init errors
    status_message = "healthy"
    init_msg = "Recommender initialized successfully."
    if not recommender_instance or not recommender_instance.assessments:
        status_message = "degraded"
        init_msg = _INITIALIZATION_ERROR_MESSAGE or "Recommender not fully initialized or no assessments loaded."

    return {
        "message": "Welcome to the SHL Assessment Recommender API.",
        "status": status_message,
        "initialization_details": init_msg,
        "health_check_url": "/health",
        "recommendations_url": "/recommend",
        "documentation_url": "/docs" # Or /redoc
    }

# How to run this file (assuming it's named main.py and at the project root):
# 1. Ensure your virtual environment is activated.
# 2. Ensure all dependencies from requirements.txt are installed.
# 3. Ensure .env file with GROQ_API_KEY is at the project root.
# 4. Run from the project root directory:
#    uvicorn main:app --reload
#
# Your API will be available at http://127.0.0.1:8000
# OpenAPI docs at http://127.0.0.1:8000/docs