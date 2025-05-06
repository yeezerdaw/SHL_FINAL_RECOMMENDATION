# streamlit_app.py
import streamlit as st
import sys
from pathlib import Path
import os
import requests # For calling your own API if you choose to do so in the future

# --- Path Setup ---
_IMPORT_ERROR_DETAILS = None
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from api.models.recommender import SHLRecommender
except ImportError as e:
    _IMPORT_ERROR_DETAILS = e

# --- Page Configuration ---
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

if _IMPORT_ERROR_DETAILS:
    st.error(f"Critical Error: Could not import SHLRecommender. App cannot start. Details: {_IMPORT_ERROR_DETAILS}")
    st.stop()

# --- Initialize Recommender (cached for performance) ---
@st.cache_resource
def load_recommender_cached():
    try:
        recommender_instance = SHLRecommender()
        if not recommender_instance.assessments:
            return None, "Recommender initialized, but failed to load assessments. Check data file and paths."
        return recommender_instance, None
    except Exception as e:
        return None, f"Failed to initialize SHLRecommender: {e}"

recommender, recommender_load_error = load_recommender_cached()

if recommender_load_error:
    st.error(recommender_load_error)
if recommender is None:
    st.warning("Recommender system is not available. Please check logs or contact support.")
    st.stop()

# --- App title ---
st.title("🚀 SHL Assessment Recommender")
st.markdown("Find the perfect SHL assessment for your hiring needs.")
st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("⚙️ Refine Your Search")

try:
    all_job_levels = [""] + recommender.get_all_job_levels()
except Exception: all_job_levels = [""] # Fallback

try:
    all_test_types_options = recommender.get_all_test_types()
except Exception: all_test_types_options = [] # Fallback


max_time_filter = st.sidebar.slider(
    "Maximum Test Duration (minutes)", min_value=5, max_value=180, value=60, step=5
)
job_level_filter = st.sidebar.selectbox(
    "Target Job Level (optional)", options=all_job_levels, index=0,
    help="Select the primary job level you are hiring for."
)
test_types_filter_ui = st.sidebar.multiselect( # Renamed to avoid conflict
    "Preferred Test Types (optional)", options=all_test_types_options,
    help="Select one or more types of tests you are interested in."
)

# --- Main Search Area ---
query_input = st.text_input(
    "📝 Describe your ideal candidate or assessment needs:",
    placeholder="e.g., 'Java developer with problem-solving skills for a remote team'",
    help="Be as descriptive as possible. You can mention skills, job titles, scenarios, etc."
)
top_k_results = st.slider(
    "Number of recommendations to display:", min_value=1, max_value=10, value=5, step=1
)

if st.button("🔍 Find Assessments", type="primary", use_container_width=True):
    if not query_input or not query_input.strip():
        st.warning("Please enter a search query to describe your needs.")
    else:
        augmented_query_parts = [query_input]
        augmented_query_parts.append(f"The assessment should take no more than {max_time_filter} minutes.")
        if job_level_filter and job_level_filter != "":
            augmented_query_parts.append(f"It is for a '{job_level_filter}' job level.")
        if test_types_filter_ui:
            augmented_query_parts.append(f"I am particularly interested in these test types: {', '.join(test_types_filter_ui)}.")
        
        final_query = " ".join(augmented_query_parts)
        st.info(f"🧠 Augmented query for recommender: \"{final_query}\"")

        with st.spinner("⏳ Finding assessments..."):
            try:
                # Streamlit directly calls the recommender logic
                recommendations_data = recommender.recommend(
                    query=final_query,
                    top_k=top_k_results
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                recommendations_data = []
        
        if not recommendations_data:
            st.error("😔 No assessments found. Try a broader search or adjust filters.")
        else:
            st.success(f"🎉 Found {len(recommendations_data)} assessment(s):")
            
            for idx, assessment_dict in enumerate(recommendations_data): # assessment_dict is from recommender.py
                st.markdown("---")
                container = st.container(border=True)

                with container:
                    cols_header = st.columns([0.8, 0.2])
                    with cols_header[0]:
                        container.subheader(f"{idx+1}. {assessment_dict.get('name', 'N/A')}")
                    
                    with cols_header[1]:
                        length_minutes = assessment_dict.get('length_minutes')
                        duration_display = f"⏱️ N/A"
                        if isinstance(length_minutes, (int, float)):
                            duration_display = f"⏱️ **~{length_minutes:.0f} min**"
                        container.markdown(duration_display)
                    
                    if assessment_dict.get('description'):
                        container.markdown(f"**Description**: {assessment_dict['description'][:250]}{'...' if len(assessment_dict['description']) > 250 else ''}")
                    
                    details_cols = st.columns(2)
                    with details_cols[0]:
                        # Domain
                        domain_display = assessment_dict.get('domain', "N/A")
                        if isinstance(domain_display, list): domain_display = ", ".join(map(str,domain_display))
                        st.markdown(f"**Domain(s)**: {domain_display}")

                        # Job Levels (from assessment data)
                        job_levels_data = assessment_dict.get('job_levels', "N/A")
                        if isinstance(job_levels_data, list): job_levels_data = ", ".join(map(str, job_levels_data)) if job_levels_data else "N/A"
                        st.markdown(f"**Job Levels**: {job_levels_data}")
                        
                        # Adaptive Support (from assessment data)
                        adaptive_bool = assessment_dict.get('adaptive', False)
                        st.markdown(f"**Adaptive Support**: {'Yes' if adaptive_bool else 'No'}")


                    with details_cols[1]:
                        # Skills
                        skills_data = assessment_dict.get('skills')
                        skills_display_str = "N/A"
                        if isinstance(skills_data, list) and skills_data:
                            skills_display_str = f"{', '.join(map(str, skills_data[:4]))}{'...' if len(skills_data) > 4 else ''}"
                        elif isinstance(skills_data, str) and skills_data:
                            skills_display_str = skills_data
                        st.markdown(f"**Key Skills**: {skills_display_str}")

                        # Test Types (from assessment data)
                        test_types_data = assessment_dict.get('test_types', "N/A")
                        if isinstance(test_types_data, list): test_types_data = ", ".join(map(str, test_types_data)) if test_types_data else "N/A"
                        st.markdown(f"**Test Types**: {test_types_data}")

                        # Remote Support (from assessment data)
                        remote_bool = assessment_dict.get('remote', False)
                        st.markdown(f"**Remote Support**: {'Yes' if remote_bool else 'No'}")

                    url = assessment_dict.get('url')
                    if url and isinstance(url, str) and url.strip() and url != '#':
                        container.link_button("🔗 View Assessment Details", url, use_container_width=True)
                    else:
                        container.markdown("🔗 Assessment URL not available.")

# --- Footer/Info ---
st.sidebar.markdown("---")
st.sidebar.info(
    "ℹ️ This tool uses AI to recommend SHL assessments. "
    "The more descriptive your query, the better the results. "
    "Filters help refine the search."
)