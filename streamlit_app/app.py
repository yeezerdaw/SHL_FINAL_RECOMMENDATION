import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("SHL Assessment Recommender")

query = st.text_input("Enter your query:")
top_k = st.number_input("Number of recommendations:", min_value=1, max_value=10, value=3)

if st.button("Get Recommendations"):
    if query.strip():
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(
                    f"{API_URL}/recommendations",
                    json={"query": query, "top_k": top_k}
                )
                if response.status_code == 200:
                    recommendations = response.json()
                    st.success("Recommendations fetched successfully!")
                    for rec in recommendations:
                        st.subheader(rec["Name"])
                        st.write(f"**Description:** {rec['Description']}")
                        st.write(f"**Duration:** {rec['Duration']}")
                        st.write(f"**Test Types:** {', '.join(rec['TestTypes'])}")
                        st.write(f"**URL:** [Link]({rec['URL']})")
                        st.write(f"**Similarity Score:** {rec['similarity']:.2f}")
                        st.write(f"**Final Score:** {rec['final_score']:.2f}")
                        st.write("---")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")