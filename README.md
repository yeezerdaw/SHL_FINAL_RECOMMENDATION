# SHL Assessment Recommender

This project implements a recommendation system for SHL assessments using FastAPI for the backend API and Streamlit for the frontend user interface. The system allows users to input queries and receive tailored assessment recommendations based on their needs.

## Project Structure

```
shl-assessment-recommender
├── api
│   ├── main.py                # Entry point for the FastAPI application
│   ├── models
│   │   └── recommender.py     # Contains the SHLRecommender class for assessment logic
│   └── requirements.txt       # Dependencies for the FastAPI application
├── streamlit_app
│   ├── app.py                 # Main entry point for the Streamlit web application
│   ├── components
│   │   └── __init__.py        # Additional components for the Streamlit app
│   └── requirements.txt       # Dependencies for the Streamlit application
├── data
│   └── assessments_structured.json # Structured data for SHL assessments
├── .env                       # Environment variables and configuration settings
├── README.md                  # Project documentation
└── .gitignore                 # Files and directories to ignore in version control
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd shl-assessment-recommender
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install API Dependencies**
   Navigate to the `api` directory and install the required packages:
   ```bash
   cd api
   pip install -r requirements.txt
   ```

4. **Install Streamlit Dependencies**
   Navigate to the `streamlit_app` directory and install the required packages:
   ```bash
   cd ../streamlit_app
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables**
   Create a `.env` file in the root directory and add your configuration settings, such as API keys.

## Usage

### Running the API

To start the FastAPI application, navigate to the `api` directory and run:
```bash
uvicorn main:app --reload
```
The API will be accessible at `http://127.0.0.1:8000`.

### Running the Streamlit Application

To start the Streamlit application, navigate to the `streamlit_app` directory and run:
```bash
streamlit run app.py
```
The Streamlit app will be accessible at `http://localhost:8501`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.