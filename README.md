# Demographic-Enhanced Recommendation System

Welcome to the Demographic-Enhanced Recommendation System! This system offers personalized product recommendations by leveraging user demographics, specifically age and gender, detected from images. It consists of a Flask API for backend operations, including age and gender detection, as well as product recommendation. Additionally, it features a user-friendly Streamlit application for the frontend interface.

## Setup and Running Instructions

To get started with the system, follow these simple steps:

1. **Setup Environment**: Set up a Python environment on your local machine.

2. **Navigate to Project Folder**: Open your command line interface and change directory to the folder containing your project files.

   ```bash
   cd path/to/your/project/folder
   ```

3. **Install Dependencies**: Run `pip install -r requirements.txt` in the terminal.

4. **Run Flask API**: Execute `python app.py` to start the backend operations.

   ```bash
   python app.py
   ```

5. **Launch Streamlit App**: In another terminal tab or window, navigate to your project folder and run the Streamlit application.

   ```bash
   streamlit run streamlit.py
   ```

6. **Access Application**: Open a web browser and visit `http://localhost:8501` to explore personalized recommendations.

The system is also deployed on Render. You can access it using  https://demographicrecommendation.onrender.com/.