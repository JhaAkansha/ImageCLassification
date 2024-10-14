# Installation
1. Create a folder called image_classification_project
2. Call 'git init' in that folder
3. Open the folder in VS Code
4. Create a new virtual environment using the command "python -m venv .venv" 
5. Activate the virtual environment using the command ".venv\Scripts\activate.bat" (Linux: "source .venv\bin\activate.bat")
6. Install the dependencies using "pip install -r requirements.txt"
7. Run the live server: uvicorn main:app --reload (uvicorn is asynchronous)
