🧠 Requirements Conflict Detection Tool.

A Python-powered AI tool for identifying conflicts in system or software requirements using the Gemini 1.5 Flash API. Built for requirements engineers, project managers, and developers, this tool streamlines the requirements analysis process, preventing costly design flaws and ensuring robust, validated specifications across domains such as software engineering, aerospace, and automotive systems.

🚀 Key Features
AI-Powered Conflict Analysis
Leverages the Gemini API to detect logical, performance, or cost-based conflicts between requirement pairs. Returns both conflict type (e.g., Performance Conflict) and reasoning.

Scalable for Any Dataset
Automatically switches between baseline and exhaustive pairwise analysis for datasets with ≤15 or >15 requirements.

Multithreaded for Speed
Uses two worker threads with intelligent API rate-limit management (15 RPM / 1,500 RPD).

Robust Execution
Supports both local environments (with Tkinter GUI) and cloud execution (e.g., Google AI Studio).

Smart Error Handling
Handles API throttling, user interrupts (Ctrl+C), and connection issues gracefully, saving partial results automatically.

Caching for Efficiency
Implements LRU caching to avoid redundant API calls for repeated prompts.

Flexible Outputs
Results are saved in CSV/XLSX formats locally or printed to console in cloud environments. UTF-8 encoding ensures international compatibility.

Detailed Logging
Tracks progress and usage in conflict_detection.log and request_counts.log.

📦 Prerequisites
Python: Version 3.8+

API Key: Free Gemini API key from Google AI Studio

Dependencies:

nginx
Copy
Edit
pandas
requests
python-dotenv
openpyxl
Supported OS: Windows, macOS, Linux, or Google AI Studio (cloud)

⚙️ Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/Nikhilkps1904/Ai-Requirement-Analysis.git
cd Ai-Requirement-Analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
🔑 Configure the API Key
Create a .env file in the project root.

Add the following line:

ini
Copy
Edit
GEMINI_API_KEY=your_api_key_here
Validate setup:

bash
Copy
Edit
python -c "import pandas, requests, dotenv, openpyxl"
🧪 Usage
✅ Local Execution
Prepare Input CSV
A file named requirements.csv with a column titled Requirements:

csv
Copy
Edit
Requirements
"Fuel efficiency > 50 km/l"
"Top speed > 120 km/h"
"Low maintenance cost"
"Advanced safety features"
Run the Tool

bash
Copy
Edit
python cust2.py
Menu Options

1: Analyze all requirement pairs

2: Compare a new requirement against the CSV

3: Exit

View Results

CSV: Results/CSV/results.csv

XLSX: Results/XLSX/results.xlsx

Logs: conflict_detection.log, request_counts.log

☁️ Cloud Execution (Google AI Studio)
Copy-paste cust2.py into the code editor.

Set GEMINI_API_KEY directly in code if .env is not supported.

Use hardcoded list input if file uploads are unavailable:

python
Copy
Edit
all_original_requirements = [
    "Fuel efficiency > 50 km/l",
    "Top speed > 120 km/h",
    "Low maintenance cost"
]
📊 Sample Output
CSV Result

csv
Copy
Edit
Requirement_1,Requirement_2,Conflict_Type,Conflict_Reason
"Fuel efficiency > 50 km/l","Top speed > 120 km/h","Performance Conflict","High speed increases fuel consumption"
"Low maintenance cost","Advanced safety features","Cost Conflict","Safety features increase maintenance costs"
Console/Log

yaml
Copy
Edit
2025-04-23 10:15:32 - INFO - Analyzing 6 pairs
2025-04-23 10:15:40 - INFO - Total API requests: 6/1500
2025-04-23 10:15:40 - INFO - Analysis complete. Results saved.
⚠️ Rate Limits
15 requests/minute

1,500 requests/day

Token usage not tracked (Max: 1,000,000 TPM)

For 52 requirements (1,326 pairs), expect ~88 minutes of processing.

🧱 Limitations
API rate limits restrict throughput for large datasets.

Cloud environments like Google AI Studio do not support file I/O.

Interrupts (Ctrl+C) may be delayed due to API response handling.

Token limits are not explicitly monitored.

🤝 Contributing
We welcome contributions! To contribute:

bash
Copy
Edit
# Fork and clone the repo
git checkout -b feature/your-feature
# Make your changes, add tests
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a pull request
For major changes, please open an issue first.

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgments
Google AI Studio – for the Gemini API.

Open-source libraries: pandas, requests, openpyxl, python-dotenv.

📬 Contact
For feedback, issues, or support, open a GitHub issue or email:
📧 nikhilkps7480@gmail.com

