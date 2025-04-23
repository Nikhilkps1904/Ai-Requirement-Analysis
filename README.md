Requirements Conflict Detection Tool
Overview
The Requirements Conflict Detection Tool is a Python application that automates the identification of conflicts between system or software requirements using the Gemini API (gemini-1.5-flash). Designed for requirements engineers, project managers, and developers, it processes requirement sets from CSV files, analyzes pairs for conflicts (e.g., "high performance" vs. "low cost"), and outputs results in CSV, XLSX, or console formats. The tool supports both local execution and cloud environments like Google AI Studio, with robust features like multithreading, API rate limit handling, and responsive interrupt handling.
This tool streamlines requirements analysis, a critical step in system design, by leveraging AI to detect conflicts that could lead to costly design errors. It’s ideal for software engineering, automotive systems, aerospace, or any domain requiring precise requirement validation.
Features

AI-Powered Conflict Detection: Uses Gemini API to analyze requirement pairs and identify conflicts with descriptive types (e.g., "Performance Conflict") and reasons.
Scalable Processing: Handles small (≤15 requirements) or large (>15) datasets with baseline or exhaustive pairwise analysis.
Multithreading: Employs two worker threads for efficient processing, respecting API rate limits (15 requests/minute, 1,500 requests/day).
Cross-Platform Compatibility: Runs locally with Tkinter GUI or in Google AI Studio with console output.
Robust Error Handling: Manages API errors (e.g., 429 rate limit), connection issues, and user interrupts (Ctrl+C) with partial result saving.
Caching: Uses lru_cache to avoid redundant API calls for identical prompts.
Detailed Logging: Tracks requests, errors, and progress in conflict_detection.log and request_counts.log.
Flexible Outputs: Saves results to CSV/XLSX files (local) or console (cloud), with UTF-8 support for diverse inputs.

Prerequisites

Python: Version 3.8 or higher.
Gemini API Key: Obtain a free API key from Google AI Studio.
Dependencies: Listed in requirements.txt.
Operating System: Windows, macOS, Linux, or Google AI Studio (cloud).

Installation

Clone the Repository:
git clone https://github.com/Nikhilkps1904/Ai-Requirement-Analysis.git
cd Ai-Requirement-Analysis


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

The requirements.txt should include:
pandas
requests
python-dotenv
openpyxl


Configure the API Key:

Create a .env file in the project root:
GEMINI_API_KEY=your_api_key_here


Replace your_api_key_here with your Gemini API key from Google AI Studio.



Verify Setup: Ensure Python is installed and the .env file is correctly configured:
python -c "import pandas, requests, dotenv, openpyxl"



Usage
Local Execution

Prepare Input CSV: Create a CSV file (e.g., requirements.csv) with a "Requirements" column:
Requirements
"Fuel efficiency > 50 km/l"
"Top speed > 120 km/h"
"Low maintenance cost"
"Advanced safety features"


Run the Script:
python cust2.py


Interact with the Menu:

Option 1: Analyze Requirements: Select a CSV file to analyze all requirement pairs.
Option 2: Enter a New Requirement: Input a single requirement and compare it against a CSV’s requirements.
Option 3: Exit: Close the program.


View Outputs:

Results: Saved in Results/CSV/results.csv and Results/XLSX/results.xlsx (or with timestamps for new requirements).
Logs: Check conflict_detection.log for general logs and request_counts.log for API usage (e.g., "Total API requests: 45/1500").
Console: Displays progress and errors in real-time.



Google AI Studio Execution

Copy the Script: Paste cust2.py into the Google AI Studio code editor.

Set the API Key: Configure GEMINI_API_KEY directly in the script or via environment settings if supported.

Hardcode Input (Optional): If CSV upload isn’t supported, modify the script to use a list:
all_original_requirements = ["Fuel efficiency > 50 km/l", "Top speed > 120 km/h", "Low maintenance cost"]


Run and Monitor:

Use the console menu (Tkinter is unavailable).
Results and logs appear in the console.
Interrupt with Ctrl+C to save partial results.



Example Output
For the CSV above (4 requirements, 6 pairs):

results.csv:
Requirement_1,Requirement_2,Conflict_Type,Conflict_Reason
"Fuel efficiency > 50 km/l","Top speed > 120 km/h","Performance Conflict","High speed increases fuel consumption"
"Low maintenance cost","Advanced safety features","Cost Conflict","Safety features increase maintenance costs"


Console/Log:
2025-04-23 10:15:32 - INFO - Analyzing 6 pairs
2025-04-23 10:15:40 - INFO - Total API requests: 6/1500
2025-04-23 10:15:40 - INFO - Analysis complete. Results saved to Results/CSV/results.csv



Rate Limits
The tool respects Gemini API’s free-tier limits:

15 Requests per Minute (RPM): Enforced with 4-second delays per worker.
1,500 Requests per Day (RPD): Stops processing if reached, with warnings at 90% (1,350 requests).
1,000,000 Tokens per Minute (TPM): Not tracked (prompts are typically short).

For 52 requirements (1,326 pairs), processing takes ~88 minutes at 15 RPM. Larger datasets may require multiple days or a paid API tier.
Limitations

API Rate Limits: 1,500 RPD restricts large datasets (e.g., >52 requirements).
File I/O in Cloud: CSV/XLSX outputs are unavailable in Google AI Studio; results are printed to console.
Interrupt Delay: Ctrl+C exits in ~1–2 seconds due to API timeouts and thread cleanup.
Token Limit: 1,000,000 TPM not monitored, which may cause throttling for long prompts.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please include tests and update documentation. For major changes, open an issue first to discuss.
License
This project is licensed under the MIT License.
Acknowledgments

Google AI Studio for providing the Gemini API.
Open-source libraries: pandas, requests, openpyxl, and python-dotenv.

Contact
For questions or feedback, open an issue on GitHub or contact nikhilkps7480@gmail.com.

Happy analyzing your requirements!
