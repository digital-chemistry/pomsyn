# POMSYN

This repository contains an automated pipeline designed to process unstructured chemical procedure text documents. Utilizing Google's Gemini API, the system systematically extracts reagents, structures methodologies, and assembles the data into a standardized, machine-readable format.



## System Overview

The pipeline is designed for robustness and data integrity, operating across four distinct phases:

**Phase 0 (Text Preprocessing):** Cleans, splits, and standardizes raw input text

**Phase 1 (Reagent Extraction):** Identifies and catalogs chemical entities, maintaining a centralized chemical database

**Phase 2 (Procedure Structuring):** Converts raw methodological text into clean, structured Markdown steps

**Phase 3 (Data Assembly):** Merges the extracted entities and structured steps into a final comprehensive JSON schema



## Prerequisites

* **Python:**  3.8 or higher
* **Dependencies:** Requires the official Google GenAI SDK
* **API Key:** A valid Google Gemini API key



## Environment Setup

1. **API Key Setup:** \
    Create a file named `google_api_key.txt` containing only your API key and place it in your intended Working Directory
   
2. **Prepare the Working Directory:** \
   The script operates dynamically based on a user-defined working directory. Before running the script, ensure your directory contains the raw input data.

   **Expected Initial Structure:**
   ```text
   Working_Directory/
   ├── google_api_key.txt       ### Required: Your Gemini API Key
   ├── text_input/              ### Required: Folder containing raw .txt files
   ├── prompts/                 ### Optional: Custom prompt overrides
   └── input_definition.csv     ### Optional: Target mapping for specific files
   ```



## Installation

To install the package and its dependencies, run the following command from the root directory of the project:

```bash
pip install .
```



## Usage

1. Execute from the terminal:
   ```bash
   $ pomsyn
   ```

2. **Interactive Configuration:**
   Upon execution, the script will prompt you for two inputs:
   * **Model Selection:** \
   Press `Enter` to use the default model (`gemini-3.1-flash-lite-preview`), or type the name of a specific model

   * **Working Directory:** \
   Provide the path to the directory containing your `text_input` folder and API key

3. **Execution:**
   The pipeline will output its progress to the console. It will sequentially process each file in the `text_input` folder, appending successful results to the master dataset



## Output & Artifacts

Upon completion (or a safe interruption), the pipeline generates several outputs within your Working Directory:

* **`procedures.json`:** \
The master output file containing the fully assembled, structured data for all successfully processed inputs
* **`chemical_database.json`:** \
A database of all extracted reagents and chemical entities encountered across the inputs
* **`runs/` Directory:** \
A historical archive of pipeline executions



### Interruption Handling
If you need to stop the pipeline, press `Ctrl+C`. The system will safely catch the interrupt, clean up any temporary files related to the currently processing document, and safely package all previously completed files into the `runs/latest/` archive and `procedures.json` output
