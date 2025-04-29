# User-Concern-Prioritazation
User Concern Prioritazation

This dashboard analyzes mobile app reviews to identify, interpret, and prioritize user concerns through an integrated and interpretable pipeline. It leverages topic modeling to detect recurring themes across reviews, sentiment analysis to focus on negative and neutral feedback, and a multi-metric scoring framework to assign priority to issues. The prioritization mechanism combines four key metrics:

- Entropy (measuring topic distribution variability),
- Topic Prevalence (how frequently a topic appears across reviews),
- Thumbs Up count (user endorsement of the review), and
- Sentiment polarity (intensity of negativity).

These metrics work together to produce a balanced and data-driven priority score for each topic. The dashboard also incorporates LLM-based topic interpretation using Mistral AI, helping users understand the nature of each concern and relate it to broader user experience (UX) factors.

A unique feature of the tool is its ability to isolate and prioritize data demand-related concerns, a relatively underexplored but critical aspect in mobile app usage. Developers can choose to analyze general user concerns or specifically filter and rank reviews that reflect data usage complaints or bandwidth-related issues. This dual capability ensures the tool serves both general and specialized diagnostic needs, enabling more targeted and informed decision-making in app development and maintenance.

## Features

- Upload app review datasets (Excel format) with required columns: review text, version, and thumbs up count
- Automatic sentiment analysis (using VADER) to identify and filter negative and neutral reviews
- Topic modeling (LDA with bi-grams) to group similar concerns across reviews
- Priority scoring based on a multi-metric approach using:
- LLM-based topic interpretation using Mistral AI, with mapping to user experience (UX) factors
- Data demand-specific filtering: isolate and prioritize only data demand-related issues if required
- Interactive visualization of topic analysis metrics
- Two-tab results display:
   Bar chart of prioritized user concerns
   Ranked list with interpreted topics and their priority scores
- Export results (CSV) for further analysis or reporting

## Setup

1. Clone this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```
Note: This will install all required packages including openpyxl for Excel file support.

3. Download NLTK data:
```python
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
```
Note: These downloads will be attempted automatically when you run the app, but you can run them manually if needed.

4. Set up environment variables:
   - Copy `.env.template` to `.env`
   - Add your Hugging Face API key to the `.env` file:
```bash
HUGGINGFACE_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Upload your Excel file containing reviews and click "Analyze Reviews"

4. View the analysis results in two tabs:
   - Topic Analysis: Shows the metrics visualization
   - Topic Interpretation: Shows the interpreted results

5. Download the results using the provided buttons

## Input File Format

The Excel file (.xlsx) must contain these required columns:
- content: Text content of the review
- thumbsUpCount: Number of thumbs up/likes for the review (numeric)

The sentiment analysis will be performed automatically on the content, so you don't need to provide sentiment scores.

## Output Files

The dashboard generates two types of output files:
1. Negative & Neutral Reviews: CSV file containing filtered reviews with calculated sentiment
2. Topic Interpretation Results: CSV file containing interpreted topics and their metrics

## Dependencies

See requirements.txt for a complete list of dependencies. Key dependencies include:
- streamlit: For the web interface
- pandas & openpyxl: For Excel file handling
- nltk: For natural language processing and sentiment analysis
- gensim: For topic modeling
- sentence-transformers: For text similarity
- python-dotenv: For environment variable management

## Troubleshooting

If you encounter NLTK resource errors, you can manually download all required resources by running this Python code:
```python
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
``` 
