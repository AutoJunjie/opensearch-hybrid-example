# Biomedical Search System with OpenSearch and Sentence Transformers

This project implements a sophisticated biomedical search system using OpenSearch and sentence transformers. It provides hybrid, vector, and keyword search capabilities for medical articles.

The system ingests medical article data, generates embeddings for abstracts, and indexes this information in OpenSearch. It then offers three types of search functionalities: hybrid search (combining keyword and semantic search), vector search (based on semantic embeddings), and keyword search (based on text matching).

## Repository Structure

- `aos-hybrid1.py`: The main Python script containing the `BiomedicalSearch` class and search system implementation.
- `README.md`: This file, providing an overview and usage instructions for the project.

## Usage Instructions

### Installation

1. Ensure you have Python 3.11+ installed.
2. Clone this repository:
   ```
   git clone https://github.com/AutoJunjie/opensearch-hybrid-example.git
   cd opensearch-hybrid-example
   ```
3. Install the required dependencies:
   ```
   pip install pandas opensearch-py sentence-transformers python-dotenv
   ```

### Configuration

1. Create a `.env` file in the project root with the following OpenSearch credentials:
   ```
   OPENSEARCH_URL=<your-opensearch-url>
   OPENSEARCH_USERNAME=<your-username>
   OPENSEARCH_PASSWORD=<your-password>
   ```

### Getting Started

1. Prepare your data:
   - Ensure you have a CSV file named `sample.csv` with columns: `id`, `keyword1`, `keyword2`, and `abstract`.

2. Run the main script:
   ```
   python aos-hybrid1.py
   ```

This will set up the search pipeline, create the index, ingest the data, and perform example searches.

### Common Use Cases

1. Hybrid Search:
   ```python
   search_system = BiomedicalSearch()
   results = search_system.hybrid_search("重组人黄体生成素", k=5)
   ```

2. Vector Search:
   ```python
   results = search_system.vector_search("胚胎移植", k=5)
   ```

3. Keyword Search:
   ```python
   results = search_system.keyword_search("重组人黄体生成素", k=5)
   ```

### Troubleshooting

1. OpenSearch Connection Issues:
   - Ensure your OpenSearch credentials in the `.env` file are correct.
   - Check if the OpenSearch cluster is accessible from your network.

2. Embedding Generation Errors:
   - Verify that you have sufficient RAM for loading the sentence transformer model.
   - Try using a smaller model if you encounter memory issues.

3. Data Ingestion Problems:
   - Confirm that your `sample.csv` file is in the correct format and location.
   - Check for any data inconsistencies or missing values in the CSV file.

### Debugging

To enable verbose logging:

1. Add the following lines at the beginning of `aos-hybrid1.py`:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Run the script again to see detailed debug output.

## Data Flow

The biomedical search system processes data and handles search requests as follows:

1. Data Ingestion:
   - Reads medical article data from a CSV file.
   - Generates embeddings for article abstracts using a sentence transformer model.
   - Indexes the data (ID, keywords, abstract, and embeddings) in OpenSearch.

2. Search Process:
   - Receives a search query from the user.
   - For hybrid and vector searches, generates an embedding for the query.
   - Constructs and sends a search request to OpenSearch based on the search type.
   - Retrieves and processes the search results.
   - Returns formatted results to the user.

```
[User Query] -> [Query Embedding] -> [OpenSearch]
                                     |
[CSV Data] -> [Data Processing] -----+
               |
               v
[Sentence Transformer] -> [Abstract Embeddings]
```

Note: The system uses a search pipeline for hybrid searches to combine and normalize results from both keyword and semantic searches.