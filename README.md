# Digital Assistant Attorney

## Overview

The Digital Assistant Attorney project aims to revolutionize legal research and assistance by leveraging state-of-the-art AI and NLP technologies. The project is divided into four key segments, each designed to enhance the efficiency and accuracy of legal research and decision-making.

## Segments

### 1. Semantic Searching and Keyword Searching in Legal Database

This segment focuses on improving the search capabilities within a legal database. It combines traditional keyword searching with advanced semantic search techniques to deliver more relevant and precise results. Users can query the database, which includes various acts and previous judgment orders from High Courts in India, using natural language or specific legal terms.

### 2. Legal-QA using DPR (Dense Passage Retrieval)

Dense Passage Retrieval (DPR) is utilized in this segment to create a sophisticated legal question-answering system. By understanding and retrieving dense representations of passages, this system can answer complex legal queries by drawing from the extensive legal database. This helps in providing quick, accurate, and contextually relevant answers to legal professionals.

### 3. Multi-Document QA using RAG (Retrieval-Augmented Generation)

In this segment, Retrieval-Augmented Generation (RAG) is implemented to handle multi-document question answering. This technique combines retrieval of information from multiple documents and generative models to provide comprehensive answers to legal questions. It ensures that responses are well-rounded and sourced from a broad range of documents within the legal database.

### 4. AutoGPT for Law

The AutoGPT for Law segment focuses on automating legal research and drafting tasks. By utilizing the capabilities of GPT (Generative Pre-trained Transformer), this segment aims to assist in generating legal documents, summarizing cases, and providing insights based on the input data. It streamlines repetitive tasks, allowing legal professionals to focus on more complex and analytical aspects of their work.

## Legal Database

The legal database is a comprehensive repository containing acts and previous judgment orders from various High Courts in India. It serves as the backbone for all segments of the Digital Assistant Attorney project, providing the necessary data to enhance search capabilities, question-answering, and document generation.

## Installation

To get started with the Digital Assistant Attorney project, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/amartya123-sudo/Digital-Assistant-Attorney.git
   cd Digital-Assistant-Attorney
   ```

2. **Create a conda environment**
   ```bash
   conda env create --name environment_name -f environment.yml
   ```

3. **Activate conda environment**
   ```bsh
    conda activate environment_name
    ```

4. **Run the Application**
   ```bash
   python app.py
   ```

## Usage

To use the system for answering legal questions:

1. **Run the steamlit app**:
    ```sh
    streamlit run app.py
    ```

## Acknowledgements

We would like to thank IITI Drishti CPS Foundation for funding this project.

## Contact

For any queries or support, please contact Project Instructor at [tanveer.ahmed@bennett.edu.in](mailto:tanveer.ahmed@bennett.edu.in).

---