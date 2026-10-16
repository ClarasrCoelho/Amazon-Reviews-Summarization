<div align="center">

# 📚 Amazon Reviews Summarization

### From raw book reviews to meaningful insights using Big Data & NLP

**An end-to-end pipeline for sentiment analysis, topic discovery, and abstractive summarization of Amazon book reviews.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS-S3-569A31?style=flat-square&logo=amazons3&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)

**Academic project · Big Data Laboratory · M.Sc. in Computer Engineering – Data Engineering · ISEP**

[✨ Highlights](#-highlights) · [🏗️ Architecture](#️-architecture) · [🧠 NLP Pipeline](#-nlp--machine-learning) · [🛠️ Tech Stack](#️-tech-stack) · [🚀 Getting Started](#-getting-started)

</div>

---

## 🎯 Project Overview

This project transforms large volumes of **unstructured Amazon book reviews** into useful, accessible information. It aims to identify recurring themes, analyse readers' sentiment, and generate concise, balanced summaries of the positive and negative feedback for each book.

The solution combines **distributed data processing**, **cloud-based storage**, and **Transformer-based NLP** with an API and web interface. It was developed for the **Big Data Laboratory** course at **ISEP – Polytechnic Institute of Porto**.

## ✨ Highlights

- ☁️ **Cloud Data Lake:** AWS S3 with **Bronze, Silver, and Gold** layers.
- ⚡ **Big Data Processing:** Data ingestion, cleaning, and transformation using **Apache Spark / PySpark**.
- 🧠 **Natural Language Processing:** Sentiment analysis, topic modelling, and text summarization.
- 🤗 **Transformer Models:** **BERT** for sentiment-related tasks and **BART** for abstractive summarization.
- 📝 **Hybrid Summarization:** Combines extractive selection with abstractive generation.
- 🔌 **Application Integration:** **FastAPI** and a web interface for accessing results.
- 🔄 **Batch & Streaming:** Lambda-inspired design accommodating both processing approaches.

## 🏗️ Architecture

```text
                 📚 Amazon Book Reviews
                     JSON / streams
                           │
                           ▼
                     📥 Ingestion
                           │
                           ▼
                 🥉 AWS S3 · BRONZE
                      Raw reviews
                           │
                           ▼
                    ⚡ PySpark ETL
             Cleaning · Deduplication
                           │
                           ▼
                 🥈 AWS S3 · SILVER
                    Prepared reviews
                           │
                           ▼
                     🧠 NLP / ML
            Sentiment · Topics · Summaries
                           │
                           ▼
                  🥇 AWS S3 · GOLD
                   Enriched outputs
                           │
                           ▼
                     🔌 FastAPI
                           │
                           ▼
                   🖥️ Web Interface
```

The pipeline follows the **Medallion Architecture** (Bronze–Silver–Gold):

| Layer | Purpose |
| :--- | :--- |
| 🥉 **Bronze** | Preserve original reviews for traceability and reprocessing. |
| 🥈 **Silver** | Clean, normalise, deduplicate, and prepare text with PySpark. |
| 🥇 **Gold** | Store NLP-enriched data, including sentiment analysis and summaries. |

> **Architecture note:** This is a conceptual view of the system, not a diagram of a production deployment. Batch processing and streaming ingestion are part of the project's design.

## 🧠 NLP & Machine Learning

| Component | Technique / Model | Purpose |
| :--- | :--- | :--- |
| 🧹 **Preprocessing** | spaCy, text cleaning | Prepare reviews for downstream analysis. |
| 😊 **Sentiment Analysis** | BERT / Transformer-based NLP | Identify sentiment expressed in reviews. |
| 🔎 **Topic Analysis** | Topic modelling | Discover frequently discussed themes. |
| ✍️ **Abstractive Summarization** | BART | Generate readable summaries from review content. |
| 🧩 **Hybrid Summarization** | Extractive + abstractive | Select important content and generate balanced summaries. |

### 📊 Evaluation

The project considers multiple evaluation approaches:

- **Precision, Recall, F1-score** — sentiment classification.
- **ROUGE** — comparison of generated summaries with reference summaries, where available.
- **Qualitative assessment** — coherence, relevance, and usefulness of summaries.

> **Results transparency:** Verified numerical benchmark scores are not included here. No particular accuracy, F1, or ROUGE value is claimed.

## 🛠️ Tech Stack

| Area | Technologies |
| :--- | :--- |
| 🐍 **Language** | Python |
| ⚡ **Big Data** | Apache Spark, PySpark, Spark SQL |
| ☁️ **Storage** | AWS S3, JSON, Parquet |
| 🤗 **NLP & Deep Learning** | spaCy, Hugging Face Transformers, PyTorch, BERT, BART |
| 📈 **Evaluation** | ROUGE, classification metrics |
| 🔌 **Backend** | FastAPI |
| 🖥️ **Frontend** | HTML, CSS, JavaScript |
| 🔧 **Version Control** | Git, GitHub |

## 📂 Repository Structure

```text
LABGDD/
├── conf/                # Project configuration
├── data/                # Local project data
├── frontend/            # Web interface
├── jobs/                # Data processing jobs
├── libs/                # Shared code and utilities
├── logs/                # Local logs
├── serving_api/         # API layer
├── sql/                 # SQL resources
├── create_sample.py     # Sample data utility
├── antes_de_correr.txt  # Local Spark / Java setup notes
└── requirements.txt     # Python dependencies
```

## 🚀 Getting Started

### ✅ Prerequisites

- Python and `pip`.
- Java and a compatible environment for **PySpark 3.5.1**.
- AWS permissions and credentials **if running the S3-backed stages**.
- Network access for any model downloads required by the configured NLP pipeline.

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ClarasrCoelho/LABGDD.git
cd LABGDD
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

The supplied `requirements.txt` includes core Spark and NLP dependencies. **Additional packages may be required** by particular API or frontend components; check their imports when setting up those services.

### 4️⃣ Configure Spark and AWS

Review the files in `conf/` and `antes_de_correr.txt`. The latter includes local Windows PowerShell examples for configuring `SPARK_HOME`, `JAVA_HOME`, and `PATH`.

Use your local AWS configuration or another appropriate credential mechanism to access S3.

> 🔐 **Security:** Never commit AWS access keys, credentials, or `.env` files containing secrets.

### 5️⃣ Run the pipeline and application

The main components are organised in these folders:

- `jobs/` — data-processing jobs.
- `serving_api/` — API serving layer.
- `frontend/` — web interface.

> ⚠️ **Setup note:** The precise commands, execution order, configuration values, and API start-up instructions still need to be verified against the project scripts before the application can be described as fully reproducible.

## 💡 What This Project Demonstrates

- Designing a **cloud-backed Medallion Data Lake**.
- Developing **ETL pipelines with distributed processing**.
- Applying **Transformer-based NLP** to large volumes of text.
- Serving model outputs through an **API and web application**.
- Combining **batch and streaming** approaches in a single architecture.

**Scope:** This is an **academic prototype**, not a claimed production deployment.

## 🔮 Future Improvements

- [ ] Document tested commands for each processing stage and the API.
- [ ] Add an architecture image and screenshots of the web interface.
- [ ] Include sample inputs and outputs where licensing and privacy permit.
- [ ] Publish reproducible sentiment and summarization evaluation results.
- [ ] Add automated tests, monitoring, and CI workflows.

## Academic Context

Developed as an academic course project.

---

<div align="center">

⭐ **Interested in the project? Explore the code and its data-processing and NLP components.**

</div>
