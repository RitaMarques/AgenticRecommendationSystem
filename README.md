# Nintendo Switch Product Recommendation System - Lightweight Pilot

## 🧠 Overview

This project implements an LLM-powered, multi-agent recommendation system for Nintendo Switch products.

The system is designed as a graph of specialized agents:

- **Supervisor Agent** — interprets user intent, routes tasks to appropriate workers.
- **Querying Agent** — retrieves relevant product and co-occurrence data from the database.
- **Recommendation Agent** — generates final recommendations using the retrieved data.

It's a lightweight implementation developed in a short-period of time based on LangGraph, LangChain, and PostgreSQL to provide data-driven recommendations. 

## 🚀 Features
- Multi-agent orchestration with LangGraph state management.
- Database-backed product search (semantic similarity search using embeddings - PostgreSQL + pgvector).
- Co-occurrence analysis to suggest related products often purchased together.

---

## 🏗️ Architecture
```bash
                                      +-----------+                                
                                      | __start__ |                                
                                      +-----------+                                
                                             *                                     
                                             *                                     
                                             *                                     
                            +--------------------------------+                     
                            |         Supervisor Agent       |                     
                            +--------------------------------+                     
                        .......              .               ......                
                   .....                     .                     ......          
               ....                          .                           ......    
+----------------+           +--------------------------------+                 ....
| Querying Agent |           |      Recommendation Agent      |                ...  
+----------------+           +--------------------------------+             ...     
                                                     ...               ....        
                                                        ...         ...            
                                                           ..     ..               
                                                          +---------+              
                                                          | __end__ |              
                                                          +---------+ 
```


## 📁 Repo structure
```bash
.
├── 00_raw_data/            # original data files 
├── 01_clean_data/          # ingestion-ready data files after data cleaning, processing and preparation
├── agentic_system/         # agentic system modules
│ ├── agents/               # agents definitions
│ │ ├── querying_agent.py/
│ │ ├── recommendation_agent.py/
│ │ ├── supervisor_agent.py/
│ ├── db/                   # database utils
│ │ ├── db_conn.py/         # connection and session management
│ │ ├── db_schemas.py/      # SQL Alchemy ORM definitions for quick setup and cross-tech usage
│ ├── utils/
│ │ ├── llm.py/             # configures LLM clients for tools and inference
│ │ ├── utils.py/           # embedding generation, shared state definitions
├── 00_data_setup.ipynb/    # notebook for data understanding and manipulation
├── 01_insert_data.py/      # 1-time run script to ingest data to PostgreSQL database
├── main.py/                # builds and executes the LangGraph state graph (run agentic system)
├── requirements.txt/         # python packages to install
├── .env                    # environment variables
└── README.md               # documentation (this file)
```

---

## Limitations & short-term developments

- **Querying qgent reliability and consistency**
While the querying agent demonstrates promising behavior—such as understanding complex user queries, there are notable inconsistencies:
    - The agent occasionally skips over invoking all relevant tools (e.g., neglecting to call the co-occurrences tool despite instructions). This limits the richness and completeness of recommendations as it does not correlate and/or validate information, recommending products that might not be in the user's request category. In this use case, the system works because the LLM has external knowledge on Nintendo Switch products, but in a real-life applications that might not be the case. This can be overcome by iterating over prompts, testing different LLM models, but specially by changing the Agent flow (implementing a more deterministic flow).
    - The variability of responses is influenced by the LLM’s inferencing temperature parameter (currently set to 0.2), which introduces some creativity but also unpredictability like answering beyond the provided database information. This parameter requires deeper study and iterative refinement to effectively meet client expectations. Additionally, stricter guardrails and prompt engineering need to be implemented to enforce data adherence.
- **Limited constraint awareness in database queries**
Currently, product filtering is not applied, only semantic search is used. There is no fine-grained, constraint-aware filtering within SQL queries to efficiently narrow down results by attributes such as major_category, min_age, franchise, or store availability. This results in unnecessary data retrieval and suboptimal performance.
- **Lack of user interface and interaction experience**
The system is currently accessible only as a backend Python module that handles single, standalone queries without any conversational memory or follow-up. It lacks a conversational-flow and user-friendly UI or chat interface to facilitate natural interactions.

## Future Work
1. **Enhanced query optimization and filtering**: introduce advanced SQL filtering capabilities that directly leverage product metadata (e.g., min_age, major_category, franchise, availability by store) to reduce data retrieval overhead and improve recommendation precision. This may include dynamic query building based on parsed user constraints.
2. **Multi-Tool orchestration**: refine agent orchestration logic and prompt design to ensure all relevant tools are invoked consistently, maximizing information coverage and improving recommendation quality. 
3. **Interactive user interface development**: build a chat-like frontend to deliver conversational recommendations, handle follow-up questions, and maintain context across sessions. Integrate user input validation and feedback capture.
4. **Feedback loop**: implement mechanisms to collect user feedback on recommendations (acceptance, rejection, rating). It will enable to collect metrics on how good the recommendation system is, and act upon improving system accuracy over time by, for example, prompt engineering, adjusting agent orchestration.
5. **API containerization**: package the system into a containerized microservice architecture supporting REST APIs. This will enable scalable deployments, easier integration, and parallel request handling using multithreading for example.

---

## 🔧 Installation & Setup
### 📝 Prerequisites
 - Python 3.10+
 - PostgreSQL 14+ with pgvector extension installed.
 - OpenAI API key (for LLM inference and embeddings).
    
### ⚙️ Setup
#### 1️⃣ Clone the repo
```bash
git clone https://github.com/RitaMarques/AgenticRecommendationSystem.git
cd AgenticRecommendationSystem
```

#### 2️⃣ Create a python virtual environment and activate it

#### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

#### 4️⃣ Configure environment variables
Create an `.env` file at the root:
```bash
    DB_USER="postgres"
    DB_PASSWORD=""
    DB_NAME="recommendation"
    OPENAPI_KEY=""
    OPENAI_EMB_MODEL="text-embedding-3-small"
    OPENAI_INFER_MODEL="gpt-4.1-nano-2025-04-14"
    OPENAI_TOOL_MODEL="gpt-4.1-nano-2025-04-14"
```

#### 5️⃣ Prepare the database
Run the data ingestion script to populate the PostgreSQL database
```bash
python 01_insert_data.py
```

#### 6️⃣ Run the recommendation system
```bash
python main.py "<your user query>"
```
*Note: you'll see application debug prints*

