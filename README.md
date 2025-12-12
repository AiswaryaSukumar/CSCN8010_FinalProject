# Condors Ask – Student Affairs Self-Service Chatbot

**Course:** CSCN8010 – Machine Learning Frameworks  
**College:** Conestoga College  

---

## 👥 Team Members

| Team Member                                   | Student ID |
|-----------------------------------------------|------------|                    
| Aiswarya Thekkuveettil Thazhath               | 8993970    |   
| Vishnu Sivaraj                                | 9025320    |         
| Rohit Iyer                                    | 8993045    |
| Cemil Caglar Yapici                           | 9081058    |

Team information follows the same format as the original workshop README. :contentReference[oaicite:0]{index=0}  

---

## 🧩 Project Overview

**Condors Ask** is a Student Affairs self-service chatbot for Conestoga students.  
It uses a TF-IDF–based retrieval system over curated FAQ CSV files (orientation, career centre, student rights, success portal). For each student question, the app:

1. Cleans the query  
2. Retrieves the most similar FAQ(s) using TF-IDF  
3. Returns a concise answer from the knowledge base  
4. Optionally shows debug info: matched FAQ, similarity, and source

The app is built with **Streamlit** and runs locally in a browser.

---

## 📁 Repository Structure (simplified)

```text
CSCN8010_FinalProject/
│  app.py                  # Main Streamlit app
│  requirements.txt        # Project dependencies
│  README.md               # This file
│  .env                    # (optional) environment variables
│
├─ data/
│   student_affairs_knowledge_base.csv   # Unified FAQ knowledge base
│   career_centre_faq.csv
│   orientation_faq.csv
│   student_rights_faq.csv
│   success_portal_resources.csv
│
├─ model/ or models/
│   tfidf_vectorizer.pkl   # Saved TF-IDF vectorizer
│   kb_tfidf_matrix.npz    # TF-IDF matrix for FAQ questions
│
└─ src/
    retrieval_service.py   # Loads KB, TF-IDF, answers queries
    intent_classifier.py   # (optional) LLM intent classification
    llm_service.py         # (optional) LLM answer generation helpers
    translation.py         # Translates the text
    hf_models.py
