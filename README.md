


https://github.com/user-attachments/assets/f0d4fff0-842f-49ae-987c-72fd6fa457c2


# FMCSA_Chatbot
Approach to respond to user queries using a knowledge base

This is a knowledge Bot that retrieves information from a Vector Database containing FMCSA(Federal Motor Carrier Safety Administration) rules that were extracted from PDFs that were uploaded to the vector store. Utilized Llama3-8b LLM to provide accurate and contextually relevant responses to user queries. Created prompts that effectively gave appropriate answers based on the questions asked.

The system was tested with dummy questions that were outside the scope of the uploaded documents to evaluate its ability to manage AI hallucination. The bot performed exceptionally well, ignoring out-of-scope questions and only providing accurate responses based on the content within the database.

# Architecture
![kb3](https://github.com/user-attachments/assets/818ac4bf-a4a4-4c2f-a313-04d5e8d1921c)

# Tools and Technologies Used: 
1) Python programming Language
2) Groq(for LLM API) -- Used Llama 3 8b as the large language model
3) Langchain
4) ChromaDB(Vector Database)
5) PdfReader
6) Sentence-Transformers for creating embeddings of text data
