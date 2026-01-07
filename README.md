🏥 Multimodal AI Healthcare Diagnosis System
Overview
This project is an AI-powered healthcare diagnosis assistant that supports voice-to-voice, image-to-voice, and text-to-text interactions. It leverages LLMs, vector databases, and multimodal pipelines to provide contextual, document-grounded medical insights.
The system is designed to simulate a real-world AI healthcare assistant capable of understanding user input across multiple modalities and responding with accurate, explainable outputs.

Key Features
Voice-to-Voice Diagnosis
Converts patient speech to text, performs AI-driven diagnosis using medical documents, and returns responses as synthesized speech.

Image-to-Voice Diagnosis
Accepts medical images or image-based queries, extracts context, performs reasoning using an LLM, and responds via voice output.

Text-to-Text Diagnosis
Supports chat-based medical queries with document-grounded, context-aware responses.

Document-Based Medical Reasoning
Uses PDF medical documents as a trusted knowledge source to avoid hallucinations.

Secure Authentication
User login and session management implemented using Supabase.


Architecture
Data Source
Medical PDFs used as the primary knowledge base.

Embedding & Storage
Documents are chunked and converted into vector embeddings.
Embeddings are stored and indexed in Pinecone for fast semantic retrieval.

Retrieval-Augmented Generation (RAG)
Relevant document chunks are retrieved based on user queries.
LangChain orchestrates retrieval and prompt construction.

LLM Reasoning
Responses are generated using a Groq-hosted LLM via API.

Multimodal Processing
Speech-to-Text for voice input
Text-to-Speech for voice output
Image input processing for visual context understanding

Frontend & Auth
Supabase handles authentication and user management.


Tech Stack
Programming Language: Python
LLM Orchestration: LangChain
Vector Database: Pinecone
LLM Provider: Groq API
Speech-to-Text: Python STT libraries
Text-to-Speech: Python TTS libraries
Authentication: Supabase
Data Source: PDF medical documents

Use Cases
AI-assisted preliminary medical diagnosis
Multimodal healthcare chatbots
Clinical document question answering
Voice-enabled medical assistants

Project Highlights
End-to-end RAG pipeline implementation
Multimodal AI (voice, image, text) integration
Secure authentication with real-world login flow
Scalable vector search for low-latency responses
Healthcare-focused AI system design


Configure environment variables

GROQ_API_KEY=your_api_key\
PINECONE_API_KEY=your_api_key\
SUPABASE_URL=your_url\
SUPABASE_KEY=your_key



Explainability layer for diagnosis reasoning

Deployment using Docker and cloud services
