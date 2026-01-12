from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


class RAGGenerator:
    """
    Handles the generation step of the RAG pipeline using local Ollama models.

    This component uses local open-source models via Ollama to generate answers 
    while enforcing strict citation requirements to reduce financial hallucinations.
    No API keys or internet connection required.

    Attributes:
        llm: ChatOllama instance configured for financial analysis
        prompt: Template for structuring prompts with context and citations
        model_name: Name of the local Ollama model being used
    """

    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the RAG generator with specified local Ollama model.

        Args:
            model_name: Ollama model to use for generation
                       Popular options: "llama3.2:3b", "llama3.1:8b", "mistral:7b", "qwen2.5:7b"
        """
        self.model_name = model_name

        # Initialize local Ollama model (no API key required)
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,  # CRITICAL for financial RAG to reduce hallucinations
            timeout=120     # Increased timeout for local model processing
        )
        print(f"✓ Using local Ollama model: {model_name}")
        print("💡 Tip: Make sure Ollama is running with 'ollama serve' and the model is downloaded")

        # Strict prompt to enforce citations and honesty
        self.prompt = ChatPromptTemplate.from_template("""
        You are a senior financial analyst assistant.
        Answer the user's question based ONLY on the following context.

        Rules:
        1. If the answer is not in the context, explicitly say "I do not have enough information to answer that."
        2. Cite your sources for every key fact using the format [Source: filename].
        3. Keep the answer professional, concise, and direct.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)

    def format_docs(self, docs: List[Document]) -> str:
        """
        Prepares documents for the LLM by adding source metadata to the text.

        This method ensures that the LLM has access to source information
        for proper citation in the generated response.

        Args:
            docs: List of retrieved documents with content and metadata

        Returns:
            Formatted string containing document content with source references
        """
        formatted = []
        for d in docs:
            # We add the filename to the text so the LLM knows what to cite
            source = d.metadata.get('source', 'Unknown')
            content = d.page_content.replace("\n", " ")
            formatted.append(f"Content: {content}\nSource: {source}")
        return "\n\n".join(formatted)

    def generate(self, query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        """
        Executes the generation step of the RAG pipeline.

        Takes user query and retrieved documents to generate a comprehensive
        answer with proper citations and source tracking.

        Args:
            query: User's financial question
            retrieved_docs: Documents retrieved from the vector store

        Returns:
            Dictionary containing:
                - answer_text: Generated response from the LLM
                - citations: List of unique source files referenced
                - retrieved_chunks: Raw content of retrieved document chunks
        """
        # Handle case where no relevant documents were retrieved
        if not retrieved_docs:
            return {
                "answer_text": "I could not find any relevant documents to answer your question.",
                "citations": [],
                "retrieved_chunks": []
            }

        # 1. Prepare Context - Format documents with source information
        context_text = self.format_docs(retrieved_docs)

        # 2. Build Chain - Create the processing pipeline
        chain = self.prompt | self.llm | StrOutputParser()

        # 3. Invoke - Execute the generation with formatted context
        response = chain.invoke({
            "context": context_text,
            "question": query
        })

        # 4. Structure Output (Task 4 requirement) - Return standardized response
        return {
            "answer_text": response,
            "citations": list(set([d.metadata.get('source') for d in retrieved_docs])),
            "retrieved_chunks": [d.page_content for d in retrieved_docs]
        }
