import numpy as np
import json
import time
import uuid
import ollama

from .memory_store import MemoryStore
from .storage import BaseStorage
from .in_memory_storage import InMemoryStorage

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage


class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")


class MemoryManager:
    """
    Manages the memory store, including loading and saving history,
    adding interactions, retrieving relevant interactions, and generating responses.
    """

    def __init__(self, api_key, chat_model="ollama", chat_model_name="llama3.1:8b", embedding_model="ollama", embedding_model_name="mxbai-embed-large", storage=None):
        self.api_key = api_key
        self.chat_model_name = chat_model_name
        self.embedding_model_name = embedding_model_name

        # Set chat model
        if chat_model.lower() == "openai":
            self.llm = ChatOpenAI(model=chat_model_name, api_key=self.api_key)
        elif chat_model.lower() == "ollama":
            self.llm = ChatOllama(model=chat_model_name, temperature=0)
        else:
            raise ValueError("Unsupported chat model. Choose either 'openai' or 'ollama'.")

        # Set embedding model and dimension
        if embedding_model.lower() == "openai":
            if embedding_model_name == "text-embedding-3-small":
                self.dimension = 1536
            else:
                raise ValueError("Unsupported OpenAI embedding model name for specified dimension.")
            self.embeddings_model = OpenAIEmbeddings(model=embedding_model_name, api_key=self.api_key)
        elif embedding_model.lower() == "ollama":
            if embedding_model_name == "mxbai-embed-large":
                self.dimension = 1024
            else:
                self.dimension = self.initialize_embedding_dimension()
            self.embeddings_model = lambda text: ollama.embeddings(model=self.embedding_model_name, prompt=text)["embedding"]
        else:
            raise ValueError("Unsupported embedding model. Choose either 'openai' or 'ollama'.")

        # Initialize memory store with the correct dimension
        self.memory_store = MemoryStore(dimension=self.dimension)

        if storage is None:
            self.storage = InMemoryStorage()
        else:
            self.storage = storage

        self.initialize_memory()

    def initialize_embedding_dimension(self):
        """
        Retrieve embedding dimension from Ollama by generating a test embedding.
        """
        print("Determining embedding dimension for Ollama model...")
        test_text = "Test to determine embedding dimension"
        response = ollama.embeddings(
            model=self.embedding_model_name,
            prompt=test_text
        )
        embedding = response.get("embedding")
        if embedding is None:
            raise ValueError("Failed to retrieve embedding for dimension initialization.")
        return len(embedding)

    def standardize_embedding(self, embedding):
        """
        Standardize embedding to the target dimension by padding with zeros or truncating.
        """
        current_dim = len(embedding)
        if current_dim == self.dimension:
            return embedding
        elif current_dim < self.dimension:
            # Pad with zeros
            return np.pad(embedding, (0, self.dimension - current_dim), 'constant')
        else:
            # Truncate to match target dimension
            return embedding[:self.dimension]

    def load_history(self):
        return self.storage.load_history()

    def save_memory_to_history(self):
        self.storage.save_memory_to_history(self.memory_store)

    def add_interaction(self, prompt, output, embedding, concepts):
        timestamp = time.time()
        interaction_id = str(uuid.uuid4())
        interaction = {
            "id": interaction_id,
            "prompt": prompt,
            "output": output,
            "embedding": embedding.tolist(),
            "timestamp": timestamp,
            "access_count": 1,
            "concepts": list(concepts),
            "decay_factor": 1.0,
        }
        self.memory_store.add_interaction(interaction)
        self.save_memory_to_history()

    def get_embedding(self, text):
        print(f"Generating embedding for the provided text...")
        if callable(self.embeddings_model):  # If embeddings_model is a function, use it directly
            embedding = self.embeddings_model(text)
        else:  # OpenAI embeddings
            embedding = self.embeddings_model.embed_query(text)
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
        standardized_embedding = self.standardize_embedding(embedding)
        return np.array(standardized_embedding).reshape(1, -1)

    def extract_concepts(self, text):
        print("Extracting key concepts from the provided text...")

        # Set up parser and prompt template for structured output with more specific instructions
        parser = JsonOutputParser(pydantic_object=ConceptExtractionResponse)

        if isinstance(self.llm, ChatOpenAI):
            # OpenAI-specific concept extraction
            prompt = PromptTemplate(
                template=(
                    "Extract key concepts from the following text in a concise, context-specific manner. "
                    "Include only highly relevant and specific concepts.\n"
                    "{format_instructions}\n{text}"
                ),
                input_variables=["text"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
        elif isinstance(self.llm, ChatOllama):
            # Ollama-specific concept extraction
            prompt = PromptTemplate(
                template=(
                    "Please analyze the following text and provide a list of key concepts that are unique to this content. "
                    "Return only the core concepts that best capture the text's meaning.\n"
                    "{format_instructions}\n{text}"
                ),
                input_variables=["text"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

        # Execute the chain based on the appropriate prompt
        chain = prompt | self.llm | parser
        response = chain.invoke({"text": text})

        # Access the extracted concepts from the dictionary response
        concepts = response.get("concepts", [])
        print(f"Concepts extracted: {concepts}")
        return concepts

    def initialize_memory(self):
        short_term, long_term = self.load_history()
        for interaction in short_term:
            # Standardize the dimension of each interaction's embedding
            interaction['embedding'] = self.standardize_embedding(np.array(interaction['embedding']))
            self.memory_store.add_interaction(interaction)
        self.memory_store.long_term_memory.extend(long_term)

        self.memory_store.cluster_interactions()
        print(f"Memory initialized with {len(self.memory_store.short_term_memory)} interactions in short-term and {len(self.memory_store.long_term_memory)} in long-term.")

    def retrieve_relevant_interactions(self, query, similarity_threshold=40, exclude_last_n=0):
        query_embedding = self.get_embedding(query)
        query_concepts = self.extract_concepts(query)
        return self.memory_store.retrieve(query_embedding, query_concepts, similarity_threshold, exclude_last_n=exclude_last_n)

    def generate_response(self, prompt, last_interactions, retrievals, context_window=3):
        context = ""
        if last_interactions:
            context_interactions = last_interactions[-context_window:]
            context += "\n".join([f"Previous prompt: {r['prompt']}\nPrevious output: {r['output']}" for r in context_interactions])
            print(f"Using the following last interactions as context for response generation:\n{context}")
        else:
            context = "No previous interactions available."
            print(context)

        if retrievals:
            retrieved_context_interactions = retrievals[:context_window]
            retrieved_context = "\n".join([f"Relevant prompt: {r['prompt']}\nRelevant output: {r['output']}" for r in retrieved_context_interactions])
            print(f"Using the following retrieved interactions as context for response generation:\n{retrieved_context}")
            context += "\n" + retrieved_context

        messages = [
            SystemMessage(content="You're a helpful assistant."),
            HumanMessage(content=f"{context}\nCurrent prompt: {prompt}")
        ]
        
        response = self.llm.invoke(messages)

        return response.content.strip()
