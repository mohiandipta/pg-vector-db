"""Main example script demonstrating pgvector usage."""

import numpy as np
from db_connection import DatabaseConnection
from vector_operations import VectorStore
from embedding_service import EmbeddingService


def main():
    embedding_service = EmbeddingService()
    
    """Run pgvector example operations."""
    
    # Initialize database connection
    db = DatabaseConnection()
    db.connect()
    
    try:
        # Create table if it doesn't exist
        db.create_items_table()
        
        # Use VectorStore for vector operations
        with VectorStore() as vector_store:
            vector_store.db = db
            
            # Example 1: Insert items with embeddings
            print("\n--- Inserting Items ---")
            items_data = [
                ("NestJS framework", None),
                ("Python machine learning", None),
                ("PostgreSQL database", None),
                ("Deep learning AI", None),
                ("Redis caching system", None),
            ]
            
            for name, _ in items_data:
                embedding = embedding_service.generate_embedding(name)
                vector_store.insert_item(name, embedding)
            
            # Example 2: Retrieve all items
            print("\n--- All Items ---")
            all_items = vector_store.get_all_items()
            for item in all_items:
                print(f"ID: {item['id']}, Name: {item['name']}, Embedding: {item['embedding']}")
            
            # Example 3: Similarity search
            print("\n--- Similarity Search ---")
            query_text = "Backend framework for Node"
            query_embedding = embedding_service.generate_embedding(query_text)
            similar_items = vector_store.similarity_search(query_embedding, limit=3)
            print(f"Similar to {query_text}:")
            for item in similar_items:
                print(f"  - {item['name']}: similarity={item['similarity']:.4f}")
            
            # Example 4: Retrieve specific item
            print("\n--- Retrieve Item ---")
            item = vector_store.get_item(1)
            if item:
                print(f"Item 1: {item['name']} - {item['embedding']}")
            
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
