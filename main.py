"""Main example script demonstrating pgvector usage."""

import numpy as np
from db_connection import DatabaseConnection
from vector_operations import VectorStore


def main():
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
                ("apple", np.array([1.0, 0.5, 0.2])),
                ("banana", np.array([0.9, 0.6, 0.1])),
                ("orange", np.array([0.95, 0.55, 0.15])),
                ("grape", np.array([0.2, 0.8, 0.9])),
                ("blueberry", np.array([0.1, 0.7, 0.95])),
            ]
            
            for name, embedding in items_data:
                vector_store.insert_item(name, embedding)
            
            # Example 2: Retrieve all items
            print("\n--- All Items ---")
            all_items = vector_store.get_all_items()
            for item in all_items:
                print(f"ID: {item['id']}, Name: {item['name']}, Embedding: {item['embedding']}")
            
            # Example 3: Similarity search
            print("\n--- Similarity Search ---")
            query_vector = np.array([0.9, 0.6, 0.15])
            similar_items = vector_store.similarity_search(query_vector, limit=3)
            print(f"Similar to {query_vector}:")
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
