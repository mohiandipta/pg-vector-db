"""Vector operations for pgvector database."""

import numpy as np
from db_connection import DatabaseConnection


class VectorStore:
    """Handle vector embeddings storage and retrieval."""

    def __init__(self):
        """Initialize vector store."""
        self.db = None

    def insert_item(self, name, embedding):
        """
        Insert an item with vector embedding.
        
        Args:
            name: Item name
            embedding: Vector embedding (list or numpy array)
        """
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        query = """
            INSERT INTO items (name, embedding)
            VALUES (%s, %s)
            RETURNING id, name, embedding;
        """
        
        self.db.execute_query(query, (name, embedding_list))
        print(f"✓ Inserted item: {name}")

    def get_item(self, item_id):
        """Retrieve an item by ID."""
        query = "SELECT id, name, embedding FROM items WHERE id = %s;"
        result = self.db.fetch_query(query, (item_id,))
        return result[0] if result else None

    def get_all_items(self):
        """Retrieve all items from the database."""
        query = "SELECT id, name, embedding FROM items ORDER BY id;"
        return self.db.fetch_query(query)

    def similarity_search(self, query_embedding, limit=5):
        """
        Find items most similar to query embedding using cosine distance.
        
        Args:
            query_embedding: Query vector
            limit: Number of results to return
            
        Returns:
            List of similar items with distance scores
        """
        embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        query = """
            SELECT id, name, embedding,
                   (1 - (embedding <=> %s::vector)) as similarity
            FROM items
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        
        return self.db.fetch_query(query, (embedding_list, embedding_list, limit))

    def delete_item(self, item_id):
        """Delete an item by ID."""
        query = "DELETE FROM items WHERE id = %s;"
        self.db.execute_query(query, (item_id,))
        print(f"✓ Deleted item with ID: {item_id}")

    def clear_all_items(self):
        """Delete all items from the database."""
        query = "DELETE FROM items;"
        self.db.execute_query(query)
        print("✓ All items deleted")

    def __enter__(self):
        """Context manager entry."""
        self.db = DatabaseConnection()
        self.db.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.db:
            self.db.disconnect()
