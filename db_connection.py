"""Database connection management for pgvector."""

import psycopg2
from psycopg2.extras import RealDictCursor
from config import CONNECTION_PARAMS, VECTOR_DIMENSION


class DatabaseConnection:
    """Manages PostgreSQL database connection with pgvector support."""

    def __init__(self):
        """Initialize database connection."""
        self.conn = None

    def connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(**CONNECTION_PARAMS)
            print("✓ Successfully connected to database")
            return self.conn
        except psycopg2.Error as e:
            print(f"✗ Database connection failed: {e}")
            raise

    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")

    def create_items_table(self):
        """Create items table with vector column."""
        try:
            with self.conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create table with vector column
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS items (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        embedding vector({VECTOR_DIMENSION}) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                self.conn.commit()
                print("✓ Items table created successfully")
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"✗ Failed to create table: {e}")
            raise

    def execute_query(self, query, params=None):
        """Execute a database query."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                self.conn.commit()
                return cur.rowcount
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"✗ Query execution failed: {e}")
            raise

    def fetch_query(self, query, params=None):
        """Execute a query and fetch results."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
        except psycopg2.Error as e:
            print(f"✗ Query fetch failed: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
