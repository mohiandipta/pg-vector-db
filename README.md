# pgvector Python Example

A complete Python project for working with pgvector (PostgreSQL vector database) integration.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Database Connection
Copy `.env.example` to `.env` and update with your PostgreSQL credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mydatabase
DB_USER=myuser
DB_PASSWORD=mypassword
```

### 3. Ensure pgvector Extension is Installed
In PostgreSQL:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Project Structure

- **`config.py`** - Configuration management and environment variables
- **`db_connection.py`** - Database connection class and table setup
- **`vector_operations.py`** - VectorStore class for embedding operations
- **`main.py`** - Example usage demonstrating CRUD operations

## Usage

### Basic Example
```python
from db_connection import DatabaseConnection
from vector_operations import VectorStore
import numpy as np

db = DatabaseConnection()
db.connect()
db.create_items_table()

with VectorStore() as vector_store:
    vector_store.db = db
    
    # Insert item with embedding
    embedding = np.array([1.0, 0.5, 0.2])
    vector_store.insert_item("apple", embedding)
    
    # Search similar items
    query = np.array([0.9, 0.6, 0.15])
    results = vector_store.similarity_search(query, limit=5)
```

### Run Example
```bash
python main.py
```

## Features

- ✅ Database connection management
- ✅ Vector table creation
- ✅ Insert items with embeddings
- ✅ Similarity search using cosine distance
- ✅ CRUD operations
- ✅ Context manager support
- ✅ Error handling

## Vector Operations Supported

- **Insert**: Add items with vector embeddings
- **Retrieve**: Get items by ID or retrieve all
- **Search**: Find similar items using cosine distance
- **Delete**: Remove items by ID
- **Clear**: Delete all items

## Dependencies

- `psycopg2-binary` - PostgreSQL adapter for Python
- `numpy` - Numerical computing library
- `python-dotenv` - Environment variable management
