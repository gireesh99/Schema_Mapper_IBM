# Schema_Mapper_IBM

An AI-powered tool that automatically maps database fields from MySQL to MongoDB using semantic embeddings and LLM reasoning.

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Setup

1. **Create environment file:**
   ```bash
   cp .env.example .env
   echo "OPENAI_API_KEY=sk-..." >> .env
   ```

2. **Install dependencies:**
   ```bash
   uv venv
   uv pip install -r pyproject.toml
   ```

3. **Run the mapper:**
   ```bash
   PYTHONPATH=src .venv/bin/python -m schema_mapper.main
   ```

Output is saved to `output/field_mappings.json`

## How It Works

**3-Stage Pipeline:**

1. **Schema Parsing** (`schema_parser.py`)
   - Reads MySQL and MongoDB schemas from JSON
   - Handles nested MongoDB objects
   - Creates searchable field objects

2. **Hybrid Search** (`embeddings.py`)
   - **Semantic Search**: OpenAI embeddings + FAISS vector index
   - **Keyword Search**: BM25 or SequenceMatcher for string similarity
   - **Hybrid Scoring**: Combines both signals (70% semantic, 30% keyword)

3. **LLM Refinement** (`llm.py`)
   - GPT validates proposed mappings
   - Generates reasoning for each mapping
   - Identifies unmappable fields
   - Produces transformation notes

## Input & Output

**Input:**
- `src/schemas/source_schema.json` - MySQL schema (33 fields)
- `src/schemas/target_schema.json` - MongoDB schema (30 fields)

**Output:**
```json
{
  "mapping_version": "1.0",
  "source": "legacy_hrm (MySQL)",
  "destination": "people_platform (MongoDB)",
  "tables": [
    {
      "source_table": "emp_master",
      "destination_collection": "employees",
      "confidence": 0.45,
      "field_mappings": [
        {
          "source_field": "emp_id",
          "destination_field": "employees._id",
          "type_transform": "INT -> ObjectId",
          "reasoning": "Primary key mapping",
          "notes": "Convert INT to ObjectId",
          "confidence": 0.51
        }
      ]
    }
  ]
}
```

## Architecture

```
Input Schemas
    ↓
Schema Parser → Extract 33 source + 30 target fields
    ↓
Embeddings → Generate OpenAI embeddings for all fields
    ↓
FAISS Index → Build vector index for similarity search
    ↓
Hybrid Search → BM25 + Semantic scoring (top-k candidates)
    ↓
LLM Refinement → GPT validates and generates reasoning
    ↓
Output → JSON with mappings, confidence, and explanations
```

## Key Features

✅ **Semantic Understanding** - Embeddings capture field meaning  
✅ **Hybrid Search** - Combines keyword + semantic matching  
✅ **LLM Validation** - GPT ensures mapping correctness  
✅ **Confidence Scores** - Shows mapping reliability (0-1 scale)  
✅ **Reasoning** - Human-readable explanation for each mapping  
✅ **Nested Object Handling** - Supports MongoDB nested fields  
✅ **Table Context** - Groups fields by source table → destination collection  

## Configuration

Edit weights in `src/schema_mapper/embeddings.py`:
```python
hybrid_score = semantic_score * 0.7 + keyword_score * 0.3
```

Change LLM model in `src/schema_mapper/llm.py`:
```python
model="gpt-5.4-mini-2026-03-17" or any newer models
```

## Results

- **33/33** source fields mapped
- **29/30** target fields utilized
- **Average confidence**: 0.45-0.60 per table
- **Processing time**: ~2-3 minutes (depends on API latency)

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

```
├── src/schema_mapper/
│   ├── main.py              # Pipeline orchestration
│   ├── schema_parser.py      # Parse JSON schemas
│   ├── embeddings.py         # Semantic + hybrid search
│   ├── llm.py               # GPT reasoning & validation
│   └── __init__.py
├── src/schemas/
│   ├── source_schema.json   # MySQL schema
│   └── target_schema.json   # MongoDB schema
├── output/
│   └── field_mappings.json  # Generated mappings
├── pyproject.toml           # Dependencies
└── README.md
```

## Troubleshooting

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY=sk-...
```

**Module not found errors**
```bash
export PYTHONPATH=src
```

**Rate limit errors**
- OpenAI API has rate limits
- Wait a few seconds and retry
- Use a higher-tier API key for faster limits

## Future improvement
- do batch processing for table fields instead of single fields

## License

Interview assignment for IBM.
