# Search Filters Feature

## New Parameters for `search_docs`

### 1. Filter by specific documentation slugs

```python
# Search only in JavaScript documentation
search_docs(
    query="async await promises",
    slugs=["javascript"],
    top_k=5
)

# Search across multiple specific docs
search_docs(
    query="HTTP client library",
    slugs=["python", "node", "axios"],
    top_k=10
)
```

### 2. Filter by source type

```python
# Search only in devdocs.io documentation
search_docs(
    query="API reference",
    source_type="devdocs",
    top_k=5
)

# Search only in local custom documentation
search_docs(
    query="internal API",
    source_type="local",
    top_k=5
)
```

### 3. Combine filters

```python
# Search for Python-specific content from devdocs only
search_docs(
    query="list comprehensions",
    slugs=["python"],
    source_type="devdocs",
    min_score=0.4,
    top_k=5
)
```

## How It Works

1. **Over-fetching**: Retrieves `top_k * 3` results initially to account for filtering
2. **Post-filter**: Applies slug and source_type filters on full document metadata
3. **Limit**: Returns only `top_k` results after filtering
4. **Feedback**: Displays active filters in the result header

## Example Output

```
Found 5 matching documents (slugs: javascript; source: devdocs):

### 1. [Async Functions](javascript) [score: 0.892] (Function) [devdocs]
Path: async#async-functions

The async function declaration creates a binding of a new async function...

### 2. [Promise](javascript) [score: 0.845] (Object) [devdocs]
Path: promise

The Promise object represents the eventual completion (or failure)...
```

## Use Cases

- **Framework-specific queries**: Search only React docs when working on React
- **Language-specific**: Filter to Python docs when writing Python code
- **Local vs External**: Distinguish between internal docs and public docs
- **Multi-doc comparison**: Search across Python, JavaScript, and TypeScript simultaneously
- **Precision**: Narrow down results when initial query returns too many irrelevant matches
