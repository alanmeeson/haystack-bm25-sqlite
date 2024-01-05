# Haystack BM25 SQLite DocumentStore

An SQLite based DocumentStore for [Haystack](https://haystack.deepset.ai/) 1.x which uses the 
[SQLite FTS module](https://sqlite.org/fts5.html) to provide BM25 based keyword search.

This DocumentStore was developed to fill a gap in the Haystack lineup of DocumentStores.  The off the shelf offerings 
cover a range of databases (eg: Elasticsearch, Weaviate, etc.) and some in-memory or similarly infrastructureless 
options (eg: FAISS).  The trouble is, there isn't an option which covers three key requirements for throwing together a 
nice prototype of a hybrid search system:

i. Doesn't require a separate server
ii. Can handle BM25 Keyword Search
iii. Can persist, or at least can do so without just pickling

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install haystack_sqlite.

```bash
pip install git+git+https://github.com/alanmeeson/haystack_sqlite
```

## Usage

```python
import os
from haystack.nodes import PreProcessor, BM25Retriever, PDFToTextConverter
from haystack.pipelines import Pipeline
from haystack_sqlite import SQLiteDocumentStore

data_dir = "/path/to/some/pdfs"
sqlite_url = "sqlite:///:memory:"  # or 'sqlite:///keyword_index.sqlite'

# Declare the document store
keyword_document_store = SQLiteDocumentStore(url=sqlite_url)

# Create the indexing pipeline
converter = PDFToTextConverter(
    remove_numeric_tables=True,
    valid_languages=['en']
)

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=512,
    split_overlap=32,
    split_respect_sentence_boundary=True
)

index_pipeline = Pipeline()
index_pipeline.add_node(component=converter, name="Converter", inputs=["File"])
index_pipeline.add_node(component=preprocessor, name="Preprocessor", inputs=["Converter"])
index_pipeline.add_node(component=keyword_document_store, name="KeywordIndex", inputs=["Preprocessor"])

# Index all the pdfs in the data_dir into the document store
files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.lower().endswith('pdf')]
index_pipeline.run(file_paths=files)

# Declare the Query Pipeline
sparse_retriever = BM25Retriever(document_store=keyword_document_store)
query_pipeline = Pipeline()
query_pipeline.add_node(component=sparse_retriever, name="SparseRetriever", inputs=["Query"])

# Perform a query
result = query_pipeline.run(query="some interesting query here")
```

## The Todo List

There are currently four improvements on the 'roadmap'.  These are in no particular order, and will be gotten around to 
when I get around to it.

### Improved support separate indexes
The SQLDocumentStore captures all documents in the same table, and separates by index with a column.  The search 
functionality implemented here will search across the whole table, and then filter down by index.  Investigate whether 
generating separate fts indexes for each DocumentStore index would be better or a waste of time.

### Parameterise the BM25 index configuration
Currently the settings for the BM25 index are hard coded; the bm25_parameters arg for the constructor currently does 
nothing.  We could make it so that the tokenizer, stemmer, etc. could be configured. 

### Re-implement more efficiently
At present this is a thin layer added onto the SQLDocumentStore to add in a bm25 index and make it implement the
KeywordDocumentStore abstract class.  
As such it is using SQLAlchemy under the hood, and tackling the search and the filtering as two separate steps.  This is 
fine for small enough corpuses, but it's a bit inefficient.  Re-implementing the backing DocumentStore and coupling it
more tightly to SQLite could make it more efficient.  Eg: performing the search and filter in a single query.

### Make a Haystack 2.0 version
This version is implemented against Haystack V1.24.  There is a new version of Haystack in the works which is 
incompatible.  It would be nice to have this work for the new version,  unless there's already something there that 
would fill this niche.

### Add unit tests
This is currently manually tested, and lightly so at that.  Adding unit tests would be a good idea.  Some high level 
paths to cover include:
1. Search with keyword query
2. Search with keyword query and filters
3. Add documents
4. Edit documents
5. Delete documents
 