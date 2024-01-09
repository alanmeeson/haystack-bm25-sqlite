from typing import Dict, Union, List, Optional

import logging
import numpy as np

from haystack.schema import Document, FilterType
from haystack.document_stores.base import KeywordDocumentStore
from haystack.document_stores.sql import SQLDocumentStore
from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.document_stores.sql import DocumentORM, MetaDocumentORM
from haystack.utils.scipy_utils import expit
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)


class SQLiteDocumentStore(SQLDocumentStore, KeywordDocumentStore):

    def __init__(
        self,
        url: str = "sqlite:///:memory:",
        index: str = "document",
        label_index: str = "label",
        duplicate_documents: str = "overwrite",
        check_same_thread: bool = False,
        isolation_level: Optional[str] = None,
        bm25_parameters: Dict = None
    ):
        """
        An SQLite backed DocumentStore which supports BM25 index.  Leverages the SQLDocumentStore and SQLite FTS.

        See: https://sqlite.org/fts5.html

        :param url: URL for SQL database as expected by SQLAlchemy. More info here: https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
        :param index: The documents are scoped to an index attribute that can be used when writing, querying, or deleting documents.
                      This parameter sets the default value for document index.
        :param label_index: The default value of index attribute for the labels.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param check_same_thread: Set to False to mitigate multithreading issues in older SQLite versions (see https://docs.sqlalchemy.org/en/14/dialects/sqlite.html?highlight=check_same_thread#threading-pooling-behavior)
        :param isolation_level: see SQLAlchemy's `isolation_level` parameter for `create_engine()` (https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.isolation_level)
        :param bm25_parameters: Parameters for BM25 implementation in a dictionary format.
                                For example: {'k1':1.5, 'b':0.75, 'epsilon':0.25}
                                You can learn more about these parameters by visiting https://github.com/dorianbrown/rank_bm25
                                By default, no parameters are set.
        """

        if not url.startswith("sqlite://"):
            raise ValueError("Must be an SQLite database url")

        super().__init__(
            url=url,
            index=index,
            label_index=label_index,
            duplicate_documents=duplicate_documents,
            check_same_thread=check_same_thread,
            isolation_level=isolation_level
        )

        self.bm25_parameters = {} if bm25_parameters is None else bm25_parameters

        self._create_bm25_index()

    def _create_bm25_index(self):
        """Creates the bm25 index table and triggers if they do not already exist."""

        # Creates the FTS index virtual table
        # TODO: look at parameterising the configuration to allow for custom tokenizers, etc.
        self.session.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS document_fts 
            USING fts5("id", "content", "index", tokenize = 'porter unicode61', content='document', content_rowid='rowid');
        """)

        # Creates triggers that update the index when the documents are added/removed/updated
        self.session.execute("""
            CREATE TRIGGER IF NOT EXISTS document_ai AFTER INSERT ON document BEGIN
                INSERT INTO document_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
        """)
        self.session.execute("""
        CREATE TRIGGER IF NOT EXISTS document_ad AFTER DELETE ON document BEGIN
          INSERT INTO document_fts(document_fts, rowid, content) VALUES('delete', old.rowid, old.content);
        END;
        """)
        self.session.execute("""
        CREATE TRIGGER IF NOT EXISTS document_au AFTER UPDATE ON document BEGIN
          INSERT INTO document_fts(document_fts, rowid, content) VALUES('delete', old.rowid, old.content);
          INSERT INTO document_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
        """)

    def query(
        self,
        query: Optional[str],
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        all_terms_must_match: bool = False,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number of documents
        that are most relevant to the query as defined by keyword matching algorithms like BM25.

        :param query: The query
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```

        :param top_k: How many documents to return per query.
        :param custom_query: Custom query to be executed.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to False.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """

        if headers:
            logger.warning("SQLiteDocumentStore does not support headers. This parameter is ignored.")
        if custom_query:
            logger.warning("SQLiteDocumentStore does not support custom_query. This parameter is ignored.")
        if filters:
            logger.warning("SQLiteDocumentStore filters won't work on metadata fields containing compound data types")

        # TODO: investigate whether we can do the query and filter step in one go.

        # Perform the BM25 query
        query_statement = text("""
            SELECT id, bm25(document_fts) as score 
            FROM document_fts
            WHERE document_fts MATCH(:q)
            ORDER BY score;
        """)
        result_set = self.session.execute(query_statement, {'q': query})

        # Get the documents which match the BM25 query
        results_as_dict = result_set.mappings().all()
        valid_doc_ids = [result['id'] for result in results_as_dict]
        doc_scores = {
            result['id']: float(expit(np.asarray(result['score'] / 8))) if scale_score else result['score']
            for result in results_as_dict
        }

        # Loop over the docs, filtering to use the actual filters and the valid docs.
        docs = []
        for doc in self._query(index=index, filters=filters, document_ids=valid_doc_ids):
            doc.score = doc_scores[doc.id]
            docs.append(doc)

        # Sort the results by score, descending
        docs.sort(key=lambda x: x.score, reverse=True)

        # Limit to the top_k results
        if top_k:
            docs = docs[:top_k]

        return docs

    def query_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        all_terms_must_match: bool = False,
        scale_score: bool = True,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the provided queries as defined by keyword matching algorithms like BM25.

        This method lets you find relevant documents for a single query string (output: List of Documents), or a
        a list of query strings (output: List of Lists of Documents).

        :param queries: Single query or list of queries.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```

        :param top_k: How many documents to return per query.
        :param custom_query: Custom query to be executed.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to False.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """

        result_documents = []
        for query in queries:
            result_documents.append(self.query(query=query, top_k=top_k, index=index, scale_score=scale_score))

        return result_documents

    def _query(
            self,
            index: Optional[str] = None,
            filters: Optional[FilterType] = None,
            vector_ids: Optional[List[str]] = None,
            only_documents_without_embedding: bool = False,
            batch_size: int = 10_000,
            document_ids: Optional[List[str]] = None
    ):
        """
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param vector_ids: List of vector_id strings to filter the documents by.
        :param only_documents_without_embedding: return only documents without an embedding.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        index = index or self.index
        # Generally ORM objects kept in memory cause performance issue
        # Hence using directly column name improve memory and performance.
        # Refer https://stackoverflow.com/questions/23185319/why-is-loading-sqlalchemy-objects-via-the-orm-5-8x-slower-than-rows-via-a-raw-my
        documents_query = self.session.query(
            DocumentORM.id, DocumentORM.content, DocumentORM.content_type, DocumentORM.vector_id
        ).filter_by(index=index)

        if document_ids:
            documents_query = documents_query.filter((DocumentORM.id.in_(document_ids)))

        if filters:
            logger.warning("filters won't work on metadata fields containing compound data types")
            parsed_filter = LogicalFilterClause.parse(filters)
            select_ids = parsed_filter.convert_to_sql(MetaDocumentORM)
            documents_query = documents_query.filter(DocumentORM.id.in_(select_ids))

        if only_documents_without_embedding:
            documents_query = documents_query.filter(DocumentORM.vector_id.is_(None))
        if vector_ids:
            documents_query = documents_query.filter(DocumentORM.vector_id.in_(vector_ids))

        documents_map = {}

        if self.use_windowed_query:
            documents_query = self._windowed_query(documents_query, DocumentORM.id, batch_size)

        for i, row in enumerate(documents_query, start=1):
            documents_map[row.id] = Document.from_dict(
                {
                    "id": row.id,
                    "content": row.content,
                    "content_type": row.content_type,
                    "meta": {} if row.vector_id is None else {"vector_id": row.vector_id},
                }
            )
            if i % batch_size == 0:
                documents_map = self._get_documents_meta(documents_map)
                yield from documents_map.values()
                documents_map = {}
        if documents_map:
            documents_map = self._get_documents_meta(documents_map)
            yield from documents_map.values()
