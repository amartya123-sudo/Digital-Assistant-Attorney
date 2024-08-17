import os
import json
import faiss
from opensearchpy import OpenSearch
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore

class LegalCaseSearchApp:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{'host': os.getenv('OPENSEARCH_HOST'), 'port': os.getenv('OPENSEARCH_PORT')}],
            http_auth=(os.getenv('OPENSEARCH_USER'), os.getenv('OPENSEARCH_PASS')),
            use_ssl=True,
            verify_certs=True,
        )
        self.index_name = 'court_cases'
        self.storage_dir = './storage'
        self.data_file = 'data_MV.json'

        self.vector_store = FaissVectorStore.from_persist_dir(self.storage_dir)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store, persist_dir=self.storage_dir)
        self.index = load_index_from_storage(storage_context=self.storage_context)
        self.query_engine = self.index.as_query_engine(similarity_top_k=5)
        
        self._create_index()
        self._index_documents()

    def _create_index(self):
        self.client.indices.create(index=self.index_name, ignore=400)

    def _index_documents(self):
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        for i, doc in enumerate(data):
            self.client.index(index=self.index_name, id=i, body=doc)

    def keyword_search(self, query):
        body = {
            "query": {
                "query_string": {
                    "query": query,
                    "fields": ["content", "case_title", "court_name"]
                }
            }
        }
        return self.client.search(index=self.index_name, body=body)

    def semantic_search(self, query):
        return self.query_engine.query(query)

    def combined_search(self, query):
        keyword_results = self.keyword_search(query)
        semantic_results = self.semantic_search(query)

        keyword_titles = [source['_source']['case_title'] for source in keyword_results['hits']['hits']]
        semantic_titles = [node.metadata['case_title'] for node in semantic_results.source_nodes]

        return [title for title in keyword_titles if title in semantic_titles]

    def get_case_details(self, titles):
        with open(self.data_file, 'r') as db:
            data = json.load(db)
        case_details = []
        for case in data:
            if case['case_title'] in titles:
                case_details.append({
                    'title': case['case_title'],
                    'court_name': case['court_name'],
                    'content': case['content']
                })
        return case_details