from pymilvus import AnnSearchRequest, CollectionSchema, FieldSchema, MilvusClient, DataType, WeightedRanker
import numpy as np
import concurrent.futures

class MilvusColbertRetriever:
    def __init__(self, milvus_client, collection_name, dim=128):
        # Initialize the retriever with a Milvus client, collection name, and dimensionality of the vector embeddings.
        # If the collection exists, load it.
        self.collection_name = collection_name
        self.client = milvus_client
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim

    def create_collection(self):
        # Create a new collection in Milvus for storing embeddings.
        # Drop the existing collection if it already exists and define the schema for the collection.
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        # Create an index on the vector field to enable fast similarity search.
        # Releases and drops any existing index before creating a new one with specified parameters.
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="IVF_FLAT",  # or any other index type you want
            metric_type="IP",  # or the appropriate metric type
            params={
                "M": 16,
                "efConstruction": 500,
            },  # adjust these parameters as needed
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def create_scalar_index(self):
        # Create a scalar index for the "doc_id" field to enable fast lookups by document ID.
        self.client.release_collection(collection_name=self.collection_name)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="doc_id",
            index_name="int32_index",
            index_type="INVERTED",  # or any other index type you want
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def search(self, data, topk):
        # Perform a vector search on the collection to find the top-k most similar documents.
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            self.collection_name,
            data,
            limit=int(50),
            output_fields=["vector", "seq_id", "doc_id"],
            search_params=search_params,
        )
        doc_ids = set()
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                doc_ids.add(results[r_id][r]["entity"]["doc_id"])

        scores = []

        def rerank_single_doc(doc_id, data, client, collection_name):
            # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=f"doc_id in [{doc_id}]",
                output_fields=["seq_id", "vector", "doc"],
                limit=1000,
            )
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
            )
            score = np.dot(data, doc_vecs.T).max(1).sum()
            return (score, doc_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, data, self.client, self.collection_name
                ): doc_id
                for doc_id in doc_ids
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id = future.result()
                scores.append((score, doc_id))

        scores.sort(key=lambda x: x[0], reverse=True)
        if len(scores) >= topk:
            return scores[:topk]
        else:
            return scores

    def insert(self, data):
        # Insert ColBERT embeddings and metadata for a document into the collection.
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        doc_ids = [data["doc_id"] for i in range(seq_length)]
        seq_ids = list(range(seq_length))
        docs = [""] * seq_length
        docs[0] = data["filepath"]

        # Insert the data as multiple vectors (one for each sequence) along with the corresponding metadata.
        self.client.insert(
            self.collection_name,
            [
                {
                    "vector": colbert_vecs[i],
                    "seq_id": seq_ids[i],
                    "doc_id": doc_ids[i],
                    "doc": docs[i],
                }
                for i in range(seq_length)
            ],
        )

class MilvusBgeM3Retriever:
    def __init__(self, milvus_client, collection_name, dim=1024):
        # Initialize the retriever with a Milvus client, collection name, and dimensionality of the vector embeddings.
        # If the collection exists, load it.
        self.collection_name = collection_name
        self.client = milvus_client
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim
    
    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

        # Specify the data schema for the new Collection
        fields = [
            # Use auto generated id as primary key
            FieldSchema(
                name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
            ),
            # Store the original text to retrieve based on semantically distance
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            # Milvus now supports both sparse and dense vectors,
            # we can store each in a separate field to conduct hybrid search on both vectors
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields)
        self.client.create_collection(
            collection_name=self.collection_name, schema=schema, consistency_level="Strong"
        )

    def create_index(self):
        # Create an index on the vector field to enable fast similarity search.
        # Releases and drops any existing index before creating a new one with specified parameters.
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="sparse_vector"
        )
        self.client.drop_index(
            collection_name=self.collection_name, index_name="dense_vector"
        )

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",  
            metric_type="IP",
        )
        
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )
    
    def insert(self, data):
        self.client.insert(
            self.collection_name,
            [
                {
                    "sparse_vector": data["sparse_vector"][i],
                    "dense_vector": data["dense_vector"][i],
                    "text": data["text"][i],
                    "doc_id": data["doc_id"][i],
                } 
                for i in range(len(data["doc_id"]))
            ]
        )

    def dense_search(self, query_dense_embedding, topk=5):
        search_params = {"metric_type": "IP", "params": {}}
        res = self.client.search(
            self.collection_name,
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=topk,
            output_fields=["text", "doc_id"],
            search_params=search_params,
        )[0]
        return [{"text": hit.get("text"), "doc_id": hit.get("doc_id")} for hit in res]


    def sparse_search(self, query_sparse_embedding, topk=5):
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        res = self.client.search(
            self.collection_name,
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=topk,
            output_fields=["text", "doc_id"],
            search_params=search_params,
        )[0]
        return [{"text": hit.get("text"), "doc_id": hit.get("doc_id")} for hit in res]


    def hybrid_search(
        self,
        query_dense_embedding,
        query_sparse_embedding,
        sparse_weight=1.0,
        dense_weight=1.0,
        topk=5,
    ):
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=topk
        )
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=topk
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.client.hybrid_search(
            [sparse_req, dense_req], rerank=rerank, limit=topk, output_fields=["text", "doc_id"]
        )[0]
        return [{"text": hit.get("text"), "doc_id": hit.get("doc_id")} for hit in res]

