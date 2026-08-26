# vector_store.py

import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# Loads variables from a .env file into the environment.
#
# Example .env:
#
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# CHROMA_PATH=./chroma_db
# COLLECTION_NAME=documents
#
# After load_dotenv(), we can access them using os.getenv().
load_dotenv()


# Name of the embedding model.
#
# os.getenv("EMBEDDING_MODEL", default)
# means:
#
#   1. Look for EMBEDDING_MODEL in the environment.
#   2. If it exists -> use its value.
#   3. If it doesn't exist -> use the default value.
#
# "sentence-transformers/all-MiniLM-L6-v2" is a pretrained
# SentenceTransformer model that converts text into numerical vectors.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# Directory where ChromaDB will permanently store its data.
#
# "./chroma_db" means:
#
#     current_project/
#         └── chroma_db/
#
# PersistentClient means the database survives after the Python
# program stops.
CHROMA_PATH = Path(
    os.getenv(
        "CHROMA_PATH",
        "./chroma_db",
    )
)


# Name of the ChromaDB collection.
#
# Think of a collection as a container for:
#
#     ID
#     embedding
#     original text
#     metadata
#
# Example:
#
#     ChromaDB
#         └── documents
#               ├── chunk-1
#               ├── chunk-2
#               └── chunk-3
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "documents",
)


# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStore:
    """
    Handles:
        1. Loading the embedding model.
        2. Connecting to ChromaDB.
        3. Storing document embeddings.
        4. Searching for similar chunks.
    """

    # ------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        chroma_path: str | Path = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """
        Initialize the vector store.

        Why use __init__?

        The embedding model and ChromaDB connection are expensive/reusable
        resources. We want to create them ONCE and reuse them for:

            store.add_chunks(...)
            store.search(...)
            store.search(...)
            store.search(...)

        Instead of loading the model again for every operation.

        The parameters have defaults, so this works:

            store = VectorStore()

        But they can also be overridden:

            store = VectorStore(
                model_name="another-model",
                collection_name="books",
            )
        """

        # Store the model name on the object.
        #
        # "self" represents this particular VectorStore instance.
        #
        # After:
        #
        #     store = VectorStore()
        #
        # we can access:
        #
        #     store.model_name
        self.model_name = model_name


        # Convert the path into a pathlib.Path object.
        #
        # This allows us to work with filesystem paths more conveniently.
        #
        # Both of these are accepted:
        #
        #     "./chroma_db"
        #     Path("./chroma_db")
        #
        # Path(...) makes sure we consistently work with a Path object.
        self.chroma_path = Path(chroma_path)


        # Store the collection name.
        #
        # Example:
        #
        #     self.collection_name == "documents"
        self.collection_name = collection_name


        # --------------------------------------------------------------------
        # LOAD EMBEDDING MODEL
        # --------------------------------------------------------------------

        # SentenceTransformer converts text into numerical vectors.
        #
        # Example:
        #
        #     "Python is easy"
        #
        # becomes something conceptually like:
        #
        #     [0.12, -0.43, 0.72, ...]
        #
        # This numerical vector is called an EMBEDDING.
        #
        # We load the model ONCE here and reuse it.
        self.model = SentenceTransformer(model_name)


        # --------------------------------------------------------------------
        # CONNECT TO CHROMADB
        # --------------------------------------------------------------------

        # PersistentClient creates/connects to a persistent ChromaDB database.
        #
        # "persistent" means the data is saved to disk instead of existing
        # only in memory.
        #
        # Example:
        #
        #     project/
        #         ├── main.py
        #         └── chroma_db/
        #
        # str(...) converts Path into a normal string path because that is
        # what we pass to ChromaDB here.
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path)
        )


        # --------------------------------------------------------------------
        # GET OR CREATE COLLECTION
        # --------------------------------------------------------------------

        # ChromaDB stores vectors inside collections.
        #
        # get_or_create_collection() means:
        #
        #     If the collection already exists:
        #         -> get it
        #
        #     If it doesn't exist:
        #         -> create it
        #
        # This saves us from manually checking whether the collection exists.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,

            # HNSW is the approximate nearest-neighbor indexing algorithm
            # used by Chroma for efficient vector search.
            #
            # "space" means the distance metric used to compare vectors.
            #
            # "cosine" means:
            #
            #     Compare embeddings using cosine distance.
            #
            # In simple terms:
            #
            #     smaller cosine distance
            #             =
            #     more similar vectors
            #
            # This is a common choice for text embeddings.
            metadata={
                "hnsw:space": "cosine",
            },
        )


    # =========================================================================
    # ADD CHUNKS
    # =========================================================================

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Convert document chunks into embeddings and store them in ChromaDB.

        Expected chunk structure:

            {
                "document_id": "abc123",
                "chunk_id": "abc123-p1-c0",
                "chunk_index": 0,
                "page": 1,
                "source": "book.pdf",
                "text": "Some document text..."
            }
        """

        # If there are no chunks, there is nothing to store.
        #
        # Returning early avoids unnecessary processing.
        if not chunks:
            return


        # --------------------------------------------------------------------
        # EXTRACT TEXT
        # --------------------------------------------------------------------

        # Our chunks are dictionaries.
        #
        # chunk["text"] means:
        #
        #     "Give me the value stored under the 'text' key."
        #
        # Example:
        #
        #     chunk = {
        #         "text": "Python is easy"
        #     }
        #
        #     chunk["text"]
        #
        # gives:
        #
        #     "Python is easy"
        texts = [
            chunk["text"]
            for chunk in chunks
        ]


        # --------------------------------------------------------------------
        # CREATE EMBEDDINGS
        # --------------------------------------------------------------------

        embeddings = self.model.encode(
            texts,

            # Normalize every embedding so its L2 length becomes 1.
            #
            # This is particularly useful when using cosine-based comparison.
            #
            # Conceptually:
            #
            #     original vector
            #            ↓
            #     normalization
            #            ↓
            #     unit-length vector
            #
            # Since our Chroma collection uses cosine distance, this is
            # a sensible configuration.
            normalize_embeddings=True,

            # Prevent SentenceTransformer from printing a progress bar.
            #
            # True would show something like:
            #
            #     Batches: 100%|████████| ...
            #
            # False keeps this class quiet.
            show_progress_bar=False,
        )

        # SentenceTransformer normally returns a NumPy array.
        #
        # .tolist() converts it into normal Python lists.
        #
        # Example:
        #
        #     NumPy array
        #         ↓
        #     .tolist()
        #         ↓
        #     Python list
        #
        # ChromaDB accepts these lists as embeddings.
        embeddings = embeddings.tolist()


        # --------------------------------------------------------------------
        # EXTRACT CHUNK IDS
        # --------------------------------------------------------------------

        # Every chunk needs a unique ID.
        #
        # Example:
        #
        #     abc123-p1-c0
        #     abc123-p1-c1
        #     abc123-p2-c0
        #
        # These IDs allow ChromaDB to identify individual chunks.
        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]


        # --------------------------------------------------------------------
        # CREATE METADATA
        # --------------------------------------------------------------------

        # Metadata is additional information stored alongside each embedding.
        #
        # It is NOT the embedding itself.
        #
        # Example metadata:
        #
        #     {
        #         "document_id": "abc123",
        #         "page": 5,
        #         "source": "book.pdf",
        #         "chunk_index": 3
        #     }
        metadatas = [
            {
                "document_id": chunk["document_id"],
                "page": chunk["page"],
                "source": chunk["source"],

                # dict.get(key, default)
                #
                # means:
                #
                #     If "chunk_index" exists:
                #         use its value.
                #
                #     Otherwise:
                #         use 0.
                #
                # This prevents a KeyError if older chunks don't contain
                # "chunk_index".
                "chunk_index": chunk.get("chunk_index", 0),
            }
            for chunk in chunks
        ]


        # --------------------------------------------------------------------
        # STORE EVERYTHING IN CHROMADB
        # --------------------------------------------------------------------

        # upsert = UPDATE + INSERT
        #
        # If the ID already exists:
        #
        #     -> update the existing record
        #
        # If the ID does not exist:
        #
        #     -> insert a new record
        #
        # This is useful when ingesting the same document again.
        #
        # Chroma stores these pieces together:
        #
        #     ID
        #       +
        #     embedding
        #       +
        #     original text
        #       +
        #     metadata
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )


    # =========================================================================
    # SEARCH
    # =========================================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search for the most semantically similar chunks.

        Example:

            results = store.search(
                "What is the vacation policy?",
                top_k=5,
            )
        """

        # Remove whitespace from the beginning and end of the query.
        #
        # Example:
        #
        #     "   What is Python?   "
        #
        # becomes:
        #
        #     "What is Python?"
        query = query.strip()


        # Don't search for an empty query.
        if not query:
            return []


        # top_k tells Chroma how many results we want.
        #
        # top_k=5
        #
        # means:
        #
        #     "Return the 5 nearest chunks."
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")


        # Convert the user's query into an embedding.
        #
        # IMPORTANT:
        #
        # We use the SAME embedding model used when storing documents.
        #
        # Document:
        #
        #     text -> embedding model -> vector
        #
        # Query:
        #
        #     query -> same model -> vector
        #
        # Then Chroma compares the vectors.
        query_embedding = self.model.encode(
            query,

            # Use the same normalization configuration used for documents.
            normalize_embeddings=True,

            show_progress_bar=False,
        ).tolist()


        # --------------------------------------------------------------------
        # QUERY CHROMADB
        # --------------------------------------------------------------------

        # Chroma searches the collection for vectors nearest to our
        # query embedding.
        #
        # query_embeddings expects a LIST of embeddings because Chroma
        # supports querying with multiple embeddings.
        #
        # We only have one query, so:
        #
        #     [query_embedding]
        #
        # is a list containing one embedding.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )


        # Chroma returns a nested dictionary structure.
        #
        # _format_results() converts that raw structure into a simpler
        # list of dictionaries that is easier for the rest of our
        # application to use.
        return self._format_results(results)


    # =========================================================================
    # RESULT FORMATTING
    # =========================================================================

    @staticmethod
    def _format_results(
        results: dict[str, Any],
    ) -> list[dict]:
        """
        Convert ChromaDB's nested result format into a simple list.

        Chroma can return something conceptually like:

            {
                "ids": [
                    ["chunk1", "chunk2"]
                ],

                "documents": [
                    ["text1", "text2"]
                ],

                "metadatas": [
                    [
                        {"page": 1},
                        {"page": 2}
                    ]
                ],

                "distances": [
                    [0.12, 0.25]
                ]
            }

        We convert it into:

            [
                {
                    "id": "chunk1",
                    "text": "text1",
                    "metadata": {"page": 1},
                    "distance": 0.12
                },

                {
                    "id": "chunk2",
                    "text": "text2",
                    "metadata": {"page": 2},
                    "distance": 0.25
                }
            ]
        """


        # --------------------------------------------------------------------
        # EXTRACT RESULTS
        # --------------------------------------------------------------------

        # dict.get(key, default)
        #
        # means:
        #
        #     Get the value for "documents".
        #
        #     If the key doesn't exist, use [[]].
        #
        # Why [[]]?
        #
        # Because Chroma's result is nested:
        #
        #     [
        #         [documents...]
        #     ]
        #
        # The [0] below extracts the results for the FIRST query.
        documents = results.get(
            "documents",
            [[]],
        )[0]


        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]


        distances = results.get(
            "distances",
            [[]],
        )[0]


        ids = results.get(
            "ids",
            [[]],
        )[0]


        # --------------------------------------------------------------------
        # BUILD CLEAN RESULTS
        # --------------------------------------------------------------------

        # zip() combines corresponding items.
        #
        # Example:
        #
        #     ids        = ["c1", "c2"]
        #     documents  = ["text1", "text2"]
        #     distances  = [0.1, 0.2]
        #
        # zip() gives:
        #
        #     ("c1", "text1", 0.1)
        #     ("c2", "text2", 0.2)
        #
        # We then create a clean dictionary for each result.
        return [
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
            )
        ]