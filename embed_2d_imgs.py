from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
import os
import argparse
from tqdm import tqdm
from model import SIG_LIP

def embed(root_dir, collection_name, embed_model):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Collection '{collection_name}' already exists. Dropped it.")

    dim = 1152
    print("Embedding dimension:", dim)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="img_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    
    schema = CollectionSchema(fields, description="Collection for ROOMELSA embeddings")

    collection = Collection(name=collection_name, schema=schema)

    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "FLAT",
            "metric_type": "COSINE",
        }
    )

    collection.load()

    print("Starting embedding process...")

    for subdir in tqdm(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file != "020.jpg":
                    continue
                file_path = os.path.join(subdir_path, file)
                object_id = os.path.basename(subdir_path)

                if os.path.isfile(file_path):
                    try:
                        embedding = embed_model.image_embedding(file_path)
                        insert_data = [
                            {
                                "object_id": object_id,
                                "img_path": file_path,
                                "embedding": embedding,
                            }
                        ]
                        collection.insert(insert_data)
                    except Exception as e:
                        print(f"Error embedding {file_path}: {e}")
        
    collection.flush()
    print("Embedding finished!")

def connect_milvus() -> None:
    try:
        connections.connect(uri='http://localhost:19530')
        print("Successfully connected to the Milvus server!")
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed images and store them into Milvus.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing image folders.")
    parser.add_argument("--collection_name", type=str, required=True, help="Milvus collection name to create.")
    args = parser.parse_args()

    connect_milvus()
    model = SIG_LIP()

    embed(root_dir=args.root_dir, collection_name=args.collection_name, embed_model=model)
