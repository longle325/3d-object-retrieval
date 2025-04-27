from pymilvus import (
    connections,
    Collection,
)
import os
import argparse
from tqdm import tqdm
from model import SIG_LIP, VIT_BIG
import csv
import shutil

def retrieve(text=None, img_path=None, collection_name=None, model=None):
    embedding = None
    try:
        if text:
            embedding = model.text_embedding(text)
        elif img_path:
            embedding = model.image_embedding(img_path)
        else:
            raise ValueError("Either text or image path must be provided.")
    except Exception as e:
        print(f"Error predicting ---{text}---: {e}")
        return [], []

    collection = Collection(name=collection_name)
    collection.load()

    search_params = {
        "metric_type": "COSINE",
    }

    num_results = 200

    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param=search_params,
        limit=num_results,
        output_fields=["object_id", "img_path"]
    )[0]

    found_object_ids = {}
    cnt = 0

    img_path_list = []
    object_ids_list = []

    for result in results:
        object_id = result.get("object_id")
        if object_id not in found_object_ids:
            img_path_list.append(result.get("img_path"))
            object_ids_list.append(object_id)
            found_object_ids[object_id] = 1
            cnt += 1
            if cnt == 10:  # Limit to 10 objects
                break

    return object_ids_list, img_path_list

def infer(query_path, output_csv_path, visualize_path, embed_model=None, chosen_collection=None):
    os.makedirs(visualize_path, exist_ok=True)

    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ["scene_id"] + [f"obj_id{i+1}" for i in range(10)]
        csv_writer.writerow(header)

        for subdir in tqdm(os.listdir(query_path)):
            subdir_path = os.path.join(query_path, subdir)

            if os.path.isdir(subdir_path):
                try:
                    file_name = "query.txt"
                    file_path = os.path.join(subdir_path, file_name)
                    scene_id = os.path.basename(subdir_path)

                    if os.path.isfile(file_path):
                        with open(file_path, 'r') as query_file:
                            query_text = query_file.readline().strip()

                        print(f"Processing scene: {scene_id}")
                        print(f"Query text: {query_text}")

                        object_ids_list, img_path_list = retrieve(
                            text=query_text,
                            collection_name=chosen_collection,
                            model=embed_model
                        )

                        while len(object_ids_list) < 10:
                            object_ids_list.append("")

                        output_visualize_dir = os.path.join(visualize_path, scene_id)
                        os.makedirs(output_visualize_dir, exist_ok=True)

                        # Copy scene 020.png first
                        scene_img_path = os.path.join(subdir_path, "020.png")
                        if os.path.exists(scene_img_path):
                            dest_scene_path = os.path.join(output_visualize_dir, "0.png")
                            shutil.copy(scene_img_path, dest_scene_path)
                        else:
                            print(f"Warning: Scene image not found for {scene_id}")

                        # Save query text
                        query_output_file = os.path.join(output_visualize_dir, "0.txt")
                        with open(query_output_file, 'w') as f:
                            f.write(query_text)

                        # Copy retrieved images
                        for idx, img_path in enumerate(img_path_list, start=1):
                            if os.path.exists(img_path):
                                dest_path = os.path.join(output_visualize_dir, f"{idx}.png")
                                shutil.copy(img_path, dest_path)
                            else:
                                print(f"Warning: Retrieved image not found at {img_path}")

                        row = [scene_id] + object_ids_list
                        csv_writer.writerow(row)

                except Exception as e:
                    print(f"Error processing {subdir_path}: {e}")

def connect_milvus(uri="http://localhost:19530"):
    try:
        connections.connect(uri=uri)
        print("Successfully connected to Milvus server!")
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Base retrieval inference")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query scenes")
    parser.add_argument("--collection_name", type=str, required=True, help="Milvus collection name")
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path to save the CSV results")
    parser.add_argument("--visualize_path", type=str, required=True, help="Path to save visualizations")
    parser.add_argument("--milvus_uri", type=str, default="http://localhost:19530", help="Milvus server URI (default localhost)")
    args = parser.parse_args()

    connect_milvus(uri=args.milvus_uri)

    model = SIG_LIP()  # or VIT_BIG() if you want to switch later easily

    infer(
        query_path=args.query_path,
        output_csv_path=args.output_csv_path,
        visualize_path=args.visualize_path,
        embed_model=model,
        chosen_collection=args.collection_name
    )
