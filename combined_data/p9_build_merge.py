import os
from pathlib import Path
import argparse
import pickle

import config
from utils import bagz_utils


def build_merged(split: str, sources: list[str], output_name: str):
    merged_data = []

    print(f"Building merged dataset for split='{split}'")
    for src in sources:
        data_path = (
            config.PROCESSED_DATA_DIR
            / f"{config.DATA_SOURCE}_{src}_think_data_{split}.bagz"
        )

        print(f"  → loading {data_path}")
        records = bagz_utils.read_record(data_path)
        merged_data.extend(records)

    print("Sorting for deterministic DDP order...")
    merged_data.sort(key=lambda x: str(x["prompt"]))

    out_path = config.PROCESSED_DATA_DIR / output_name
    
    with open(out_path, "wb") as f:
        pickle.dump(merged_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(merged_data)} records → {out_path}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output_name", required=True)
    args = parser.parse_args()

    SOURCES = [
        "Automotive",
        "Baby",
        "Beauty",
        "Cell_Phones_and_Accessories",
        "Clothing_Shoes_and_Jewelry",
        "Grocery_and_Gourmet_Food",
        "Health_and_Personal_Care",
        "Home_and_Kitchen",
        "Musical_Instruments",
        "Office_Products",
        "Patio_Lawn_and_Garden",
        "Pet_Supplies",
        "Sports_and_Outdoors",
        "Tools_and_Home_Improvement",
        "Toys_and_Games",
    ]

    build_merged(
        split=args.split,
        sources=SOURCES,
        output_name=f"{config.PROCESSED_DATA_DIR}/{args.output_name}",
    )
