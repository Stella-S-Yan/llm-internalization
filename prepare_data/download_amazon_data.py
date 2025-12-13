"""Downloads and unzips the Amazon Beauty review dataset."""

import os
import urllib.request
import gzip
import shutil
import config

def download_amazon_data():
  """Downloads and unzips the Amazon Beauty review dataset.

  This function fetches two gzipped JSON files from the Stanford SNAP website:
  - `meta_Beauty.json.gz`: Contains product metadata.
  - `reviews_Beauty_5.json.gz`: Contains product reviews. We use 5-core data by
  following the conventions in recsys papers.

  The files are downloaded to a subdirectory within `config.DATA_DIR`
  (specifically, `config.DATA_DIR/config.DATA_SOURCE`). The gzipped files
  are then unzipped, and the resulting `.json` files are saved in the same
  directory.
  """

  urls = [
      # "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
      # "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",

      "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
      # "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz"
  ]

  outdir = os.path.join(config.DATA_DIR, config.DATA_SOURCE)
  print(outdir)
  os.makedirs(outdir, exist_ok=True)

  for url in urls:
    filename = os.path.basename(url)
    gz_path = os.path.join(outdir, filename)
    out_path = os.path.join(outdir, filename.replace(".gz", ""))

    print(f"Downloading {url} into {outdir}...")
    urllib.request.urlretrieve(url, gz_path)

    print(f"Unzipping {filename}...")
    with gzip.open(gz_path, "rb") as f_in:
      with open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"Done! File is at {out_path}")

if __name__== "__main__":
    download_amazon_data()