import bagz
import io
import pandas as pd
import json
import pickle

def save_parquet(df: pd.DataFrame, target_file: str):
    """Save a DataFrame to a Bagz file as a single Parquet record."""
    with io.BytesIO() as buffer:
        df.to_parquet(buffer, index=False, engine="pyarrow")
        buffer.seek(0)
        with bagz.Writer(target_file) as writer:
            writer.write(buffer.getvalue())


def read_parquet(bagz_file: str) -> pd.DataFrame:
    """Read a Bagz file containing a single Parquet DataFrame. """
    reader = bagz.Reader(bagz_file)
    buffer = io.BytesIO(reader[0])  # one record
    df = pd.read_parquet(buffer)
    return df


def save_object(data, target_file: str):
    """
    Save any Python dictionary (including tuple keys) to Bagz.
    """
    with bagz.Writer(target_file) as writer:
        writer.write(pickle.dumps(data))

def read_object(bagz_file: str):
    """
    Load a dictionary from Bagz that was saved with pickle.
    """
    reader = bagz.Reader(bagz_file)
    return pickle.loads(reader[0])


def save_record(records, target_file):
    with bagz.Writer(target_file) as writer:
        for record in records:
            writer.write(json.dumps(record).encode("utf-8"))  # convert string to bytes


def read_record(bagz_file):
    """Read string records from a Bagz file."""
    reader = bagz.Reader(bagz_file)
    records = [json.loads(record.decode("utf-8")) for record in reader]
    return records