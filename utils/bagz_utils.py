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


def save_record_list(record_list: dict, target_file):
    """
    Save multiple lists of fields into a bagz file as records.

    Args:
        target_file (str): path to output bagz file
        field_lists (dict): keys are field names, values are lists of field values
                            All lists must have the same length
    """

    # Check all lists are same length
    lengths = [len(v) for v in record_list.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"All input lists must have same length, got lengths={lengths}")

    num_records = lengths[0]

    with bagz.Writer(target_file) as writer:
        for i in range(num_records):
            record = {field: record_list[field][i] for field in record_list}
            writer.write(json.dumps(record).encode("utf-8"))


