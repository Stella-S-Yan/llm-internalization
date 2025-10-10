"""
Reuse the fixed_grain_dataset.py's result which is used in TIGER experiment. 
Map reivewer_ids to sequential ids for fine tunning experiment
"""

from utils import bagz_utils
import config

def main():
    records = bagz_utils.read_record(config.USER_SID_SEQUENCE)
    # Create mapping from reviewerID to sequential UID
    reviewer_map = {rid: i for i, rid in enumerate({r['reviewerID'] for r in records})}

    # Add UID field as "UID_xxxx" to each record
    new_records = [
        {**r, 'UID': f"UID_{reviewer_map[r['reviewerID']]}"} for r in records
    ]

    # Save updated data
    bagz_utils.save_record(new_records, config.USER_UID_SID_SEQUENCE)

if __name__ == "__main__":
    main()

