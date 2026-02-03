"""
Pregenerate all possible subsequences and create a fixed training dataset
"""

import config
from utils import bagz_utils
import random


class GenFixedData():
    def __init__(self):
        # self.train_eval_data = []
        self.train_data = []
        self.eval_data = []
        self.test_data = []


    def gen_data(self):
        records = bagz_utils.read_record(config.USER_SEQUENCE)
        
        for record in records:
            self._process_one_record(record)

        self._save_data()



    def _process_one_record(self, record):
        uid = record["id"]
        reviewerID = record["reviewerID"]
        sid_seq = record["sequence"]

        # Test
        sid_seq = sid_seq[-config.MAX_HISTORY_LEN-1:]
        input_seq_d, target_seq_d = self._make_data_point(sid_seq)
        self.test_data.append( {
                "uid": uid,
                "input": input_seq_d,
                "target": target_seq_d,
                }
        )

        # Eval
        sid_seq = record["sequence"][:-1]
        sid_seq = sid_seq[-config.MAX_HISTORY_LEN-1:]
        input_seq_d, target_seq_d = self._make_data_point(sid_seq)
        self.eval_data.append( {
                "uid": uid,
                "input": input_seq_d,
                "target": target_seq_d,
                }
        )

        # Train
        sid_seq = record["sequence"][:-2]
        sub_seqs = self._get_subsequence(sid_seq)
        sub_sequences = self._gen_train_data_point(sub_seqs, uid, reviewerID)
        self.train_data.extend(sub_sequences)

        # # Train + eval
        # sid_seq = record["sequence"][:-1]
        # sub_seqs = self._get_subsequence(sid_seq)
        # sub_sequences = self._gen_train_data_point(sub_seqs, uid, reviewerID)
        # self.train_eval_data.extend(sub_sequences)

        return


    def _save_data(self):
        rng = random.Random(411)
        # rng.shuffle(self.train_eval_data)
        # bagz_utils.save_record(self.train_eval_data, config.TRAIN_EVAL_DATA)
        rng.shuffle(self.train_data)
        bagz_utils.save_record(self.train_data, config.TRAIN_DATA)
        rng.shuffle(self.eval_data)
        bagz_utils.save_record(self.eval_data, config.EVAL_DATA)
        rng.shuffle(self.test_data)
        bagz_utils.save_record(self.test_data, config.TEST_DATA)
    

    def _gen_train_data_point(self, sub_seqs, uid, reviewerID):
        del reviewerID
        res = []
        for seq in sub_seqs:
            input_seq_d, target_seq_d = self._make_data_point(seq)
            if input_seq_d is not None:
                res.append(
                    {   
                        "uid": uid,
                        "input": input_seq_d, 
                        "target": target_seq_d,
                    }
                )
        return res


    def _get_subsequence(self, seq):
        subsequences = []
        n = len(seq)
        for start in range(n):
            # end index goes from start+2 to start+max_seq_len (inclusive), but not beyond sequence length
            for end in range(start + config.MIN_HISTORY_LEN, min(n, start + config.MAX_HISTORY_LEN) + 1):
                subsequences.append(seq[start:end])
        return subsequences


    def _make_data_point(self, seq):
        target = seq[-1]
        input = seq[:-1]

        try:
            input_seq_str = ' '.join(
                item if isinstance(item, str) else ' '.join(item)
                for item in input
            )
        except Exception as e:
            print(f"Error processing sequence {seq}: {e}")
            return None, None

        return input_seq_str, target


def generate_fixed_split_data():
    gf = GenFixedData()
    gf.gen_data()


if __name__ == "__main__":
    generate_fixed_split_data()