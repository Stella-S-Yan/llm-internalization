"""
Only do it for eval and test data. Training data are generated on the fly.
"""

import config
from utils import bagz_utils
import random



def format_sid(seq):
    prefixes = ["A", "B", "C", "D"]
    return [f"{p}{n}" for p, n in zip(prefixes, seq)]


class GenFixedData():
    def __init__(self):
        self.eval_data = []
        self.test_data = []

        self.max_len = 0

    def gen_data(self):
        records = bagz_utils.read_record(config.USER_SEQUENCE)
        
        for record in records:
            self._process_one_record(record)

        self._save_data()

        print(self.max_len)


    def _process_one_record(self, record):
        uid = record["id"]
        reviewerID = record["reviewerID"]
        sid_seq = record["sequence"]

        # Test
        sid_seq = sid_seq[-config.MAX_HISTORY_LEN:]
        input_seq_d, target_seq_d = self._make_data_point(sid_seq)
        self.test_data.append( {
                "uid": uid,
                "input": input_seq_d,
                "target": target_seq_d,
                }
        )

        # Eval
        sid_seq = record["sequence"][:-1]
        sid_seq = sid_seq[-config.MAX_HISTORY_LEN+1:]
        input_seq_d, target_seq_d = self._make_data_point(sid_seq)
        self.eval_data.append( {
                "uid": uid,
                "input": input_seq_d,
                "target": target_seq_d,
                }
        )

        return


    def _save_data(self):
        rng = random.Random(411)
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

        input_seq_str = ' '.join(
            item if isinstance(item, str) else ' '.join(item)
            for item in input
        )

        return input_seq_str, target


def generate_fixed_split_data():
    gf = GenFixedData()
    gf.gen_data()


if __name__ == "__main__":
    generate_fixed_split_data()