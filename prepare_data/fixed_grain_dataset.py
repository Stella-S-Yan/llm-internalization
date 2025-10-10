"""
Pregenerate all possible subsequences and create a fixed training dataset
"""

import numpy as np
import config
import hashlib
from utils import bagz_utils
from utils import tokenizer_utils



def format_sid(seq):
    prefixes = ["A", "B", "C", "D"]
    return [f"{p}{n}" for p, n in zip(prefixes, seq)]


class GenFixedData():
    def __init__(self):
        self.train_eval_data = []
        self.train_data = []
        self.eval_data = []
        self.test_data = []
        self.tokenizer = tokenizer_utils.load_tokenizer()
        self.max_encoder_seq_len = config.MAX_HISTORY_LEN * 4 + 1
        self.max_decoder_seq_len = 1 * 4   

    def gen_data(self):
        records = bagz_utils.read_record(config.USER_UID_SID_SEQUENCE)
        
        for record in records:
            self._process_one_record(record)

        self._save_data()


    def _process_one_record(self, record):
        uid = record["UID"]
        sid_seq = record["sid_seq"]

        # Test
        sid_seq = sid_seq[-config.MAX_HISTORY_LEN:]
        inp_seq_d, tgt_seq_d = self._make_data_point(sid_seq, uid)
        self.test_data.append( {
                "input_ids": inp_seq_d,
                "target_ids": tgt_seq_d,
                }
        )

        # Eval
        sid_seq = record["sid_seq"][:-1]
        sid_seq = sid_seq[-config.MAX_HISTORY_LEN:]
        inp_seq_d, tgt_seq_d = self._make_data_point(sid_seq, uid)
        self.eval_data.append( {
                "input_ids": inp_seq_d,
                "target_ids": tgt_seq_d,
                }
        )

        # Train
        sid_seq = record["sid_seq"][:-2]
        sub_seqs = self._get_subsequence(sid_seq)
        sub_sequences = self._gen_train_data_point(sub_seqs, uid)
        self.train_data.extend(sub_sequences)

        # Train + eval
        sid_seq = record["sid_seq"][:-1]
        sub_seqs = self._get_subsequence(sid_seq)
        sub_sequences = self._gen_train_data_point(sub_seqs, uid)
        self.train_eval_data.extend(sub_sequences)

        return


    def _save_data(self):
        bagz_utils.save_record(self.train_eval_data, config.TRAIN_EVAL_DATA)
        bagz_utils.save_record(self.train_data, config.TRAIN_DATA)
        bagz_utils.save_record(self.eval_data, config.EVAL_DATA)
        bagz_utils.save_record(self.test_data, config.TEST_DATA)
    

    def _gen_train_data_point(self, sub_seqs, uid):
        res = []
        for seq in sub_seqs:
            inp_seq_d, tgt_seq_d = self._make_data_point(seq, uid)
            res.append(
                {
                    "input_ids": inp_seq_d, 
                    "target_ids": tgt_seq_d, 
                }
            )
        return res


    def _get_subsequence(self, seq):
        subsequences = []
        n = len(seq)
        for start in range(n):
            # end index goes from start+2 to start+max_seq_len (inclusive), but not beyond sequence length
            for end in range(start + 2, min(n, start + config.MAX_HISTORY_LEN) + 1):
                subsequences.append(seq[start:end])
        return subsequences


    def _make_data_point(self, seq, hashed_uid):
        inp_seq = seq[:-1]
        tgt_seq = seq[-1]

        inp_seq = [format_sid(seq) for seq in inp_seq]
        tgt_seq = format_sid(tgt_seq)

        inp_seq = [hashed_uid] + inp_seq
        inp_seq_str = ' '.join(
            item if isinstance(item, str) else ' '.join(item)
            for item in inp_seq
        )
        tgt_seq_str = ' '.join(tgt_seq)
        # inp_seq, tgt_seq = self._tokenize_pad_sequence(inp_seq_str, tgt_seq_str)

        return inp_seq_str, tgt_seq_str


def generate_fixed_split_data():
    gf = GenFixedData()
    gf.gen_data()


if __name__ == "__main__":
    generate_fixed_split_data()