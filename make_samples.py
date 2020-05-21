import torch
import torch.nn as nn

SEQ_LEN = 15
BATCH_SIZE = 5
INPUT_DIM = 30
ENC_EMB_DIM = DEC_EMB_DIM = 32
ENC_HID_DIM = DEC_HID_DIM = 64
ENC_DROPOUT_P = DEC_DROPOUT_P = 0.2

BOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
SPECIAL_TOKENS = [
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
]
INPUT_DIM += len(SPECIAL_TOKENS)


def pad_sequence(seq, max_seq_len, special_tokens):
    bos, eos, pad = special_tokens
    assert seq.dim() == 1
    assert isinstance(pad, int)
    seq_len = len(seq)
    assert max_seq_len >= seq_len
    seq[0] = bos
    seq[-1] = eos
    padded_seq = torch.cat(
        [seq, torch.ones(max_seq_len-seq_len).long() * pad],
        dim=0
    )
    return padded_seq.unsqueeze(0)


def gen_example_sequence(special_tokens, 
                         num_tokens,
                         max_seq_len):
    size = (torch.randint(int(max_seq_len//2), max_seq_len-1, (1,)).item(),)
    seq = torch.randint(len(special_tokens), num_tokens, size=size)
    padded_seq = pad_sequence(seq, max_seq_len, special_tokens)
    return padded_seq


def gen_batch_sequences(special_tokens,
                        num_tokens,
                        max_seq_len,
                        batch_size):
    x = torch.cat(
        [gen_example_sequence(special_tokens, num_tokens, max_seq_len)
        for _ in range(batch_size)],
        dim=0
    )
    return x


x = gen_batch_sequences(SPECIAL_TOKENS, INPUT_DIM-1, SEQ_LEN, BATCH_SIZE)
print(x)

# tensor([[ 0, 10, 13,  6,  7,  7,  5, 20, 15,  1,  2,  2,  2,  2,  2],
#         [ 0, 26, 12, 21, 29, 14, 29,  1,  2,  2,  2,  2,  2,  2,  2],
#         [ 0, 12, 25, 17,  3, 19, 30, 31,  4,  1,  2,  2,  2,  2,  2],
#         [ 0, 31,  4, 20, 17,  5, 28, 14, 17,  1,  2,  2,  2,  2,  2],
#         [ 0, 10, 16, 30, 29, 16,  7, 12, 19,  1,  2,  2,  2,  2,  2]])
