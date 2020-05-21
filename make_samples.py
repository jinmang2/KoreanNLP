import torch
import torch.nn as nn

torch.manual_seed(42)       # CPU Seed 고정
torch.cuda.manual_seed(42)  # GPU Seed 고정

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
x = x.to(device)
print(x)

# tensor([[ 0, 24, 24, 29,  4,  4, 30,  1,  2,  2,  2,  2,  2,  2,  2],
#         [ 0, 24, 20,  3, 21, 24,  1,  2,  2,  2,  2,  2,  2,  2,  2],
#         [ 0, 12,  4, 14,  5, 17,  8, 30,  9, 28, 19,  7,  1,  2,  2],
#         [ 0, 24, 31, 23, 31,  8, 16,  8,  3, 23, 24,  1,  2,  2,  2],
#         [ 0, 15, 22, 29, 13, 24, 25, 23, 10, 12, 27, 30,  1,  2,  2]],
#        device='cuda:0')
