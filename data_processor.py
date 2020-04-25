import operator
import numpy as np
import torch

def math_expressions_generation(n_samples=1000, n_digits=3, invert=True):
    X, Y = [], []
    math_operators = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "%": operator.mod,
    }
    for i in range(n_samples):
        a, b = np.random.randint(1, 10 ** n_digits, size=2)
        op = np.random.choice(list(math_operators.keys()))
        res = math_operators[op](a, b)
        x = "".join([str(elem) for elem in (a, op, b)])
        if invert is True:
            x = x[::-1]
        y = "{:.5f}".format(res) if isinstance(res, float) else str(res)
        X.append(x)
        Y.append(y)
    return X, Y

class DataProcessor():
    def __init__(
        self, X, y
    ):
        self.X = X
        self.y = y
        self.GO = "="
        self.EOS = "\n"
        self.dataset_size = None
        self.char_index = None
        self.index_char = None
        self.vocabulary_size = None
        self.max_encoder_sequence_length = None
        self.max_decoder_sequence_length = None
        self.encoder_input_tr = None
        self.encoder_input_val = None
        self.decoder_input_tr = None
        self.decoder_input_val = None
        self.target_tr = None
        self.target_val = None
        self._set_data_properties_attributes()
        self._construct_data_set()

    def _set_data_properties_attributes(self):
        self.y = list(map(lambda token: self.GO + token + self.EOS, self.y))
        self.dataset_size = len(self.X)
        characters = sorted(list(set("".join(self.X) + "".join(self.y))))
        self.char_index = {c: i for i, c in enumerate(characters)}
        self.index_char = {i: c for i, c in enumerate(characters)}
        self.vocabulary_size = len(self.char_index)
        self.max_encoder_sequence_length = max([len(sequence) for sequence in self.X])
        self.max_decoder_sequence_length = max([len(sequence) for sequence in self.y])
        print("Number of samples:", self.dataset_size)
        print("Max sequence length for encoding:", self.max_encoder_sequence_length)
        print("Max sequence length for decoding:", self.max_decoder_sequence_length)

    def _construct_data_set(self):
        encoder_input = torch.zeros(
            (
                self.max_encoder_sequence_length,
                self.dataset_size,
                self.vocabulary_size
            ),
            dtype=torch.float32,
        )
        decoder_input = torch.zeros(
            (
                self.max_decoder_sequence_length,
                self.dataset_size,
                self.vocabulary_size
            ),
            dtype=torch.float32,

        )
        target = torch.zeros(
            (
                self.max_decoder_sequence_length,
                self.dataset_size,
                self.vocabulary_size
            ),
            dtype=torch.float32,
        )

        for i, (X_i, y_i) in enumerate(zip(self.X, self.y)):
            for t, char in enumerate(X_i):
                encoder_input[t, i, self.char_index[char]] = 1.0
            for t, char in enumerate(y_i):
                decoder_input[t, i, self.char_index[char]] = 1.0
                if t > 0:
                    target[t - 1, i, self.char_index[char]] = 1.0

        p_val = 0.25
        size_val = int(p_val * self.dataset_size)
        idxs = np.arange(self.dataset_size)
        np.random.shuffle(idxs)
        idxs_tr = idxs[:-size_val]
        idxs_val = idxs[-size_val:]
        (
            self.encoder_input_tr,
            self.encoder_input_val,
            self.decoder_input_tr,
            self.decoder_input_val,
            self.target_tr,
            self.target_val,
        ) = (
            encoder_input[:, idxs_tr, :],
            encoder_input[:, idxs_val, :],
            decoder_input[:, idxs_tr, :],
            decoder_input[:, idxs_val, :],
            target[:, idxs_tr, :],
            target[:, idxs_val, :],
        )
        self.encoder_input_tr = self.encoder_input_tr.refine_names('time', 'batch', 'word_dim')
        self.encoder_input_val = self.encoder_input_val.refine_names('time', 'batch', 'word_dim')
        self.decoder_input_tr = self.decoder_input_tr.refine_names('time', 'batch', 'word_dim')
        self.decoder_input_val = self.decoder_input_val.refine_names('time', 'batch', 'word_dim')
        self.target_tr = self.target_tr.refine_names('time', 'batch', 'dec_vocabulary')
        self.target_val = self.target_val.refine_names('time', 'batch', 'dec_vocabulary')
