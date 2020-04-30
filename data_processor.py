import numpy as np
import torch

from transformer import positional_encoding


class DataProcessor:
    def __init__(self, X, y, with_positional_encodings=True):
        self.X = X
        self.y = y
        self.with_positional_encodings = with_positional_encodings
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
            (self.dataset_size, self.max_encoder_sequence_length, self.vocabulary_size),
            dtype=torch.float32,
        )
        decoder_input = torch.zeros(
            (self.dataset_size, self.max_decoder_sequence_length, self.vocabulary_size),
            dtype=torch.float32,
        )
        target = torch.zeros(
            (self.dataset_size, self.max_decoder_sequence_length, self.vocabulary_size),
            dtype=torch.float32,
        )

        for i, (X_i, y_i) in enumerate(zip(self.X, self.y)):
            for t, char in enumerate(X_i):
                encoder_input[i, t, self.char_index[char]] = 1.0
            for t, char in enumerate(y_i):
                # todo: impression decoder_input has all tokens
                # todo: from GO to EOS see if that is ok
                decoder_input[i, t, self.char_index[char]] = 1.0
                if t > 0:
                    target[i, t - 1, self.char_index[char]] = 1.0

        encoder_input = encoder_input.refine_names("batch", "time", "word_dim")
        decoder_input = decoder_input.refine_names("batch", "time", "word_dim")
        target = target.refine_names("batch", "time", "dec_vocabulary")

        if self.with_positional_encodings:
            max_nb_steps = max(encoder_input.size("word_dim"), decoder_input.size("word_dim"))
            position_encodings = torch.tensor(
                [
                    positional_encoding(t, encoder_input.size("word_dim"))
                    for t in range(max_nb_steps)
                ],
                dtype=torch.float32,
            ).refine_names("time", "word_dim")
            encoder_input += position_encodings[: encoder_input.size("time")]
            decoder_input += position_encodings[: decoder_input.size("time")]

        p_val = 0.25
        size_val = int(p_val * self.dataset_size)
        idxs = np.arange(self.dataset_size)
        np.random.shuffle(idxs)
        idxs_tr = idxs[:-size_val]
        idxs_val = idxs[-size_val:]
        (
            self.X_tr,
            self.X_val,
            self.y_tr,
            self.y_val,
            self.encoder_input_tr,
            self.encoder_input_val,
            self.decoder_input_tr,
            self.decoder_input_val,
            self.target_tr,
            self.target_val,
        ) = (
            np.array(self.X)[idxs_tr],
            np.array(self.X)[idxs_val],
            np.array(self.y)[idxs_tr],
            np.array(self.y)[idxs_val],
            encoder_input.rename(None)[idxs_tr].refine_names(*encoder_input.names),
            encoder_input.rename(None)[idxs_val].refine_names(*encoder_input.names),
            decoder_input.rename(None)[idxs_tr].refine_names(*decoder_input.names),
            decoder_input.rename(None)[idxs_val].refine_names(*decoder_input.names),
            target.rename(None)[idxs_tr].refine_names(*target.names),
            target.rename(None)[idxs_val].refine_names(*target.names),
        )
