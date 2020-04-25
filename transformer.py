import numpy as np
import torch.nn as nn
from torch.nn.modules.transformer import Transformer as PytorchTransformer
import torch
#from pytorch.torch.nn.modules.transformer import Transformer as PytorchTransformerDummy


from data_processor import math_expressions_generation, DataProcessor


def positional_encoding(position, encoding_dim):
    dimensions = np.arange(encoding_dim)
    dimensions_i = dimensions // 2
    dimensions_are_even = dimensions % 2 == 0
    positional_encoding = position / (10000 ** (2 * dimensions_i / encoding_dim))
    positional_encoding[dimensions_are_even] = np.sin(
        positional_encoding[dimensions_are_even]
    )
    positional_encoding[~dimensions_are_even] = np.cos(
        positional_encoding[~dimensions_are_even]
    )
    return positional_encoding


def test_encoding_relative_positions(
    embedding_size=15, number_samples=1000, number_positions=30
):
    def create_dataset(number_samples, number_positions):
        X = []
        y = []
        for _ in range(number_samples):
            positions = np.random.randint(number_positions, size=2)
            embedding_max = positional_encoding(positions.max(), embedding_size)
            embedding_min = positional_encoding(positions.min(), embedding_size)
            X.append(np.concatenate([embedding_max, embedding_min]))
            y.append(positions.max() - positions.min())
        X = np.array(X)
        y = np.array(y)
        X_tr, X_val = X[: int(0.75 * len(X))], X[int(0.75 * len(X)) :]
        y_tr, y_val = y[: int(0.75 * len(y))], y[int(0.75 * len(y)) :]
        return X_tr, X_val, y_tr, y_val

    def create_model(embedding_size):
        import tensorflow

        model = tensorflow.keras.Sequential(
            [
                tensorflow.keras.layers.Input(shape=(2 * embedding_size)),
                tensorflow.keras.layers.Dense(16, activation="relu"),
                tensorflow.keras.layers.Dense(8, activation="relu"),
                tensorflow.keras.layers.Dense(1, activation="relu"),
            ]
        )
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.001), loss="mse")
        return model

    X_tr, X_val, y_tr, y_val = create_dataset(number_samples, number_positions)
    model = create_model(embedding_size)
    history_callback = model.fit(X_tr, y_tr, epochs=100, validation_data=(X_val, y_val))
    assert history_callback.history["val_loss"][-1] < 0.2


# multihead special case of self-attention where self attention is performed several times
# input transtated to K, Q, V is Self-Attention in general
# multi_head_repr is a special case I believe but happens in the middle of Self-Attention so a bit difficult
# to use Self-Attention as a tool


# attention in general is K, Q, V which dimensions must end with time and dim
# implement attention in general with K, Q, V and use in Self-Attention and Encoder-Decoder Attention


def attention(keys, queries, values, mask=None, temperature=None):
    assert keys.names == values.names == queries.names
    assert "time" in keys.names and "dim" in keys.names
    assert keys.size("time") == values.size("time")
    other_names = tuple(name for name in keys.names if name not in ("time", "dim"))

    Z = torch.matmul(
        queries,
        keys.rename(time="time_keys").align_to(*other_names, "dim", "time_keys"),
    )
    if temperature:
        Z /= temperature
    assert Z.names == other_names + ("time", "time_keys")

    # if mask provided then mask some positions from softmax computation
    if mask is not None:
        mask_infs = torch.zeros(mask.shape)
        mask_infs[mask] -= torch.Tensor([float("Inf")])
        Z += mask_infs
        #Z.masked_fill_(mask, float('-inf'))
    Z = torch.nn.functional.softmax(Z, "time_keys")

    Z = torch.matmul(
        Z, values.rename(time="time_keys").align_to(*other_names, "time_keys", "dim")
    )
    return Z


class MultiHead(nn.Module):
    def __init__(
        self,
        dim_input_query,
        dim_input_key,
        dim_input_value,
        dim_representation,
        d_k=64,
        num_heads=8,
    ):
        super(MultiHead, self).__init__()
        assert dim_representation // num_heads * num_heads == dim_representation
        self.dim_representation = dim_representation
        self.num_heads = num_heads
        #self.d_k = d_k
        self.d_k = float(dim_representation)
        # scaling = float(head_dim) ** -0.5
        #self.Q = nn.Linear(dim_input_query, dim_representation * num_heads, bias=True)
        #self.K = nn.Linear(dim_input_key, dim_representation * num_heads, bias=True)
        #self.V = nn.Linear(dim_input_value, dim_representation * num_heads, bias=True)
        self.Q = nn.Linear(dim_input_query, dim_representation, bias=True)
        self.K = nn.Linear(dim_input_key, dim_representation, bias=True)
        self.V = nn.Linear(dim_input_value, dim_representation, bias=True)
        #self.W_o = nn.Linear(
        #    num_heads * dim_representation, dim_input_query, bias=False #bias=True
        #)
        self.W_o = nn.Linear(
            dim_representation, dim_input_query, bias=True  # bias=True
        )
        #from torch.nn.modules.activation import MultiheadAttention
        #self.multi_head = MultiheadAttention(dim_input_query, num_heads, dropout=0.)

    def __init__tmp(
        self,
        dim_input_query,
        dim_input_key,
        dim_input_value,
        dim_representation,
        d_k=64,
        num_heads=8,
    ):
        super(MultiHead, self).__init__()
        assert dim_representation // num_heads * num_heads == dim_representation
        from pytorch.torch.nn.modules.activation import MultiheadAttention
        self.multi_head = MultiheadAttention(dim_input_query, num_heads, dropout=0.)

    def forward_tmp(self, input_query, input_key, input_value, mask=None):
        multi = self.multi_head(input_query, input_key, input_value, attn_mask=mask)[0]
        return multi


    def forward(self, input_query, input_key, input_value, mask=None):
        assert "batch" in input_query.names and "time" in input_query.names
        assert "batch" in input_key.names and "time" in input_key.names
        assert "batch" in input_value.names and "time" in input_value.names

        def multi_head_repr(lin_layer, input, num_heads, dim_representation):

            multi_head_representation = lin_layer(input).refine_names(
                ..., "dim_times_n_head"
            )
            #multi_head_representation = multi_head_representation.unflatten(
            #    "dim_times_n_head", [("head", num_heads), ("dim", dim_representation)]
            #)
            multi_head_representation = multi_head_representation.unflatten(
                "dim_times_n_head", [("head", num_heads), ("dim", dim_representation // num_heads)]
            )
            multi_head_representation = multi_head_representation.align_to(
                "head", "batch", "time", "dim"
            )
            return multi_head_representation

        multi_head_q = multi_head_repr(
            self.Q, input_query, self.num_heads, self.dim_representation
        )
        multi_head_k = multi_head_repr(
            self.K, input_key, self.num_heads, self.dim_representation
        )
        multi_head_v = multi_head_repr(
            self.V, input_value, self.num_heads, self.dim_representation
        )

        assert (
            multi_head_k.names
            == multi_head_q.names
            == multi_head_v.names
            == ("head", "batch", "time", "dim")
        )

        Z = attention(
            multi_head_k,
            multi_head_q,
            multi_head_v,
            mask=mask,
            temperature=np.sqrt(self.d_k),
        )
        Z = Z.align_to("batch", "time", "head", "dim").flatten(
            ["head", "dim"], "dim_all_heads"
        )
        Z = self.W_o(Z).refine_names("batch", "time", "word_dim")
        return Z


class LinearLayer(nn.Module):
    def __init__(self, dim_word):
        super(LinearLayer, self).__init__()
        #self.alpha = torch.nn.Parameter(
        #    data=torch.rand(1, dim_word).refine_names("time", "word_dim"),
        #    requires_grad=True,
        #)
        ## they perform special init with ones and zeros for w and b
        #torch.nn.init.xavier_normal_(self.alpha)
        #self.beta = torch.nn.Parameter(
        #    data=torch.rand(1, dim_word).refine_names("time", "word_dim"),
        #    requires_grad=True,
        #)
        #torch.nn.init.xavier_normal_(self.beta)
        from torch.nn.modules.normalization import LayerNorm
        self.layer_norm = LayerNorm(dim_word)

    def forward(self, Z):
        #mean = Z.mean("word_dim")
        #std = Z.std("word_dim")
        #mean = (
        #    mean.rename(None)
        #    .reshape(mean.shape + (1,))
        #    .refine_names(*mean.names, "word_dim")
        #)
        #std = (
        #    std.rename(None)
        #    .reshape(std.shape + (1,))
        #    .refine_names(*std.names, "word_dim")
        #)
        #assert std.names == mean.names == Z.names == ("batch", "time", "word_dim")
        #Z = (Z - mean) / std
        #return Z * self.alpha + self.beta
        norm = self.layer_norm(Z.rename(None)).refine_names(*Z.names)
        return norm


class FeedForward(nn.Module):
    def __init__(self, word_dim, dim_feedforward=2048):
        super(FeedForward, self).__init__()
        # todo: check this in paper input dimension is weird
        self.linear1 = nn.Linear(word_dim, dim_feedforward, bias=True)
        self.linear2 = nn.Linear(dim_feedforward, word_dim, bias=True)

    def forward(self, input):
        # return self.ffnn(input)
        return self.linear2(torch.nn.functional.relu(self.linear1(input)))


class Encoder(nn.Module):
    def __init__(self, dim_word, dim_representation, num_heads=8, dim_feedforward=2048):
        super(Encoder, self).__init__()
        self.multi_head = MultiHead(
            dim_word, dim_word, dim_word, dim_representation, num_heads=num_heads
        )
        self.lin_layer = LinearLayer(dim_word)
        self.ffnn = FeedForward(dim_word, dim_feedforward=dim_feedforward)
        self.lin_layer_2 = LinearLayer(dim_word)

    # input is either original input or Z from previous encoder
    def forward(self, input):

        # align time batch dim first
        #Z = self.multi_head(
        #    input_query=input.align_to("time", "batch", "word_dim").rename(None),
        #    input_key=input.align_to("time", "batch", "word_dim").rename(None),
        #    input_value=input.align_to("time", "batch", "word_dim").rename(None)
        #).refine_names("time", "batch", "word_dim").align_to("batch", "time", "word_dim")
        Z = self.multi_head(
            input_query=input,
            input_key=input,
            input_value=input
        )
        Z = self.lin_layer((Z + input.align_as(Z)))
        assert Z.names == ("batch", "time", "word_dim")
        Z_forwarded = self.ffnn(Z).refine_names(*Z.names)
        Z_final = self.lin_layer_2((Z + Z_forwarded))
        return Z_final


class Decoder(nn.Module):
    def __init__(
        self,
        dim_word_decoder,
        dim_word_encoder,
        dim_representation,
        num_heads=8,
        dim_feedforward=2048,
    ):
        super(Decoder, self).__init__()
        self.multi_head_self_att = MultiHead(
            dim_word_decoder,
            dim_word_decoder,
            dim_word_decoder,
            dim_representation,
            num_heads=num_heads,
        )
        self.lin_layer = LinearLayer(dim_word_decoder)
        self.multi_head_enc_dec_att = MultiHead(
            dim_word_decoder,
            dim_word_encoder,
            dim_word_encoder,
            dim_representation,
            num_heads=num_heads,
        )
        self.lin_layer_2 = LinearLayer(dim_word_decoder)
        self.ffnn = FeedForward(dim_word_decoder, dim_feedforward=dim_feedforward)
        self.lin_layer_3 = LinearLayer(dim_word_decoder)

    def forward(self, input_decoder, input_seq_encodings, mask=None):
        # change it
        # mask = None
        z_self_att = self.multi_head_self_att(
            input_decoder.align_to("batch", "time", "word_dim"),
            input_decoder.align_to("batch", "time", "word_dim"),
            input_decoder.align_to("batch", "time", "word_dim"),
            mask=mask,
        )
        #z_self_att = self.multi_head_self_att(
        #    input_query=input_decoder.align_to("time", "batch", "word_dim").rename(None),
        #    input_key=input_decoder.align_to("time", "batch", "word_dim").rename(None),
        #    input_value=input_decoder.align_to("time", "batch", "word_dim").rename(None)
        #).refine_names("time", "batch", "word_dim").align_to("batch", "time", "word_dim")

        z_norm1 = self.lin_layer((z_self_att + input_decoder.align_as(z_self_att)))

        # Encoder Decoder attention
        z_enc_dec_att = self.multi_head_enc_dec_att(
            z_norm1, input_seq_encodings, input_seq_encodings
        )
        #z_enc_dec_att = self.multi_head_enc_dec_att(
        #    input_query=z_norm1.align_to("time", "batch", "word_dim").rename(None),
        #    input_key=input_seq_encodings.align_to("time", "batch", "word_dim").rename(None),
        #    input_value=input_seq_encodings.align_to("time", "batch", "word_dim").rename(None)
        #).refine_names("time", "batch", "word_dim").align_to("batch", "time", "word_dim")
        enc_dec_focused_normalized = self.lin_layer_2(
            # self.dec_w(z_enc_dec_att) + z_norm1
            z_enc_dec_att
            + z_norm1
        )
        Z_forwarded = self.ffnn(enc_dec_focused_normalized).refine_names(
            *enc_dec_focused_normalized.names
        )
        Z_final = self.lin_layer_3((enc_dec_focused_normalized + Z_forwarded))
        return Z_final


class Transformer(nn.Module):
    def __init__(
        self,
        dim_word_encoder,
        dim_word_decoder,
        dim_representation,
        decoder_vocabulary_size,
        num_heads=8,
        num_encoders=6,
        num_decoders=6,
        max_seq_Length=1000,
        dim_feedforward=2048,
    ):
        super(Transformer, self).__init__()
        # dummy for no change later
        assert dim_word_decoder == dim_word_encoder
        # dummy change it later
        dim_representation = dim_word_encoder
        # test_encoding_relative_positions(embedding_size=64)
        self.position_encodings_encoder = torch.tensor(
            [
                positional_encoding(timestep, dim_word_encoder)
                for timestep in range(max_seq_Length)
            ],
            dtype=torch.float32,
        ).refine_names("time", "word_dim")
        self.position_encodings_decoder = torch.tensor(
            [
                positional_encoding(timestep, dim_word_decoder)
                for timestep in range(max_seq_Length)
            ],
            dtype=torch.float32,
        ).refine_names("time", "word_dim")

        self.encoders = nn.ModuleList(
            [
                Encoder(
                    dim_word_encoder,
                    dim_representation,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_encoders)
            ]
        )
        # change later
        self.lin_layer_encoder = LinearLayer(dim_word_decoder)
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    dim_word_decoder,
                    dim_word_encoder,
                    dim_representation,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_decoders)
            ]
        )
        # change later
        self.lin_layer_decoder = LinearLayer(dim_word_decoder)
        self.final_linear = nn.Linear(
            dim_word_decoder, decoder_vocabulary_size, bias=True
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # put there optimizer parameters already registered



        self.trans_final_linear = nn.Linear(
            dim_word_decoder, decoder_vocabulary_size, bias=True
        )
        #self.transformer = nn.Transformer(
        self.transformer = PytorchTransformer(
            d_model=dim_word_encoder,
            nhead=num_heads,
            num_encoder_layers=num_encoders,
            num_decoder_layers=num_decoders,
            dim_feedforward=dim_feedforward,
            dropout=0.0,  # 0.1
            activation="relu",
            custom_encoder=None,
            custom_decoder=None,
        )
        self.optimizer_pytorch = torch.optim.Adam(
            list(self.transformer.parameters())
            + list(self.trans_final_linear.parameters()),
            lr=0.01,
        )
        self.criterion = nn.NLLLoss(reduction="mean")

    def forward(
        self, input_encoder, input_decoder, pytorch_transformer=True, mask_decoder=True
    ):
        #if mask_decoder:
        #    mask = torch.triu(
        #        torch.ones(input_decoder.size("time"), input_decoder.size("time")),
        #        diagonal=1,
        #    ).type(torch.bool)
        #else:
        #    mask = None
        mask = None
        input_encoder = (
            input_encoder.align_to("batch", "time", "word_dim")
            + self.position_encodings_encoder[: input_encoder.size("time")]
        )
        input_decoder = (
            input_decoder.align_to("batch", "time", "word_dim")
            + self.position_encodings_decoder[: input_decoder.size("time")]
        )
        if not pytorch_transformer:
            z_enc = input_encoder
            for encoder in self.encoders:
                z_enc = encoder(z_enc)
            # change later
            z_enc = self.lin_layer_encoder(z_enc)
            output_encoder = z_enc
            z_dec = input_decoder
            for decoder in self.decoders:
                z_dec = decoder(z_dec, output_encoder, mask=mask)
            # change later
            z_dec = self.lin_layer_decoder(z_dec)
            final_linear = self.final_linear(z_dec).refine_names(..., "dec_vocabulary")
        else:
            input_encoder = input_encoder.align_to("time", "batch", "word_dim").rename(
                None
            )
            input_decoder = input_decoder.align_to("time", "batch", "word_dim").rename(
                None
            )
            z_dec = (
                self.transformer(input_encoder, input_decoder, tgt_mask=mask)
                .refine_names("time", "batch", "word_dim")
                .align_to("batch", "time", "word_dim")
            )
            final_linear = self.trans_final_linear(z_dec).refine_names(
                ..., "dec_vocabulary"
            )

        prediction = torch.nn.functional.log_softmax(final_linear, "dec_vocabulary")
        return prediction

    def train_on_batch(
        self, input_encoder, input_decoder, target, pytorch_transformer=True
    ):
        target = target.align_to("batch", "time", "dec_vocabulary")
        target_idx = target.rename(None).argmax(2).refine_names("batch", "time")
        # change it later
        optimizer = self.optimizer_pytorch if pytorch_transformer else self.optimizer
        # self.optimizer.zero_grad()
        optimizer.zero_grad()
        prediction = transformer(
            input_encoder, input_decoder, pytorch_transformer=pytorch_transformer
        )
        loss_on_batch = self.criterion(
            prediction.flatten(["batch", "time"], "batch_time").rename(None),
            target_idx.flatten(["batch", "time"], "batch_time").rename(None),
        )
        loss_on_batch.backward()
        # self.optimizer.step()
        optimizer.step()
        return loss_on_batch


quick_for_debugg = False
n_samples = 200 if quick_for_debugg else int(1e5)

X, y = math_expressions_generation(n_samples=n_samples, n_digits=3, invert=True)
# for X_i, y_i in list(zip(X, y))[:5]:
#    print(X_i[::-1], "=", y_i)
data_processor = DataProcessor(X, y)
input_encoder = data_processor.encoder_input_tr
dim_word_encoder = input_encoder.size("word_dim")
# maybe should make dim word such as is a multiple of 2 as pointed in: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# arbitrary


# try this later see if it works / faster
# https://discuss.pytorch.org/t/concatenate-two-sequential-blocks/6639

input_decoder = data_processor.decoder_input_tr
dim_word_decoder = input_decoder.size("word_dim")

transformer = Transformer(
    dim_word_encoder,
    dim_word_decoder,
    10,
    # data_processor.decoder_vocabulary_size,
    data_processor.vocabulary_size,
    num_heads=2,  # 8
    num_encoders=1,  # 6
    num_decoders=1,  # 6
)

# 0.01

target = data_processor.target_tr


target.names

batch_size = 128
arr = np.arange(data_processor.encoder_input_tr.size("batch"))
np.random.shuffle(arr)
nb_batch = data_processor.encoder_input_tr.size("batch") // batch_size
data_processor.encoder_input_tr = data_processor.encoder_input_tr.align_to("batch", ...)
data_processor.decoder_input_tr = data_processor.decoder_input_tr.align_to("batch", ...)
data_processor.target_tr = data_processor.target_tr.align_to("batch", ...)

nb_epoch = 2000
for epoch in range(nb_epoch):
    for batch_idx in range(nb_batch):
        idxs = arr[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        msg = ""
        for pytorch_transformer in True, False:
            batch_loss_tr = transformer.train_on_batch(
                data_processor.encoder_input_tr.rename(None)[idxs].refine_names(
                    *data_processor.encoder_input_tr.names
                ),
                data_processor.decoder_input_tr.rename(None)[idxs].refine_names(
                    *data_processor.decoder_input_tr.names
                ),
                data_processor.target_tr.rename(None)[idxs].refine_names(
                    *data_processor.target_tr.names
                ),
                pytorch_transformer=pytorch_transformer,
            )
            if pytorch_transformer:
                msg += f"pytorch implem loss on batch: {batch_loss_tr}"
            else:
                msg += f" personal implem loss on batch: {batch_loss_tr}"
        msg += f" epoch: {epoch}"
        print(msg)
        # print(f"loss on batch: {batch_loss_tr} epoch: {epoch}")
