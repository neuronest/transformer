import numpy as np
import torch.nn as nn
import torch
import argparse

ARG_PARSER = argparse.ArgumentParser()
ARGS = None


def decoder_triangular_training_mask(nb_timesteps):
    mask = torch.triu(torch.ones(nb_timesteps, nb_timesteps), diagonal=1).type(
        torch.bool
    )
    return mask


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
        # Z.masked_fill_(mask, float('-inf'))
    Z = torch.nn.functional.softmax(Z, "time_keys")

    Z = torch.matmul(
        Z, values.rename(time="time_keys").align_to(*other_names, "time_keys", "dim")
    )
    return Z


class MultiHead(nn.Module):
    def __init__(self, dim_word, num_heads=8):
        super(MultiHead, self).__init__()
        if not ARGS.use_pytorch_multi_head:
            assert dim_word // num_heads * num_heads == dim_word
            self.dim_word = dim_word
            self.num_heads = num_heads
            self.dim_repr = (
                dim_word if ARGS.use_pytorch_dim_per_head else dim_word * num_heads
            )
            self.d_k = float(self.dim_repr)
            self.Q = nn.Linear(dim_word, self.dim_repr, bias=True)
            self.K = nn.Linear(dim_word, self.dim_repr, bias=True)
            self.V = nn.Linear(dim_word, self.dim_repr, bias=True)
            self.linear_out = nn.Linear(
                self.dim_repr, dim_word, bias=ARGS.use_pytorch_linearout_bias
            )  # bias=False
        else:
            from torch.nn.modules.activation import MultiheadAttention

            self.multi_head = MultiheadAttention(dim_word, num_heads, dropout=0.0)

    def forward(self, input_query, input_key, input_value, mask=None):
        if ARGS.use_pytorch_multi_head:
            return (
                self.multi_head(
                    input_query.align_to("time", "batch", "word_dim").rename(None),
                    input_key.align_to("time", "batch", "word_dim").rename(None),
                    input_value.align_to("time", "batch", "word_dim").rename(None),
                    attn_mask=mask,
                )[0]
                .refine_names("time", "batch", "word_dim")
                .align_to("batch", "time", "word_dim")
            )
        else:
            assert "batch" in input_query.names and "time" in input_query.names
            assert "batch" in input_key.names and "time" in input_key.names
            assert "batch" in input_value.names and "time" in input_value.names

            def multi_head_repr(linear_layer, input, num_heads, dim_all_heads):

                multi_head_representation = linear_layer(input).refine_names(
                    ..., "dim_all_heads"
                )
                # other possibility, dim_representation does not have to be divisible by
                # multi_head_representation = multi_head_representation.unflatten(
                #    "dim_times_n_head", [("head", num_heads), ("dim", dim_representation)]
                # )
                multi_head_representation = multi_head_representation.unflatten(
                    "dim_all_heads",
                    [("head", num_heads), ("dim", dim_all_heads // num_heads)],
                )
                multi_head_representation = multi_head_representation.align_to(
                    "head", "batch", "time", "dim"
                )
                return multi_head_representation

            multi_head_q = multi_head_repr(
                self.Q, input_query, self.num_heads, self.dim_repr
            )
            multi_head_k = multi_head_repr(
                self.K, input_key, self.num_heads, self.dim_repr
            )
            multi_head_v = multi_head_repr(
                self.V, input_value, self.num_heads, self.dim_repr
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
            Z = self.linear_out(Z).refine_names("batch", "time", "word_dim")
            return Z


class NormalizationLayer(nn.Module):
    def __init__(self, dim_word):
        super(NormalizationLayer, self).__init__()
        if ARGS.use_pytorch_norm_layer:
            from torch.nn.modules.normalization import LayerNorm

            self.layer_norm = LayerNorm(dim_word)
        else:
            self.alpha = torch.nn.Parameter(
                data=torch.rand(1, dim_word).refine_names("time", "word_dim"),
                requires_grad=True,
            )
            # they perform special init with ones and zeros for w and b
            torch.nn.init.xavier_normal_(self.alpha)
            self.beta = torch.nn.Parameter(
                data=torch.rand(1, dim_word).refine_names("time", "word_dim"),
                requires_grad=True,
            )
            torch.nn.init.xavier_normal_(self.beta)

    def forward(self, Z):
        if ARGS.use_pytorch_norm_layer:
            return self.layer_norm(Z.rename(None)).refine_names(*Z.names)
        else:
            mean = Z.mean("word_dim")
            std = Z.std("word_dim")
            mean = (
                mean.rename(None)
                .reshape(mean.shape + (1,))
                .refine_names(*mean.names, "word_dim")
            )
            std = (
                std.rename(None)
                .reshape(std.shape + (1,))
                .refine_names(*std.names, "word_dim")
            )
            assert std.names == mean.names == Z.names == ("batch", "time", "word_dim")
            Z = (Z - mean) / std
            return Z * self.alpha + self.beta


class FeedForward(nn.Module):
    def __init__(self, word_dim, dim_feedforward=2048):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(word_dim, dim_feedforward, bias=True)
        self.linear2 = nn.Linear(dim_feedforward, word_dim, bias=True)

    def forward(self, input):
        return self.linear2(torch.nn.functional.relu(self.linear1(input)))


class Encoder(nn.Module):
    def __init__(self, dim_word, num_heads=8, dim_feedforward=2048):
        super(Encoder, self).__init__()
        self.multi_head = MultiHead(dim_word, num_heads=num_heads)
        self.norm_layer = NormalizationLayer(dim_word)
        self.ffnn = FeedForward(dim_word, dim_feedforward=dim_feedforward)
        self.norm_layer2 = NormalizationLayer(dim_word)

    # input is either original input or Z from previous encoder
    def forward(self, input):
        Z = self.multi_head(input_query=input, input_key=input, input_value=input)
        Z = self.norm_layer((Z + input.align_as(Z)))
        assert Z.names == ("batch", "time", "word_dim")
        Z_forwarded = self.ffnn(Z).refine_names(*Z.names)
        Z_final = self.norm_layer2((Z + Z_forwarded))
        return Z_final


class Decoder(nn.Module):
    def __init__(self, dim_word_decoder, num_heads=8, dim_feedforward=2048):
        super(Decoder, self).__init__()
        self.multi_head_self_att = MultiHead(dim_word_decoder, num_heads=num_heads)
        self.norm_layer = NormalizationLayer(dim_word_decoder)
        self.multi_head_enc_dec_att = MultiHead(dim_word_decoder, num_heads=num_heads)
        self.norm_layer2 = NormalizationLayer(dim_word_decoder)
        self.ffnn = FeedForward(dim_word_decoder, dim_feedforward=dim_feedforward)
        self.norm_layer3 = NormalizationLayer(dim_word_decoder)

    def forward(self, input_decoder, input_seq_encodings, mask=None):
        z_self_att = self.multi_head_self_att(
            input_decoder, input_decoder, input_decoder, mask=mask
        )

        # z_self_att = self.multi_head_self_att(
        #    input_query=input_decoder.align_to("time", "batch", "word_dim").rename(None),
        #    input_key=input_decoder.align_to("time", "batch", "word_dim").rename(None),
        #    input_value=input_decoder.align_to("time", "batch", "word_dim").rename(None)
        # ).refine_names("time", "batch", "word_dim").align_to("batch", "time", "word_dim")

        z_norm1 = self.norm_layer((z_self_att + input_decoder.align_as(z_self_att)))

        # Encoder Decoder attention
        z_enc_dec_att = self.multi_head_enc_dec_att(
            z_norm1, input_seq_encodings, input_seq_encodings
        )
        # z_enc_dec_att = self.multi_head_enc_dec_att(
        #    input_query=z_norm1.align_to("time", "batch", "word_dim").rename(None),
        #    input_key=input_seq_encodings.align_to("time", "batch", "word_dim").rename(None),
        #    input_value=input_seq_encodings.align_to("time", "batch", "word_dim").rename(None)
        # ).refine_names("time", "batch", "word_dim").align_to("batch", "time", "word_dim")
        enc_dec_focused_normalized = self.norm_layer2(z_enc_dec_att + z_norm1)
        Z_forwarded = self.ffnn(enc_dec_focused_normalized).refine_names(
            *enc_dec_focused_normalized.names
        )
        Z_final = self.norm_layer3((enc_dec_focused_normalized + Z_forwarded))
        return Z_final


class Transformer(nn.Module):
    def __init__(
        self,
        dim_word,
        decoder_vocabulary_size,
        num_heads=8,
        num_encoders=6,
        num_decoders=6,
        dim_feedforward=2048,
    ):
        super(Transformer, self).__init__()
        self.encoders = nn.ModuleList(
            [
                Encoder(dim_word, num_heads=num_heads, dim_feedforward=dim_feedforward)
                for _ in range(num_encoders)
            ]
        )
        # layer norm after encoders comes from Pytorch implemetation
        if ARGS.use_pytorch_norm_after_encoders_decoders:
            self.layer_norm_encoder = NormalizationLayer(dim_word)
        self.decoders = nn.ModuleList(
            [
                Decoder(dim_word, num_heads=num_heads, dim_feedforward=dim_feedforward)
                for _ in range(num_decoders)
            ]
        )
        # layer norm after decoders comes from Pytorch implemetation
        if ARGS.use_pytorch_norm_after_encoders_decoders:
            self.layer_norm_decoder = NormalizationLayer(dim_word)
        self.final_linear = nn.Linear(dim_word, decoder_vocabulary_size, bias=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=ARGS.lr)

        # pytorch transformer implem part
        # put there optimizer parameters already registered
        self.trans_final_linear = nn.Linear(
            dim_word, decoder_vocabulary_size, bias=True
        )
        # self.transformer = nn.Transformer(
        from torch.nn.modules.transformer import Transformer as PytorchTransformer
        #from pytorch.torch.nn.modules.transformer import Transformer as PytorchTransformer
        self.transformer = PytorchTransformer(
            d_model=dim_word,
            nhead=num_heads,
            num_encoder_layers=num_encoders,
            num_decoder_layers=num_decoders,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            custom_encoder=None,
            custom_decoder=None,
        )
        self.optimizer_pytorch = torch.optim.Adam(
            list(self.transformer.parameters())
            + list(self.trans_final_linear.parameters()),
            lr=ARGS.lr,
        )

        self.criterion = nn.NLLLoss(reduction="mean")

    def forward(
        self, input_encoder, input_decoder, pytorch_transformer=True, mask_decoder=None
    ):
        if not pytorch_transformer:
            z_enc = input_encoder
            for encoder in self.encoders:
                z_enc = encoder(z_enc)
            # pytorch implem adds norm layer after encoders
            if ARGS.use_pytorch_norm_after_encoders_decoders:
                z_enc = self.layer_norm_encoder(z_enc)
            output_encoder = z_enc

            z_dec = input_decoder
            for decoder in self.decoders:
                z_dec = decoder(z_dec, output_encoder, mask=mask_decoder)
            # pytorch implem adds norm layer after decoders
            if ARGS.use_pytorch_norm_after_encoders_decoders:
                z_dec = self.layer_norm_decoder(z_dec)
            return torch.nn.functional.log_softmax(
                self.final_linear(z_dec).refine_names(..., "dec_vocabulary"),
                "dec_vocabulary",
            )
        else:
            input_encoder = input_encoder.align_to("time", "batch", "word_dim").rename(
                None
            )
            input_decoder = input_decoder.align_to("time", "batch", "word_dim").rename(
                None
            )
            z_dec = (
                self.transformer(input_encoder, input_decoder, tgt_mask=mask_decoder)
                .refine_names("time", "batch", "word_dim")
                .align_to("batch", "time", "word_dim")
            )
            final_linear = self.trans_final_linear(z_dec).refine_names(
                ..., "dec_vocabulary"
            )
            return torch.nn.functional.log_softmax(final_linear, "dec_vocabulary")

    def train_on_batch(
        self,
        input_encoder,
        input_decoder,
        target,
        mask_decoder=None,
        pytorch_transformer=True,
    ):
        target = target.align_to("batch", "time", "dec_vocabulary")
        target_idx = target.rename(None).argmax(2).refine_names("batch", "time")
        # change it later
        optimizer = self.optimizer_pytorch if pytorch_transformer else self.optimizer
        # self.optimizer.zero_grad()
        optimizer.zero_grad()
        prediction = self(
            input_encoder,
            input_decoder,
            pytorch_transformer=pytorch_transformer,
            mask_decoder=mask_decoder,
        )
        loss_on_batch = self.criterion(
            prediction.flatten(["batch", "time"], "batch_time").rename(None),
            target_idx.flatten(["batch", "time"], "batch_time").rename(None),
        )
        loss_on_batch.backward()
        # self.optimizer.step()
        optimizer.step()
        return loss_on_batch


def handle_arguments():
    # Debug arguments
    ARG_PARSER.add_argument(
        "--quick-debug", default=True, type=lambda x: str(x).lower() == "true", help=""
    )
    ARG_PARSER.add_argument(
        "--use-mask",
        default=True,
        type=lambda x: str(x).lower() == "true",
        help="",
    )
    ARG_PARSER.add_argument(
        "--use-pytorch-multi-head",
        default=True,
        type=lambda x: str(x).lower() == "true",
        help="",
    )
    ARG_PARSER.add_argument(
        "--use-pytorch-dim-per-head",
        default=True,
        type=lambda x: str(x).lower() == "true",
        help="",
    )
    ARG_PARSER.add_argument(
        "--use-pytorch-norm-layer",
        default=True,
        type=lambda x: str(x).lower() == "true",
        help="",
    )
    ARG_PARSER.add_argument(
        "--use-pytorch-linearout-bias",
        default=True,
        type=lambda x: str(x).lower() == "true",
        help="",
    )
    ARG_PARSER.add_argument(
        "--use-pytorch-norm-after-encoders-decoders",
        default=True,
        type=lambda x: str(x).lower() == "true",
        help="",
    )

    # Transformer architecture arguments
    ARG_PARSER.add_argument("--num-heads", default=2, type=int, help="")
    ARG_PARSER.add_argument("--num-encoders", default=1, type=int, help="")
    ARG_PARSER.add_argument("--num-decoders", default=1, type=int, help="")

    # Training arguments
    ARG_PARSER.add_argument("--lr", default=0.01, type=float, help="")
    ARG_PARSER.add_argument("--batch-size", default=128, type=int, help="")
    ARG_PARSER.add_argument("--epochs", default=2000, type=int, help="")

    return ARG_PARSER.parse_args()


def generate_batches(encoder_inputs, decoder_inputs, targets, batch_size):
    assert (
        encoder_inputs.names[0]
        == decoder_inputs.names[0]
        == targets.names[0]
        == "batch"
    )
    samples_idxs = np.arange(encoder_inputs.size("batch"))
    np.random.shuffle(samples_idxs)
    nb_batch = encoder_inputs.size("batch") // batch_size
    # so can keep on calling generator after inner loop has expired all batches
    for batch_idx in range(nb_batch):
        batch_idxs = samples_idxs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch = (
            encoder_inputs.rename(None)[batch_idxs].refine_names(*encoder_inputs.names),
            decoder_inputs.rename(None)[batch_idxs].refine_names(*decoder_inputs.names),
            targets.rename(None)[batch_idxs].refine_names(*targets.names),
        )
        yield batch


# this code is for toy data generation, will be discarded later
def get_math_data(small_nb_samples=200, big_nb_samples=int(1e5), big_dataset=True):
    import operator

    # raises error if put at file top as data_processor
    # imports transformer module function
    # not a problem since get_math_data will be discarded eventually
    from data_processor import DataProcessor

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

    n_samples = big_nb_samples if big_dataset else small_nb_samples
    X, y = math_expressions_generation(n_samples=n_samples, n_digits=3, invert=True)
    data_processor = DataProcessor(X, y, with_positional_encodings=True)
    assert data_processor.encoder_input_tr.size(
        "word_dim"
    ) == data_processor.decoder_input_tr.size("word_dim")
    return data_processor


if __name__ == "__main__":

    ARGS = handle_arguments()
    data_processor = get_math_data(big_dataset=not ARGS.quick_debug)
    transformer = Transformer(
        data_processor.encoder_input_tr.size("word_dim"),
        data_processor.vocabulary_size,
        num_heads=ARGS.num_heads,
        num_encoders=ARGS.num_encoders,
        num_decoders=ARGS.num_decoders,
    )
    mask_decoder = (
        decoder_triangular_training_mask(data_processor.decoder_input_tr.size("time"))
        if ARGS.use_mask
        else None
    )
    for epoch in range(ARGS.epochs):
        for (
            batch_encoder_inputs,
            batch_decoder_inputs,
            batch_targets,
        ) in generate_batches(
            data_processor.encoder_input_tr,
            data_processor.decoder_input_tr,
            data_processor.target_tr,
            ARGS.batch_size,
        ):
            msg = ""
            # train alternatively pytorch implementation and our implementation
            for use_pytorch_transformer in True, False:
                batch_loss = transformer.train_on_batch(
                    batch_encoder_inputs,
                    batch_decoder_inputs,
                    batch_targets,
                    mask_decoder=mask_decoder,
                    pytorch_transformer=use_pytorch_transformer,
                )
                if use_pytorch_transformer:
                    msg += f"pytorch implem loss on batch: {batch_loss}"
                else:
                    msg += f" personal implem loss on batch: {batch_loss}"
                msg += f" epoch: {epoch}"
            print(msg)
