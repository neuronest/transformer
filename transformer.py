import numpy as np
import torch.nn as nn
import torch


class Transformer(nn.Module):
    def __init__(
        self,
        dim_word,
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
        self.layer_norm_encoder = NormalizationLayer(dim_word)
        self.decoders = nn.ModuleList(
            [
                Decoder(dim_word, num_heads=num_heads, dim_feedforward=dim_feedforward)
                for _ in range(num_decoders)
            ]
        )
        # layer norm after decoders comes from Pytorch implemetation
        self.layer_norm_decoder = NormalizationLayer(dim_word)

    def forward(self, input_encoder, input_decoder, mask_decoder=None):
        z_enc = input_encoder
        for encoder in self.encoders:
            z_enc = encoder(z_enc)
        # pytorch implem adds norm layer after encoders
        z_enc = self.layer_norm_encoder(z_enc)
        output_encoder = z_enc

        z_dec = input_decoder
        for decoder in self.decoders:
            z_dec = decoder(z_dec, output_encoder, mask=mask_decoder)
        # pytorch implem adds norm layer after decoders
        z_dec = self.layer_norm_decoder(z_dec)
        return z_dec


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

        z_norm1 = self.norm_layer((z_self_att + input_decoder.align_as(z_self_att)))

        # Encoder Decoder attention
        z_enc_dec_att = self.multi_head_enc_dec_att(
            z_norm1, input_seq_encodings, input_seq_encodings
        )
        enc_dec_focused_normalized = self.norm_layer2(z_enc_dec_att + z_norm1)
        Z_forwarded = self.ffnn(enc_dec_focused_normalized).refine_names(
            *enc_dec_focused_normalized.names
        )
        Z_final = self.norm_layer3((enc_dec_focused_normalized + Z_forwarded))
        return Z_final


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


class FeedForward(nn.Module):
    def __init__(self, word_dim, dim_feedforward=2048):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(word_dim, dim_feedforward, bias=True)
        self.linear2 = nn.Linear(dim_feedforward, word_dim, bias=True)

    def forward(self, input):
        return self.linear2(torch.nn.functional.relu(self.linear1(input)))


class NormalizationLayer(nn.Module):
    def __init__(self, dim_word):
        super(NormalizationLayer, self).__init__()
        self.alpha = torch.nn.Parameter(
            data=torch.rand(1, dim_word).refine_names("time", "word_dim"),
            requires_grad=True,
        )
        # pytorch trick, special init with ones and zeros for alpha and beta
        torch.nn.init.ones_(self.alpha)

        self.beta = torch.nn.Parameter(
            data=torch.rand(1, dim_word).refine_names("time", "word_dim"),
            requires_grad=True,
        )
        torch.nn.init.zeros_(self.beta)

    def forward(self, Z):
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


class MultiHead(nn.Module):
    def __init__(
        self,
        dim_word,
        num_heads=8,
        dim_key_query_per_head=None,
        dim_value_per_head=None,
    ):
        super(MultiHead, self).__init__()
        self.dim_word = dim_word
        self.num_heads = num_heads
        self.dim_key_query_all_heads, self.dim_value_all_heads = (
            self.dim_word,
            self.dim_word,
        )
        if dim_key_query_per_head is not None:
            self.dim_key_query_all_heads = dim_key_query_per_head * num_heads
        if dim_value_per_head is not None:
            self.dim_value_all_heads = dim_value_per_head * num_heads

        assert (
            self.dim_key_query_all_heads // num_heads * num_heads
            == self.dim_key_query_all_heads
        )
        assert (
            self.dim_value_all_heads // num_heads * num_heads
            == self.dim_value_all_heads
        )
        # the greater the number of dimensions involved in the dot products
        # before softmax the more we scale to flatten the probabilities
        self.d_k = float(self.dim_key_query_all_heads)
        self.Q = nn.Linear(dim_word, self.dim_key_query_all_heads, bias=True)
        self.K = nn.Linear(dim_word, self.dim_key_query_all_heads, bias=True)
        self.V = nn.Linear(dim_word, self.dim_value_all_heads, bias=True)
        self.linear_out = nn.Linear(self.dim_value_all_heads, dim_word, bias=True)

    def forward(self, input_query, input_key, input_value, mask=None):
        assert "batch" in input_query.names and "time" in input_query.names
        assert "batch" in input_key.names and "time" in input_key.names
        assert "batch" in input_value.names and "time" in input_value.names

        def multi_head_repr(linear_layer, input, num_heads, dim_all_heads):

            multi_head_representation = linear_layer(input).refine_names(
                ..., "dim_all_heads"
            )
            multi_head_representation = multi_head_representation.unflatten(
                "dim_all_heads",
                [("head", num_heads), ("dim", dim_all_heads // num_heads)],
            )
            multi_head_representation = multi_head_representation.align_to(
                "head", "batch", "time", "dim"
            )
            return multi_head_representation

        multi_head_q = multi_head_repr(
            self.Q, input_query, self.num_heads, self.dim_key_query_all_heads
        )
        multi_head_k = multi_head_repr(
            self.K, input_key, self.num_heads, self.dim_key_query_all_heads
        )
        multi_head_v = multi_head_repr(
            self.V, input_value, self.num_heads, self.dim_value_all_heads
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
    Z = torch.nn.functional.softmax(Z, "time_keys")

    Z = torch.matmul(
        Z, values.rename(time="time_keys").align_to(*other_names, "time_keys", "dim")
    )
    return Z


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


def decoder_triangular_training_mask(nb_timesteps):
    mask = torch.triu(torch.ones(nb_timesteps, nb_timesteps), diagonal=1).type(
        torch.bool
    )
    return mask


class TrainableTransformer(Transformer):
    def __init__(self, decoder_vocabulary_size, lr, *args, **kwargs):
        super(TrainableTransformer, self).__init__(*args, **kwargs)
        self.final_linear = nn.Linear(self.dim_word, decoder_vocabulary_size, bias=True)
        # see how change optimizer lr during training
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, input_encoder, input_decoder, mask_decoder=None):
        z_dec = Transformer.forward(
            self, input_encoder, input_decoder, mask_decoder=None
        )
        return torch.nn.functional.log_softmax(
            self.final_linear(z_dec).refine_names(..., "dec_vocabulary"),
            "dec_vocabulary",
        )

    def train_on_batch(
        self, batch_input_encoder, batch_input_decoder, batch_target, mask_decoder=None
    ):
        self.optimizer.zero_grad()
        prediction = self(
            batch_input_encoder, batch_input_decoder, mask_decoder=mask_decoder
        )
        loss_on_batch = self._batched_ce_loss(prediction, batch_target)
        loss_on_batch.backward()
        self.optimizer.step()
        return loss_on_batch

    def train(self, input_encoder, input_decoder, target, do_target_mask=True, target_mask=None):
        training_target_mask = TrainableTransformer._handle_target_mask_arg(
            do_target_mask, target_mask, input_decoder
        )


    @staticmethod
    def _handle_target_mask_arg(do_target_mask, target_mask, input_decoder):
        if do_target_mask:
            if target_mask is None:
                mask = decoder_triangular_training_mask(input_decoder.size("time"))
            else:
                mask = target_mask
        else:
            if target_mask is not None:
                raise Exception(
                    "target_mask bool arg says no mask but target mask provided"
                )
            else:
                mask = None
        return mask

    def _batched_ce_loss(self, prediction, target, reduction="mean"):
        target = target.align_to("batch", "time", "dec_vocabulary")
        target_idx = target.rename(None).argmax(2).refine_names("batch", "time")
        batched_ce_loss = nn.NLLLoss(reduction=reduction)(
            prediction.flatten(["batch", "time"], "batch_time").rename(None),
            target_idx.flatten(["batch", "time"], "batch_time").rename(None),
        )
        return batched_ce_loss

    def validate(self, validation_data, mask_decoder=None):
        input_val, target_val = validation_data
        prediction_val = self(*input_val, mask_decoder=mask_decoder)
        loss_on_val = self._batched_ce_loss(prediction_val, target_val)
        return loss_on_val
