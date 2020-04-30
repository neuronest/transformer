import os

import numpy as np
import argparse
import operator

from sequence_processor import SequenceProcessor
from transformer import generate_batches, decoder_triangular_training_mask, Transformer

ARG_PARSER = argparse.ArgumentParser()
ARGS = None


def handle_arguments():
    # Debug arguments
    ARG_PARSER.add_argument(
        "--quick-debug", default=True, type=lambda x: str(x).lower() == "true", help=""
    )
    # Transformer architecture arguments
    ARG_PARSER.add_argument("--num-heads", default=2, type=int, help="")
    ARG_PARSER.add_argument("--num-encoders", default=1, type=int, help="")
    ARG_PARSER.add_argument("--num-decoders", default=1, type=int, help="")

    # Training arguments
    ARG_PARSER.add_argument("--lr", default=0.01, type=float, help="")
    ARG_PARSER.add_argument("--batch-size", default=128, type=int, help="")
    ARG_PARSER.add_argument("--epochs", default=2000, type=int, help="")
    ARG_PARSER.add_argument(
        "--validate", default=True, type=lambda x: str(x).lower() == "true", help=""
    )
    ARG_PARSER.add_argument(
        "--do-inference", default=True, type=lambda x: str(x).lower() == "true", help=""
    )

    return ARG_PARSER.parse_args()


# this code is for toy data generation, will be discarded later
def get_math_data(small_nb_samples=200, big_nb_samples=int(1e5), big_dataset=True):

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
    X, y = math_expressions_generation(n_samples=n_samples, n_digits=3, invert=False)
    data_processor = SequenceProcessor(X, y, with_positional_encodings=True)
    assert data_processor.encoder_input_tr.size(
        "word_dim"
    ) == data_processor.decoder_input_tr.size("word_dim")
    return data_processor


def infer(
    self,
    math_expression,
    math_expression_target,
    char_to_idx,
    idx_to_char,
    max_encode_len,
    max_decode_len,
    pytorch_transformer,
    voc_size,
    GO="=",
    EOS="\n",
    enc_input=None,
    dec_input=None,
):
    target_idxs = [char_to_idx[c] for c in math_expression_target]

    dec_input_idxs = dec_input.rename(None).argmax(2)[0]

    # +1 for GO term
    position_encodings = torch.tensor(
        [
            positional_encoding(t, voc_size)
            for t in range(max(max_encode_len, max_decode_len) + 1)
        ],
        dtype=torch.float32,
    ).refine_names("time", "word_dim")

    mask_decoder = decoder_triangular_training_mask(dec_input.size("time"))
    teacher_forcing_preds = self(
        enc_input,
        dec_input,
        pytorch_transformer=pytorch_transformer,
        mask_decoder=mask_decoder,
    )
    teacher_forcing_preds_idxs = teacher_forcing_preds.rename(None).argmax(2)
    teacher_forcing_preds_chars = [
        idx_to_char[int(idx)] for idx in teacher_forcing_preds_idxs[0]
    ]

    encoder_input_idxs = [char_to_idx[c] for c in math_expression]
    encoder_input = torch.zeros(1, max_encode_len, voc_size)
    encoder_input[0, np.arange(len(encoder_input_idxs)), encoder_input_idxs] = 1
    encoder_input = encoder_input.refine_names("batch", "time", "word_dim")
    encoder_input += position_encodings[: encoder_input.size("time")]

    decoder_input_idxs = [char_to_idx[GO]]
    decoder_input = torch.zeros(1, len(decoder_input_idxs), voc_size)
    decoder_input[0, np.arange(len(decoder_input_idxs)), decoder_input_idxs] = 1
    decoder_input = decoder_input.refine_names("batch", "time", "word_dim")
    decoder_input += position_encodings[: decoder_input.size("time")]

    decoded_expression = []
    for _ in range(max_decode_len):
        # see that
        mask_decoder = decoder_triangular_training_mask(decoder_input.size("time"))
        pred_tensor = self(
            encoder_input, decoder_input, pytorch_transformer=pytorch_transformer
        )
        # last timestep
        pred_tensor = pred_tensor.rename(None)[0, -1, :]
        pred_idx = int(pred_tensor.argmax())
        if pred_idx == char_to_idx[EOS]:
            break
        decoder_input_idxs.append(int(pred_idx))
        decoder_input = torch.zeros(1, len(decoder_input_idxs), voc_size)
        decoder_input[0, np.arange(len(decoder_input_idxs)), decoder_input_idxs] = 1
        decoder_input = decoder_input.refine_names("batch", "time", "word_dim")
        decoder_input += position_encodings[: decoder_input.size("time")]

        pred_char = idx_to_char[pred_idx]
        decoded_expression.append(pred_char)
    return "".join(decoded_expression)


if __name__ == "__main__":

    ARGS = handle_arguments()
    sequence_processor = get_math_data(big_dataset=not ARGS.quick_debug)
    transformer = Transformer(
        sequence_processor.encoder_input_tr.size("word_dim"),
        sequence_processor.vocabulary_size,
        num_heads=ARGS.num_heads,
        num_encoders=ARGS.num_encoders,
        num_decoders=ARGS.num_decoders,
    )
    mask_decoder = (
        decoder_triangular_training_mask(sequence_processor.decoder_input_tr.size("time"))
    )
    validation_data = (
        (
            (sequence_processor.encoder_input_val, sequence_processor.decoder_input_val),
            sequence_processor.target_val,
        )
        if ARGS.validate
        else None
    )
    for epoch in range(ARGS.epochs):
        for (
            batch_encoder_inputs,
            batch_decoder_inputs,
            batch_targets,
        ) in generate_batches(
            sequence_processor.encoder_input_tr,
            sequence_processor.decoder_input_tr,
            sequence_processor.target_tr,
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
        if validation_data is not None:
            msg = ""
            for use_pytorch_transformer in True, False:
                loss_val = transformer.validate(
                    validation_data, use_pytorch_transformer, mask_decoder=None
                )
                if use_pytorch_transformer:
                    msg += f"pytorch implem val loss: {loss_val}"
                else:
                    msg += f" personal implem val loss: {loss_val}"
            print(msg)

        if (ARGS.do_inference and not ARGS.quick_debug) or (
            ARGS.quick_debug and ARGS.do_inference and batch_loss < 0.1
        ):
            if ARGS.quick_debug:
                X = sequence_processor.X_tr
                y = sequence_processor.y_tr
                enc_input = sequence_processor.encoder_input_tr
                dec_input = sequence_processor.decoder_input_tr
            else:
                X = sequence_processor.X_vals
                y = sequence_processor.y_val
                enc_input = sequence_processor.encoder_input_val
                dec_input = sequence_processor.decoder_input_val

            for use_pytorch_transformer in True, False:
                if use_pytorch_transformer:
                    print(os.linesep, "Inferences for pytorch implem")
                else:
                    print(os.linesep, "Inferences for personal implem")
                for i in range(10):
                    y_pred = transformer.infer(
                        X[i],
                        y[i].replace(sequence_processor.GO, ""),
                        sequence_processor.char_index,
                        sequence_processor.index_char,
                        sequence_processor.max_encoder_sequence_length,
                        sequence_processor.max_decoder_sequence_length,
                        use_pytorch_transformer,
                        sequence_processor.vocabulary_size,
                        GO=sequence_processor.GO,
                        EOS=sequence_processor.EOS,
                        enc_input=enc_input.rename(None)[i : i + 1].refine_names(
                            *enc_input.names
                        ),
                        dec_input=dec_input.rename(None)[i : i + 1].refine_names(
                            *dec_input.names
                        ),
                    )
                    print(
                        f"{X[i]} = {y_pred} predicted / {y[i].replace(sequence_processor.GO, '').replace(sequence_processor.EOS, '')} expected"
                    )
                print()