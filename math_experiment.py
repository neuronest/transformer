import os

import torch
import numpy as np
import argparse
import operator

from sequence_processor import SequenceProcessor
from transformer import TrainableTransformer


# this code is for toy data generation, will be discarded later
def get_math_data(small_nb_samples=200, big_nb_samples=int(1e5), big_dataset=True):
    def math_expressions_generation(n_samples=1000, n_digits=3, invert=False):
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


def do_inference(transformer, math_expression, sequence_processor):
    # +1 for GO term
    def get_input(idxs_words, for_encoder=True):
        nb_timesteps = (
            sequence_processor.max_encoder_sequence_length
            if for_encoder
            else len(idxs_words)
        )
        input = torch.zeros(1, nb_timesteps, sequence_processor.vocabulary_size)
        input[0, np.arange(len(idxs_words)), idxs_words] = 1
        input = input.refine_names("batch", "time", "word_dim")
        input += sequence_processor.position_encodings[: input.size("time")]
        return input

    encoder_input_idxs = [sequence_processor.char_index[c] for c in math_expression]
    encoder_input = get_input(encoder_input_idxs, for_encoder=True)

    decoder_input_idxs = [sequence_processor.char_index[sequence_processor.GO]]
    decoder_input = get_input(decoder_input_idxs, for_encoder=False)

    decoded_expression = []
    for _ in range(sequence_processor.max_decoder_sequence_length - 1):
        # take last timestep prediction
        # final timestep prediction has been optained in same conditions as training
        # other timesteps predictions are wrong as future position have been attended
        pred_tensor = transformer(encoder_input, decoder_input).rename(None)[0, -1, :]
        pred_idx = int(pred_tensor.argmax())
        if pred_idx == sequence_processor.char_index[sequence_processor.EOS]:
            break
        decoder_input_idxs.append(int(pred_idx))
        decoder_input = get_input(decoder_input_idxs, for_encoder=False)
        pred_char = sequence_processor.index_char[pred_idx]
        decoded_expression.append(pred_char)
    return "".join(decoded_expression)


def handle_arguments():
    # Debug arguments
    arg_parser.add_argument(
        "--quick-debug", default=False, type=lambda x: str(x).lower() == "true", help=""
    )
    # Transformer architecture arguments
    arg_parser.add_argument("--num-heads", default=2, type=int, help="")
    arg_parser.add_argument("--num-encoders", default=1, type=int, help="")
    arg_parser.add_argument("--num-decoders", default=1, type=int, help="")

    # Training arguments
    arg_parser.add_argument("--lr", default=0.01, type=float, help="")
    arg_parser.add_argument("--batch-size", default=128, type=int, help="")
    arg_parser.add_argument("--epochs", default=2000, type=int, help="")
    arg_parser.add_argument(
        "--validate", default=True, type=lambda x: str(x).lower() == "true", help=""
    )

    # Others
    arg_parser.add_argument(
        "--do-inference", default=True, type=lambda x: str(x).lower() == "true", help=""
    )

    return arg_parser.parse_args()


if __name__ == "__main__":

    args = handle_arguments(argparse.ArgumentParser())
    sequence_processor = get_math_data(big_dataset=not args.quick_debug)
    transformer = TrainableTransformer(
        sequence_processor.vocabulary_size,
        args.lr,
        sequence_processor.encoder_input_tr.size("word_dim"),
        num_heads=args.num_heads,
        num_encoders=args.num_encoders,
        num_decoders=args.num_decoders,
    )
    validation_data = (
        (
            (
                sequence_processor.encoder_input_val,
                sequence_processor.decoder_input_val,
            ),
            sequence_processor.target_val,
        )
        if args.validate
        else None
    )
    for epoch in range(args.epochs):
        transformer.train(
            sequence_processor.encoder_input_tr,
            sequence_processor.decoder_input_tr,
            sequence_processor.target_tr,
            epochs=1,
            batch_size=args.batch_size,
            do_target_mask=True,
            validation_data=validation_data,
        )
        print(epoch)
        if (args.do_inference and not args.quick_debug) or (
                args.quick_debug and args.do_inference and epoch > 150
        ):
            if args.quick_debug:
                X = sequence_processor.X_tr
                y = sequence_processor.y_tr
                enc_input = sequence_processor.encoder_input_tr
                dec_input = sequence_processor.decoder_input_tr
            else:
                X = sequence_processor.X_val
                y = sequence_processor.y_val
                enc_input = sequence_processor.encoder_input_val
                dec_input = sequence_processor.decoder_input_val
            print(os.linesep, "Inferences:")
            for i in range(15):
                y_pred = do_inference(transformer, X[i], sequence_processor)
                print(
                    f"{X[i]} = {y_pred} predicted / "
                    f"{y[i].replace(sequence_processor.GO, '').replace(sequence_processor.EOS,'')} expected"
                    f"{os.linesep}"
                )
