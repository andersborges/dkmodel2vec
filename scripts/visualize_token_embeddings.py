import argparse
from pathlib import Path

import pandas as pd
import umap

from model2vec import StaticModel


def main():
    parser = argparse.ArgumentParser(
        description="Project token space to 2D for visualization."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to Model2vec model from which to fit UMAP model and reduce dimensions. ",
    )
    parser.add_argument(
        "--outpath", type=Path, required=True, help="Path to dump parquet to"
    )

    parser.add_argument(
        "--other-model-path",
        type=Path,
        required=False,
        help="Path to second Model2vec model for comparison. Projections will be added as columns.",
    )

    parser.add_argument(
        "--diff-outpath",
        type=Path,
        required=False,
        help="If set, will also add diff_projection_x and diff_projection_y columns showing the difference.",
    )

    args = parser.parse_args()

    m2v = StaticModel.from_pretrained(args.model_path)
    reducer = umap.UMAP()

    print(f"Embedding shape: {m2v.embedding.shape}")
    print(f"Fitting and transforming...")
    embeds2d = reducer.fit_transform(m2v.embedding)

    print("Sorting...")
    tokens = sorted(m2v.tokenizer.get_vocab().items(), key=lambda x: x[1])
    tokens = [
        {
            "index": t[1],
            "text": t[0],
            "embedding": e_n,
            "projection_x": e_x,
            "projection_y": e_y,
        }
        for t, e_n, e_x, e_y in zip(
            tokens, m2v.embedding, embeds2d[:, 0], embeds2d[:, 1]
        )
    ]

    df = pd.DataFrame(tokens)

    if args.other_model_path:
        print(f"\nLoading other model from {args.other_model_path}...")
        other_m2v = StaticModel.from_pretrained(args.other_model_path)

        # Check vocabularies
        vocab1 = m2v.tokenizer.get_vocab()
        vocab2 = other_m2v.tokenizer.get_vocab()

        if vocab1 != vocab2:
            print("WARNING: Vocabularies differ between models!")
            only_in_model1 = set(vocab1.keys()) - set(vocab2.keys())
            only_in_model2 = set(vocab2.keys()) - set(vocab1.keys())

            if only_in_model1:
                print(f"  Tokens only in model 1: {len(only_in_model1)} tokens")
                if len(only_in_model1) <= 10:
                    print(f"    {only_in_model1}")

            if only_in_model2:
                print(f"  Tokens only in model 2: {len(only_in_model2)} tokens")
                if len(only_in_model2) <= 10:
                    print(f"    {only_in_model2}")

        # Transform other model embeddings using the fitted reducer
        print("Transforming other model embeddings...")
        other_embeds2d = reducer.transform(other_m2v.embedding)

        # Create lookup dictionary for other model projections
        other_vocab_to_projection = {}
        for token_text, token_idx in other_m2v.tokenizer.get_vocab().items():
            other_vocab_to_projection[token_text] = {
                "x": other_embeds2d[token_idx, 0],
                "y": other_embeds2d[token_idx, 1],
            }

        # Add columns for other model projections and redundancy flag
        df["other_projection_x"] = df["text"].map(
            lambda t: other_vocab_to_projection.get(t, {}).get("x", None)
        )
        df["other_projection_y"] = df["text"].map(
            lambda t: other_vocab_to_projection.get(t, {}).get("y", None)
        )
        df["is_redundant"] = df["text"].map(
            lambda t: 0 if t in other_vocab_to_projection else 1
        )
        df["only_in_other"] = (
            0  # All tokens in df are from model 1, so none are "only in other"
        )

        print(
            f"Tokens marked as redundant (in model 1 but not in other): {df['is_redundant'].sum()}"
        )

        # Add tokens that are only in the other model
        only_in_other_tokens = set(vocab2.keys()) - set(vocab1.keys())
        if only_in_other_tokens:
            print(
                f"Adding {len(only_in_other_tokens)} tokens that are only in the other model..."
            )

            other_only_rows = []
            for token_text in only_in_other_tokens:
                token_idx = vocab2[token_text]
                other_only_rows.append(
                    {
                        "index": None,  # No index in model 1
                        "text": token_text,
                        "projection_x": None,  # No projection in model 1
                        "projection_y": None,
                        "other_projection_x": other_embeds2d[token_idx, 0],
                        "other_projection_y": other_embeds2d[token_idx, 1],
                        "is_redundant": 0,  # Not redundant - it exists in other model
                        "only_in_other": 1,  # Mark as only in other model
                    }
                )

            # Append these tokens to the dataframe
            other_only_df = pd.DataFrame(other_only_rows)
            df = pd.concat([df, other_only_df], ignore_index=True)
            print(f"Total tokens after adding other-only tokens: {len(df)}")

        # Compute difference columns if requested
        if args.diff_outpath:
            print("Computing projection differences...")
            df["diff_projection_x"] = df["other_projection_x"] - df["projection_x"]
            df["diff_projection_y"] = df["other_projection_y"] - df["projection_y"]

            # Report statistics
            non_redundant = df[df["is_redundant"] == 0]
            print(f"Computed differences for {len(non_redundant)} non-redundant tokens")

    # Write the final dataframe
    print(f"Writing to parquet: {args.outpath}")
    if args.other_model_path:
        # Include other model columns if they exist
        columns_to_save = [
            "index",
            "text",
            "projection_x",
            "projection_y",
            "other_projection_x",
            "other_projection_y",
            "is_redundant",
            "only_in_other",
        ]
        if args.diff_outpath:
            columns_to_save.extend(["diff_projection_x", "diff_projection_y"])
        df[columns_to_save].to_parquet(args.outpath)
    else:
        df[["index", "text", "projection_x", "projection_y"]].to_parquet(args.outpath)

    print("Done!")
    return


if __name__ == "__main__":
    main()
