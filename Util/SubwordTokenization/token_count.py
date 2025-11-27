#!/usr/bin/env python3
"""
Count the number of tokens in a text file with tiktoken.

Usage:
    python count_tokens.py path/to/file.txt [--model gpt-3.5-turbo]

Dependencies:
    pip install tiktoken
"""

import argparse
import sys

import tiktoken


def count_tokens_in_file(file_path: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Return the total number of tokens that the file would have when encoded for the chosen model.

    Parameters
    ----------
    file_path : str
        Path to the input text file.
    model : str, optional
        OpenAI model name (or any name that tiktoken knows). Defaults to "gpt-3.5-turbo".
        For GPT‑4 you might use "gpt-4o" or "gpt-4o-mini", etc.

    Returns
    -------
    int
        Total token count.
    """
    # Get the appropriate encoding for the model.
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Error: Unknown model '{model}'. Falling back to 'cl100k_base'.", file=sys.stderr)
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0

    # Open the file in text mode (utf‑8).  Adjust encoding if needed.
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            # Encode the line and count the tokens.
            total_tokens += len(encoding.encode(line))

    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Count tiktoken tokens in a file.")
    parser.add_argument("file", help="Path to the file you want to analyse.")
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI model name (e.g. gpt-3.5-turbo, gpt-4o). "
        "Defaults to 'gpt-3.5-turbo'.",
    )
    args = parser.parse_args()

    try:
        tokens = count_tokens_in_file(args.file, args.model)
    except FileNotFoundError:
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"Encoding error while reading the file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Total tokens in '{args.file}': {tokens}")


if __name__ == "__main__":
    main()
