# src/nlp_imdb/cli/main.py
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP IMDb Sentiment Analysis CLI")

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    if args.config:
        print(f"Using config file: {args.config}")
    else:
        print("No config file provided.")

    sys.exit(0)


if __name__ == "__main__":
    main()
