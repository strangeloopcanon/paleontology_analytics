import argparse

from src._logging import get_logger
from src.acquisition.pbdb import fetch_pbdb_occurrences
from src.acquisition.neotoma import fetch_neotoma_data
from src.normalization.normalize import normalize_pbdb, normalize_neotoma, merge_datasets
from src.analysis.build_dashboard import build_dashboard_assets

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Paleontology Data Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    download_parser = subparsers.add_parser("download", help="Download data from PBDB")
    download_parser.add_argument("--interval", type=str, default="Cambrian,Cretaceous", help="Time interval (e.g., 'Cambrian,Cretaceous')")
    download_parser.add_argument("--source", type=str, default="pbdb", choices=["pbdb", "neotoma"], help="Source database")
    download_parser.add_argument("--output", type=str, default="data/raw", help="Output directory")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize data")
    normalize_parser.add_argument("--source", type=str, default="pbdb", choices=["pbdb", "neotoma", "merge"], help="Source to normalize or merge")
    normalize_parser.add_argument("--input", type=str, default="data/raw", help="Input directory (or processed for merge)")
    normalize_parser.add_argument("--output", type=str, default="data/processed", help="Output directory")

    analyze_parser = subparsers.add_parser("analyze", help="Build static dashboard assets (JSON files for the Vercel site)")
    analyze_parser.add_argument("--input", type=str, default="data/processed/merged_occurrences.parquet", help="Input parquet path")
    analyze_parser.add_argument("--dashboard-dir", type=str, default="dashboard", help="Output directory for dashboard JSON files")

    args = parser.parse_args()

    if args.command == "download":
        logger.info(
            "starting_download",
            extra={"source": args.source, "interval": getattr(args, "interval", None), "output_dir": args.output},
        )
        if args.source == "pbdb":
            fetch_pbdb_occurrences(interval=args.interval, output_dir=args.output)
        elif args.source == "neotoma":
            fetch_neotoma_data(output_dir=args.output)
    elif args.command == "normalize":
        logger.info(
            "starting_normalize",
            extra={"source": args.source, "input_dir": args.input, "output_dir": args.output},
        )
        if args.source == "pbdb":
            normalize_pbdb(input_dir=args.input, output_dir=args.output)
        elif args.source == "neotoma":
            normalize_neotoma(input_dir=args.input, output_dir=args.output)
        elif args.source == "merge":
            merge_datasets(input_dir=args.input, output_dir=args.output)
    elif args.command == "analyze":
        logger.info(
            "starting_dashboard_build",
            extra={"input_path": args.input, "dashboard_dir": args.dashboard_dir},
        )
        build_dashboard_assets(data_path=args.input, dashboard_dir=args.dashboard_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
