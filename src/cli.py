import argparse
import sys
from src.acquisition.pbdb import fetch_pbdb_occurrences
from src.acquisition.neotoma import fetch_neotoma_data
from src.normalization.normalize import normalize_pbdb, normalize_neotoma, merge_datasets
from src.analysis.basic_stats import plot_diversity_curve, plot_map
from src.analysis.advanced_stats import plot_biogeographic_network, calculate_sqs_diversity
from src.analysis.sota_stats import analyze_biogeographic_dynamics
from src.analysis.ml_extinction import run_ml_extinction_analysis

def main():
    parser = argparse.ArgumentParser(description="Paleontology Data Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download data from PBDB")
    download_parser.add_argument("--interval", type=str, default="Cambrian,Cretaceous", help="Time interval (e.g., 'Cambrian,Cretaceous')")
    download_parser.add_argument("--source", type=str, default="pbdb", choices=["pbdb", "neotoma"], help="Source database")
    download_parser.add_argument("--output", type=str, default="data/raw", help="Output directory")

    # Normalize command
    normalize_parser = subparsers.add_parser("normalize", help="Normalize data")
    normalize_parser.add_argument("--source", type=str, default="pbdb", choices=["pbdb", "neotoma", "merge"], help="Source to normalize or merge")
    normalize_parser.add_argument("--input", type=str, default="data/raw", help="Input directory (or processed for merge)")
    normalize_parser.add_argument("--output", type=str, default="data/processed", help="Output directory")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis")
    analyze_parser.add_argument("--type", type=str, default="basic", choices=["basic", "advanced", "sota", "ml"], help="Type of analysis")
    analyze_parser.add_argument("--input", type=str, default="data/processed/merged_occurrences.parquet", help="Input file")
    analyze_parser.add_argument("--output", type=str, default="data/analysis", help="Output directory")

    args = parser.parse_args()

    if args.command == "download":
        if args.source == "pbdb":
            fetch_pbdb_occurrences(interval=args.interval, output_dir=args.output)
        elif args.source == "neotoma":
            fetch_neotoma_data(output_dir=args.output)
    elif args.command == "normalize":
        if args.source == "pbdb":
            normalize_pbdb(input_dir=args.input, output_dir=args.output)
        elif args.source == "neotoma":
            normalize_neotoma(input_dir=args.input, output_dir=args.output)
        elif args.source == "merge":
            merge_datasets(input_dir=args.output, output_dir=args.output)
    elif args.command == "analyze":
        if args.type == "basic":
            plot_diversity_curve(data_path=args.input, output_dir=args.output)
            plot_map(data_path=args.input, output_dir=args.output)
        elif args.type == "advanced":
            plot_biogeographic_network(data_path=args.input, output_dir=args.output)
            calculate_sqs_diversity(data_path=args.input, output_dir=args.output)
        elif args.type == "sota":
            analyze_biogeographic_dynamics(data_path=args.input, output_dir=args.output)
        elif args.type == "ml":
            run_ml_extinction_analysis(data_path=args.input, output_dir=args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
