from src.normalization.normalize import normalize_pbdb, normalize_neotoma, merge_datasets

print("Running PBDB normalization...")
normalize_pbdb()

print("Running Neotoma normalization...")
normalize_neotoma()

print("Merging datasets...")
merge_datasets()

print("Done.")
