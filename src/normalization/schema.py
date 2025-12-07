# Canonical Schema Definition

OCCURRENCE_SCHEMA = {
    "occurrence_id": "string",
    "scientific_name": "string",
    "rank": "string",
    "max_ma": "float64",
    "min_ma": "float64",
    "mid_ma": "float64",
    "lat": "float64",
    "lng": "float64",
    "phylum": "string",
    "class": "string",
    "order": "string",
    "family": "string",
    "genus": "string",
    "environment": "string",
    "source_db": "string",
    "reference_no": "string",
    "primary_reference": "string"
}

# Mapping from PBDB columns to Canonical Schema
PBDB_MAPPING = {
    "occurrence_no": "occurrence_id",
    "identified_name": "scientific_name",
    "identified_rank": "rank",
    "max_ma": "max_ma",
    "min_ma": "min_ma",
    "lat": "lat",
    "lng": "lng",
    "phylum": "phylum",
    "class": "class",
    "order": "order",
    "family": "family",
    "genus": "genus",
    "environment": "environment",
    "reference_no": "reference_no",
    "primary_reference": "primary_reference"
}
