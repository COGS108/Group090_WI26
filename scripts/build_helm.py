"""Runner script: compile HELM parquet and write to 02-processed/."""
from modules.cleaning.helm import build_helm_parquet

if __name__ == "__main__":
    build_helm_parquet()
