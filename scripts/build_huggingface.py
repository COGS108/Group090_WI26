"""Runner script: clean HuggingFace data and write to 02-processed/."""
from modules.cleaning.huggingface import clean_huggingface

if __name__ == "__main__":
    clean_huggingface()
