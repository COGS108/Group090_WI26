"""Runner script: clean Epoch AI data and write to 02-processed/."""
from modules.cleaning.epoch import clean_epoch

if __name__ == "__main__":
    clean_epoch()
