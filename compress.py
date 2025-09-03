import pandas as pd

# Load the large CSV file
df = pd.read_csv('creditcard.csv')

# Compress and save to a GZIP-compressed CSV
df.to_csv('compressed_creditcard.csv.gz', index=False, compression='gzip')

import os

file_path = "compressed_creditcard.csv.gz"
size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"Compressed file size: {size_mb:.2f} MB")