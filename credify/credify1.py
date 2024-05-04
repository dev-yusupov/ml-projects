import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Read the original dataset
data = pd.read_csv("data.csv")

# Create metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# Filter the data to only include instances of class 0
data_class_0 = data[data['RESPONSE'] == 0]

# Initialize CTGAN synthesizer
synthesizer = CTGANSynthesizer(metadata=metadata)

# Fit the synthesizer to the data of class 0
synthesizer.fit(data_class_0)

# Generate synthetic data for class 0
synthetic_data_class_0 = synthesizer.sample(num_rows=700)

# Concatenate the original instances of class 1 with the synthetic instances of class 0
synthetic_data = pd.concat([data[data['RESPONSE'] == 1], synthetic_data_class_0], ignore_index=True)

# Save synthetic data to a CSV file
synthetic_data.to_csv("synthetic_data.csv", index=False)
