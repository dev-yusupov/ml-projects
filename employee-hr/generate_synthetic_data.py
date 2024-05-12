import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

df = pd.read_csv("data.csv")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

synthesizer = CTGANSynthesizer(metadata=metadata, epochs=500)
synthesizer.fit(df)

synthetic_data = synthesizer.sample(num_rows=50000)

synthetic_data.to_csv("synthetic_data.csv")