import pandas as pd
import time
from validation_engine import ValidationEngine

engine = ValidationEngine()
active_models = ["bounds", "pm", "surrogate", "figs"]
df = pd.read_csv("SWATDatasets/attack.csv", nrows=1000)

start = time.time()
for i, row in df.iterrows():
    engine.evaluate_row(row, active_models=active_models)
end = time.time()
print(f"Time for 1000 rows: {end - start:.2f} seconds")
