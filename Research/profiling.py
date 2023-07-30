from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_parquet('/home/jagac/projects/taxi-tip-mlapp/Research/yellow_tripdata_2023-04.parquet')
profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("eda_report.html")
