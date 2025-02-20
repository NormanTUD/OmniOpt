import sys
import hiplot as hip
import pandas as pd
import os

csv_file = sys.argv[1]
output_file = sys.argv[2]

csv_file_stripped = csv_file + "_stripped.csv"

enabled_titles = os.getenv("enabled_titles", "")
keep_cols = enabled_titles.split(";")
cols = list(pd.read_csv(csv_file, nrows =1, sep=';'))
f = pd.read_csv(csv_file, sep=";", usecols = [i for i in cols if i in keep_cols])
f.to_csv(csv_file_stripped, index=False)

iris_hiplot = hip.Experiment.from_csv(csv_file_stripped)
_ = iris_hiplot.to_html(output_file)
