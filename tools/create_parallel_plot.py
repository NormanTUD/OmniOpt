import sys
import hiplot as hip
import pandas as pd

csv_file = sys.argv[1]
output_file = sys.argv[2]

csv_file_stripped = csv_file + "_stripped.csv"

drop_cols = ["time", "endtime", "hostname", "logfile_path", "loss", "starttime"]
cols = list(pd.read_csv(csv_file, nrows =1, sep=';'))
f = pd.read_csv(csv_file, sep=";", usecols = [i for i in cols if i not in drop_cols])
f.to_csv(csv_file_stripped, index=False)

iris_hiplot = hip.Experiment.from_csv(csv_file_stripped)
_ = iris_hiplot.to_html(output_file)
