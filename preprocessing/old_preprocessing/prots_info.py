import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# prots stats
prots_info = (
    pd.read_csv("data.csv")
    .rename(columns={"Unnamed: 0": "proteins"}).set_index("proteins")
)
prots_info.head()
