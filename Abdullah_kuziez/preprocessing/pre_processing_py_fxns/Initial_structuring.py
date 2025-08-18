"""
README:
This file contains functions that shape the data into an intelligble format for the rest of the downstream analysis
in general, these functions will change for different datasets. if properly adjusted for different datasets, the 
downstream filtering and plotting functions should remain the exact same:

End structure of the data:
data_by_cell_line: a dictionary where the keys are the cell lines, and the values are dataframes with the following structure:
Rows=experiments, columns=proteins+phenotypes+metadata (noted with the regex prefix meta_!!)

"""













