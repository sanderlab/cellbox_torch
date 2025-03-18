import json


class Config(object):
    """
    An object reading in a json file of model configuration
    """
    def __init__(self, config_file):

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        self.prot_data_file = config_dict["prot_data_file"]
        self.prot_info_file = config_dict["prot_info_file"]
        self.last_prot_index = config_dict["last_prot_index"]
        self.first_prot_index = config_dict["first_prot_index"]
        self.id_key = config_dict["id_key"]
        self.metadata_cols = config_dict["metadata_cols"]
        self.nan_thres = config_dict["nan_thres"]
        self.target_col = config_dict["target_col"]
        self.file_prefix = config_dict["file_prefix"]
        self.file_path = config_dict["file_path"]
        self.intensity_upper = config_dict['intensity_upper']
        self.intensity_lower = config_dict['intensity_lower']
        self.dispersion_lower = config_dict['dispersion_lower']
        self.__dict__.update(config_dict)
