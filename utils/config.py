import os


class BaseConfig(object):

    def __init__(self):

        # default data directory
        self.root_dir = "/home/jorg/repository/dcnn_mri_seg"
        self.data_dir = os.path.join(self.root_dir, "data/HVSMR2016/")
        self.log_root_path = "logs"
        self.figure_path = "figures"
        self.logger_filename = "run_out.log"
        # standard image name (Sunnybrook have all the same but in different directories
        self.dflt_image_name = "*image*"
        self.dflt_label_name = "*label*"
        # used as filename for the logger
        self.logger_filename = "dcnn_run.log"
        # directory for models
        self.model_path = "models"
        self.numpy_save_filename = "aug_"

        # optimizer
        self.optimizer = "adam"


config = BaseConfig()
