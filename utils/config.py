class BaseConfig(object):

    def __init__(self):

        # default data directory
        self.data_dir = "/mnt/u_drive/u_share/data/la_sunnybrook/"
        # standard image name (Sunnybrook have all the same but in different directories
        self.dflt_image_name = "image"
        self.dflt_label_name = "gt_binary"
        # used as filename for the logger
        self.logger_filename = "dcnn_run.log"
        # directory for models
        self.model_path = "models"


config = BaseConfig()
