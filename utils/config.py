import os
import torch.nn as nn

DEFAULT_DCNN_2D = {'num_of_layers': 10,
                   'kernels': [3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
                   'channels': [32, 32, 32, 32, 32, 32, 32, 32, 192, 3],  # NOTE: last channel is num_of_classes
                   'dilation': [1, 1, 2, 4, 8, 16, 32, 1, 1, 1],
                   'stride': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   'batch_norm': [False, False, False, False, False, False, False, True, True, False],
                   'non_linearity': [True, True, True, True, True, True, True, True, True, False],
                   'dropout': [0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.],
                   'loss_function': nn.NLLLoss2d,
                   'output': nn.LogSoftmax
                   }


class BaseConfig(object):

    def __init__(self):

        # default data directory
        # remember to ADD env variable REPO_PATH on machine. REPO_PATH=<absolute path to repository >
        self.root_dir = self.get_rootpath()
        self.data_dir = os.path.join(self.root_dir, "data/HVSMR2016/")
        self.log_root_path = "logs"
        self.figure_path = "figures"
        self.stats_path = "stats"
        self.checkpoint_path = "checkpoints"
        self.logger_filename = "run_out.log"
        # standard image name
        self.dflt_image_name = "*image*"
        self.dflt_label_name = "*label*"
        # used as filename for the logger
        self.logger_filename = "dcnn_run.log"
        # directory for models
        self.model_path = "models"
        self.numpy_save_filename = "aug_"

        # optimizer
        self.optimizer = "adam"

        # normalization method "normalize or rescaling"
        self.norm_method = "normalize"

        # padding to left and right of the image in order to reach the final image size for classification
        self.pad_size = 65

        # class labels
        self.class_lbl_background = 0
        self.class_lbl_myocardium = 1
        self.class_lbl_bloodpool = 2

        # noise threshold
        self.noise_threshold = 0.01

    def copy_from_object(self, obj):

        for key, value in obj.__dict__.iteritems():
            self.__dict__[key] = value

    def get_rootpath(self):
        return os.environ.get("REPO_PATH", os.environ.get('HOME'))

    def datapath(self, dataset=None):
        return self.get_datapath(dataset)

    def get_datapath(self, dataset=None):
        if dataset is None:
            return os.environ.get("PYTHON_DATA_FOLDER", "data")
        env_variable = "PYTHON_DATA_FOLDER_%s" % dataset.upper()
        return os.environ.get(env_variable, "data")


config = BaseConfig()
