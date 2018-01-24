from models.dilated_cnn import BaseDilated2DCNN


def load_model(exper):

    if exper.run_args.model == 'dcnn':
        exper.logger.info("Creating new model {}".format(exper.run_args.model))
        model = BaseDilated2DCNN(use_cuda=exper.run_args.cuda)
    else:
        raise ValueError("{} name is unknown and hence cannot be created".format(exper.run_args.model))

    return model