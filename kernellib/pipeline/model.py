from kernellib.models.kernels import kernel_fn_factory, KernelParams
from kernellib.models.gps import gp_model_factory


def run_model_pipeline(config):

    # init kernel params
    kernel_params = KernelParams().init_from_config(config.model.kernel)

    # initialize kernel
    kernel = kernel_fn_factory(kernel_params.kernel_fn, kernel_params)

    # init gp params
    model, likelihood = gp_model_factory(config.model, kernel)

    return model, likelihood
