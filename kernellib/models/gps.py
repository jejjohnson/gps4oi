from kernellib.models.exact import init_exact_gp

def gp_model_factory(gp_params, kernel_params):
    
    if gp_params.model == "exact":
        return init_exact_gp(gp_params, kernel_params)
    
    else:
        raise ValueError(f"Unrecognized kernel option: {kernel}")
        