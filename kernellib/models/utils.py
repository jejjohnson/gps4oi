import gpytorch
import torch
import torch.utils.data as data_utils
from tqdm import tqdm, trange


def gp_predict(model, likelihood, test_x):
    model.eval()
    likelihood.eval()

    if torch.cuda.is_available():
        test_x = test_x.cuda()

    with gpytorch.settings.max_preconditioner_size(
        10
    ), gpytorch.settings.fast_pred_var(), torch.no_grad():
        preds = model(test_x)

    try:
        y_mu = preds.mean.detach().numpy()
        y_var = preds.variance.detach().numpy()
    except TypeError:
        y_mu = preds.mean.detach().cpu().numpy()
        y_var = preds.variance.detach().cpu().numpy()

    return y_mu, y_var


def gp_batch_predict(
    model,
    likelihood,
    test_x,
    n_batches: int = 100,
    cached: bool = True,
    preconditioner_size: int = 100,
):
    model.eval()
    likelihood.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        test_x = test_x.cuda()

    test_ds = data_utils.TensorDataset(test_x)
    test_dl = data_utils.DataLoader(test_ds, batch_size=n_batches, drop_last=False)

    y_mus, y_vars = [], []
    if not cached:
        with gpytorch.settings.max_preconditioner_size(
            preconditioner_size
        ), gpytorch.settings.fast_pred_var(), torch.no_grad():
            for X_batch in test_dl:

                preds = model(X_batch[0])

                y_mus.append(preds.mean.squeeze())
                y_vars.append(preds.variance.squeeze())
    else:
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            for X_batch in test_dl:

                preds = model(X_batch[0])

                y_mus.append(preds.mean.squeeze())
                y_vars.append(preds.variance.squeeze())

    try:
        y_mu = torch.cat(y_mus).detach().numpy()
        y_var = torch.cat(y_vars).detach().numpy()
    except TypeError:
        y_mu = torch.cat(y_mus).detach().cpu().numpy()
        y_var = torch.cat(y_vars).detach().cpu().numpy()

    return y_mu, y_var


def gp_samples(
    model,
    likelihood,
    test_x,
    n_samples: int = 100,
    cached: bool = True,
    preconditioner_size: int = 100,
):

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        test_x = test_x.cuda()

    if not cached:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.fast_pred_samples():
                preds = model(test_x)
                y_samples = preds.rsample(sample_shape=torch.Size((n_samples,)))
    else:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.fast_pred_samples():
                preds = model(test_x)
                y_samples = preds.rsample(sample_shape=torch.Size((n_samples,)))

    try:
        y_samples = y_samples.detach().numpy()
    except TypeError:
        y_samples = y_samples.detach().cpu().numpy()

    return y_samples


def fit_gp_model(
    train_x,
    train_y,
    model,
    likelihood,
    optimizer=None,
    loss_fn=None,
    n_iterations=100,
    wandb_logger=None,
    lr: float=0.01,
    scheduler=None,
    
):
    model = model(train_x, train_y)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    if torch.cuda.is_available():
        print("Training on GPU")
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()
    else:
        print("Training on CPU")

    if optimizer is None:
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=n_iterations,
                last_epoch=-1,
                eta_min=0
            )

    if loss_fn is None:
        # "Loss" for GPs - the marginal log likelihood
        loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []

    for i in (pbar := trange(n_iterations)):

        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss
        loss = -loss_fn(output, train_y)
        # backprop derivatives
        loss.backward()
        pbar.set_description(
            "Iter %d/%d - Loss: %.3f" % (i + 1, n_iterations, loss.item())
        )
        losses.append(loss.item())
        if wandb_logger is not None:
            wandb_logger.log({"nll": loss.item(), "iteration": i})
        # update the params
        optimizer.step()
        scheduler.step()

    return losses, model, likelihood
