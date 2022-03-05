import gpytorch
import torch
import torch.utils.data as data_utils
from tqdm import tqdm, trange


def gp_predict(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
        preds = model(test_x)
    y_mu = preds.mean.detach().numpy()
    y_var = preds.variance.detach().numpy()

    return y_mu, y_var


def gp_batch_predict(model, likelihood, test_x, n_batches: int = 100):
    model.eval()
    likelihood.eval()

    test_ds = data_utils.TensorDataset(test_x)
    test_dl = data_utils.DataLoader(test_ds, batch_size=n_batches, drop_last=False)

    y_mus, y_vars = [], []
    with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
        for X_batch in test_dl:

            preds = model(X_batch[0])

            y_mus.append(preds.mean.squeeze())
            y_vars.append(preds.variance.squeeze())

    y_mu = torch.cat(y_mus).detach().numpy()
    y_var = torch.cat(y_vars).detach().numpy()

    return y_mu, y_var


def gp_samples(model, likelihood, test_x, n_samples: int = 100):

    with torch.no_grad():
        preds = model(test_x)
        y_samples = preds.rsample(sample_shape=torch.Size((n_samples,)))

    y_samples = y_samples.detach().numpy()

    return y_samples


def fit_gp_model(
    train_x, train_y, model, likelihood, optimizer=None, loss_fn=None, n_iterations=100
):

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    if optimizer is None:
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    if loss_fn is None:
        # "Loss" for GPs - the marginal log likelihood
        loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []

    for i in (pbar := trange(n_iterations)):

        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -loss_fn(output, train_y)
        loss.backward()
        pbar.set_description(
            "Iter %d/%d - Loss: %.3f" % (i + 1, n_iterations, loss.item())
        )
        losses.append(loss.item())
        optimizer.step()

    return losses, model, likelihood
