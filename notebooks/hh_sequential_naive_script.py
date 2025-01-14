import os
from time import time_ns; from datetime import datetime
import numpy as np; import pandas as pd; import matplotlib.pyplot as plt
import torch
from corner import corner
from omegaconf import OmegaConf as OC

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from corner import corner
from matplotlib import pyplot as plt

# hodgkin-huxley
from sourcerer.utils import scale_tensor
from sourcerer.hh_simulator import EphysModel # hh simulator
from sourcerer.hh_utils import HHSurro, PRIOR_MAX, PRIOR_MIN, DEF_RESTRICTED

from sourcerer.fit_surrogate import (
    create_train_val_dataloaders,
    fit_conditional_normalizing_flow,
    train_val_split,
    create_dataloader
)
from sourcerer.real_nvp import (
    Sampler,
    RealNVPs,
    TemperedUniform, # a little slanted uniform to make log-density well defined
    VariableTemperedUniform,
    kozachenko_leonenko_estimator,
)

from sourcerer.sbi_classifier_two_sample_test import c2st_scores

# other simulators
from sourcerer.simulators import (
    InverseKinematicsSimulator,
    LotkaVolterraSimulator,
    SIRSimulator,
    SLCPSimulator,
    TwoMoonsSimulator,
    GaussianMixtureSimulator,
)

# wasserstein source
from sourcerer.sliced_wasserstein import sliced_wasserstein_distance
from sourcerer.wasserstein_estimator import train_source

# saving utils
from sourcerer.utils import (
    save_cfg_as_yaml,
    save_fig,
    save_numpy_csv,
    save_state_dict,
    script_or_command_line_cfg,
    set_seed,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f'run id: {run_id}')
print(f"using {device}")


# Define config
# NOTE: These overrides only take effect if this script is run interactively
local_overrides = [
    "base.tag=debug",
    "base.folder=sequential_hh",
    "source=wasserstein_hh",
    "source.fin_lambda=0.25",
    "surrogate=hh_train_surrogate", #load_hh_surrogate
    # "surrogate.ydim=5",
    # "surrogate.surrogate_path=./hh_surrogate_epochs_100.pt",
]

cfg = script_or_command_line_cfg(
    config_name="config",
    config_path="../conf",
    local_overrides=local_overrides,
    name=__name__,
)

assert cfg.base.tag is not None
assert cfg.base.folder is not None

print(OC.to_yaml(cfg))

save_cfg_as_yaml(
    cfg,
    f"{cfg.base.tag}_cfg.yaml",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)


# set seed
if cfg.base.seed is None:
    random_random_seed = np.random.randint(2**16)
    set_seed(random_random_seed)
    save_numpy_csv(
        np.array([random_random_seed], dtype=int),
        file_name=f"{cfg.base.tag}_seed.csv",
        folder=cfg.base.folder,
        base_path=cfg.base.base_path,
    )
    print(f"seed: {random_random_seed}")
else:
    set_seed(cfg.base.seed)


# notation here: θ (which is x here), and x (which is x here) 
print(f"theta/parameter dimension: {cfg.surrogate.xdim}")
print(f"x/observation dimension: {cfg.surrogate.ydim}")
theta_labels = [fr"$\theta_{i+1}$" for i in range(cfg.surrogate.xdim)]
y_labels = [fr"$x_{i+1}$" for i in range(cfg.surrogate.ydim)]


# Define surrogate model
surrogate = HHSurro(hidden_layer_dim=cfg.surrogate.hidden_layer_dim,
                    xdim=cfg.surrogate.xdim,
                    ydim=cfg.surrogate.ydim
                   ).to(device)

surro_optimizer = optim.Adam(surrogate.parameters(),
                             lr=cfg.surrogate.surrogate_lr, # 5e-4
                             weight_decay=cfg.surrogate.surrogate_weight_decay # 1e-5
                            )

assert cfg.surrogate.self == 'hh_train_surrogate'


# load data
def fetch_hh_data_full(path='./full_batch.npz'):
    sim_data = np.load(path)
    theta = sim_data["theta"]
    stats = sim_data["stats"]

    # Remove undefined simulations (either only the 5 out of 15mil that completelly fail, or the ones without undefined stats)
    keeping = (~np.isnan(np.mean(stats, axis=1))) & (~np.isinf(np.mean(stats, axis=1)))
    moment_keeping = (~np.isnan(stats[:, 22])) & (~np.isinf(stats[:, 22]))  # 22 is a moment
    #print(theta[~moment_keeping, :])  # 5 sims out of 15mil completely fail
    # print(np.where(moment_keeping == 0)[0])

    stats = stats[moment_keeping, :]  # delete Nan simulations that completely fail
    theta = theta[moment_keeping, :]  # delete Nan simulations that completely fail
    
    stats = stats[:, DEF_RESTRICTED]
    # reverse engineer unnecessarily undefined counts -- why??
    stats[:, :1][np.isnan(stats[:, :1])] = np.log(3)

    # standardization
    source_dim = 13
    # standardize source to range from -1 to 1
    source = scale_tensor(
        torch.from_numpy(np.float32(theta)),
        PRIOR_MIN,               # current range
        PRIOR_MAX,
        -torch.ones(source_dim), # target range
        torch.ones(source_dim),
    )
    
    stats_torch = torch.from_numpy(np.float32(stats))
    stats_mean = torch.mean(stats_torch, dim=0)
    stats_std = torch.std(stats_torch, dim=0)
    # print(stats_mean)
    # print(stats_std)
    stats_torch = (stats_torch - stats_mean) / stats_std
    
    return source, stats_torch # both normalized


params, xs = fetch_hh_data_full(path='/mnt/qb/work/macke/mwe102/sourcerer-sequential/notebooks/full_batch.npz') # note that both params and xs are standardized here
print(params.shape, xs.shape)

cfg.surrogate.num_training_samples = cfg.sequential.total_simulation_budget
number_of_sims_source = cfg.sequential.number_of_sims_source # how many x-s to train the source on? It is typically 1M, also it is to be normalized
source_dim = cfg.surrogate.xdim

# choose thetas that are outside of what source model would see (in terms of observations)
# note that here thetas are in [-1, 1], the x-s are standardized
gt_source = params[number_of_sims_source:number_of_sims_source+10_000, :]            # 10k thetas from true source distribution
gt_source_two = params[number_of_sims_source+10_000:number_of_sims_source+20_000, :] # another 10k thetas from true source distribution

# simulations from the dataset (note that these are standardized)
gt_simulator = xs[number_of_sims_source:number_of_sims_source+10_000, :]
gt_simulator_two = xs[number_of_sims_source+10_000:number_of_sims_source+20_000, :]

# we should also use our simulator, first need to move the sources to actual range
gt_source_moved = scale_tensor(gt_source, -torch.ones(source_dim), torch.ones(source_dim), PRIOR_MIN, PRIOR_MAX)
gt_source_two_moved = scale_tensor(gt_source_two, -torch.ones(source_dim), torch.ones(source_dim), PRIOR_MIN, PRIOR_MAX)

gt_source_kole = kozachenko_leonenko_estimator(gt_source_two, on_torus=False).item() # true source entropy, just compute and save
print(f"Ground truth source entropy estimate: {gt_source_kole}")
save_numpy_csv(
    np.array([gt_source_kole]),
    file_name=f"{cfg.base.tag}_gt_source_kole.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

# print("Sliced wassertein distance on groundtruth y:")
expected_distance = sliced_wasserstein_distance(
    gt_simulator.to(device),     # from theta1
    gt_simulator_two.to(device), # from theta2 (both ground truth)
    num_projections=4096,
    device=device,
).item()


print(f"Sliced wassertein distance on groundtruth y (expected/baseline swd): {expected_distance}, log: {np.log(expected_distance)}")
save_numpy_csv(np.array([expected_distance]),
               file_name=f"{cfg.base.tag}_expected_swd.csv", folder=cfg.base.folder, base_path=cfg.base.base_path)


# Define source flows
if cfg.source_model.self == "sampler":
    source = Sampler(
        xdim=cfg.surrogate.xdim,
        input_noise_dim=cfg.surrogate.xdim,
        hidden_layer_dim=cfg.source_model.hidden_layer_dim,
        num_hidden_layers=cfg.source_model.num_hidden_layers,
    )
elif cfg.source_model.self == "real_nvp":
    source = RealNVPs(
        data_dim=cfg.surrogate.xdim,
        context_dim=0,
        hidden_layer_dim=cfg.source_model.hidden_layer_dim,
        flow_length=cfg.source_model.flow_length,
        low=cfg.simulator.box_domain_lower,
        high=cfg.simulator.box_domain_upper,
    )
else:
    raise ValueError

source = source.to(device)

cfg.source.lambda_steps = 8000  #11000, limit total epochs to 12k
# Train source model
optimizer = torch.optim.Adam(
    source.parameters(),
    lr=cfg.source.learning_rate,
    weight_decay=cfg.source.weight_decay,
)



# This is the scheduled values of lambda - first we linearly decay from lambda=1.0 (only entropy) until a minimum value of lambda. Then we stay at that value of lambda for more iterations.
schedule_org = torch.cat(
    [
        torch.ones(cfg.source.pretraining_steps),
        torch.linspace(
            1.0,
            cfg.source.fin_lambda,
            cfg.source.linear_decay_steps,
        ),
        cfg.source.fin_lambda * torch.ones(cfg.source.lambda_steps),
    ]
)

first_iteration_schedule = schedule_org.clone().detach()
next_iters_schedule = cfg.source.fin_lambda * torch.ones(cfg.source.pretraining_steps + cfg.source.linear_decay_steps + cfg.source.lambda_steps)

BUDGET_SCHEDULER = np.ones(cfg.sequential.number_of_iterations) / cfg.sequential.number_of_iterations
BUDGET_PER_ITERATION = (BUDGET_SCHEDULER * cfg.sequential.total_simulation_budget).astype(int) # divide budget uniformly

if cfg.sequential.frontloading:
    if cfg.sequential.front_frac > (1 / cfg.sequential.number_of_iterations):
        BUDGET_PER_ITERATION[0] = cfg.sequential.total_simulation_budget * cfg.sequential.front_frac # "frontload"
        remaining = cfg.sequential.total_simulation_budget * (1-cfg.sequential.front_frac)
        BUDGET_PER_ITERATION[1:] = remaining // (cfg.sequential.number_of_iterations-1) # divide remaining budget uniformly
    
print(BUDGET_PER_ITERATION)


## this is the initial distribution to sample unnormalized theta (!!!) from to train the surrogate
box_domain = VariableTemperedUniform(lower_bound=PRIOR_MIN, upper_bound=PRIOR_MAX)
params_gpu = params.to(device)


def generate_data_from_box_prior(number, xs_all, device):

    source_dim=13
    from sourcerer.hh_utils import PRIOR_MAX, PRIOR_MIN
    from sourcerer.utils import scale_tensor
    from sourcerer.real_nvp import VariableTemperedUniform

    box_domain = VariableTemperedUniform(lower_bound=PRIOR_MIN, upper_bound=PRIOR_MAX)
    surro_theta = box_domain.sample(number).detach()
    surro_theta_norm = scale_tensor(surro_theta, PRIOR_MIN, PRIOR_MAX, -torch.ones(source_dim), torch.ones(source_dim))

    # nearest neighbours search
    # params_gpu = params_all.to(device)
    surro_theta_norm_gpu = surro_theta_norm.to(device)

    nns_idx = torch.zeros(surro_theta_norm.shape[0])
    for i, test_theta in enumerate(surro_theta_norm_gpu):
        test = test_theta.unsqueeze(0)
        #print(test.shape)
        dist = torch.norm(params_gpu - test, dim=1, p=2)
        #print(dist.shape)
        nns_idx[i] = dist.argmin().item()
        # print(i)
    # print(nns_idx)

    surro_push_forward_norm = xs_all[nns_idx.long()]
    return surro_theta_norm_gpu, surro_push_forward_norm.to(device)


def generate_data_from_source(domain_distribution, number, xs_all, device):

    surro_theta_norm = domain_distribution.sample(number).detach()
    # nearest neighbours search
    # params_gpu = params_all.to(device)
    surro_theta_norm_gpu = surro_theta_norm.to(device)

    nns_idx = torch.zeros(surro_theta_norm.shape[0])
    for i, test_theta in enumerate(surro_theta_norm_gpu):
        test = test_theta.unsqueeze(0)
        #print(test.shape)
        dist = torch.norm(params_gpu - test, dim=1, p=2)
        #print(dist.shape)
        nns_idx[i] = dist.argmin().item()
        # print(i)
    # print(nns_idx)

    surro_push_forward_norm = xs_all[nns_idx.long()]
    return surro_theta_norm_gpu, surro_push_forward_norm.to(device)


# # Load data and standardize to same scale as for training.
cfg.source.xo_path = './full_batch.npz'
npz = np.load(cfg.source.xo_path)
print(npz['theta'].shape); print(npz['stats'].shape)
# data = read_pickle(cfg.source.xo_path)
# print(data["X_o"].head())

full_xo_stats_np = npz["stats"]
num_xo = full_xo_stats_np.shape[0]
print(num_xo)

xo_stats_np = full_xo_stats_np[:, DEF_RESTRICTED]
xo_stats_np = xo_stats_np[np.where(~np.isnan(xo_stats_np).any(axis=1))[0]] #remove nan

xo_stats = torch.from_numpy(np.float32(xo_stats_np)) # unnormalized
print(f"after nan removal: {xo_stats.shape}")

supervised_mean = torch.tensor(
    [  # correct restricted 1 mil
        2.3512,
        -93.2657,
        -52.7358,
        278.4319,
        0.4392,
    ],
)

supervised_std = torch.tensor(
    [  # correct restricted 1 mil
        1.1922,
        20.0920,
        19.6483,
        300.1352,
        4.4579,
    ],
)
xo_stats_norm = (xo_stats - supervised_mean) / supervised_std
# xo_stats_norm = xo_stats_norm.to(device)


# take 1M samples to fit source
print(f'will train source on # {number_of_sims_source} observations')
xo_stats_norm = xo_stats_norm[:number_of_sims_source, :].to(device) # xo_stats_norm lives on GPU!!
# print(xo_stats_norm.shape[0])

train_losses = []
val_losses = []
train_source_losses = []

# surrogate quality metrics list (one value per iteration)
surro_c2sts = np.zeros(cfg.sequential.number_of_iterations)
surro_swds = np.zeros(cfg.sequential.number_of_iterations)

# source quality metrics list (one value per iteration)
source_simu_pf_swds = np.zeros(cfg.sequential.number_of_iterations)
source_simu_pf_c2sts = np.zeros(cfg.sequential.number_of_iterations)
source_surro_pf_c2sts = np.zeros(cfg.sequential.number_of_iterations) # these 2 are pfs
source_c2sts = np.zeros(cfg.sequential.number_of_iterations) # how well the source dists match? not necessary
source_entropies = np.zeros(cfg.sequential.number_of_iterations)

surro_train_domain_distribution = box_domain  # initialization

print(f"naive sequential method on hodgkin-huxley with {cfg.sequential.number_of_iterations} iters and total budget: {cfg.sequential.total_simulation_budget}")
print(f"aggregating data? {cfg.sequential.collate_last_iter_data}, varying lambda over iters? {not cfg.sequential.lambda_stays_same}")
for iteration in range(cfg.sequential.number_of_iterations):
    print(f"ITER: {iteration+1}/{cfg.sequential.number_of_iterations} fitting surrogate with {BUDGET_PER_ITERATION[iteration]} samples")

    if iteration == 0:
        # here generate data from box domain
        surro_domain, surro_push_forward = generate_data_from_box_prior(number=BUDGET_PER_ITERATION[iteration], xs_all=xs, device=device)
        (surro_train_push_forward, surro_train_domain), (surro_val_push_forward, surro_val_domain) = train_val_split(
            surro_push_forward, surro_domain
        )
        print(f"first iter tr. data θ: {surro_train_domain.shape}, x: {surro_train_push_forward.shape}")
        print(f"first iter val data θ: {surro_val_domain.shape}, x: {surro_val_push_forward.shape}")
    
    else:
        # use the current source to generate data
        surro_domain, surro_push_forward = generate_data_from_source(domain_distribution=source,
                                                                     number=BUDGET_PER_ITERATION[iteration], xs_all=xs, device=device)
        
        # for subsequent iterations either collate or not
        if cfg.sequential.collate_last_iter_data:
            # concat data to (surro_train_push_forward, surro_train_domain), (surro_val_push_forward, surro_val_domain)
            (surro_train_push_forward_new, surro_train_domain_new), (surro_val_push_forward_new, surro_val_domain_new) = train_val_split(
                surro_push_forward, surro_domain
            )
            surro_train_domain = torch.cat((surro_train_domain, surro_train_domain_new), 0)
            surro_train_push_forward = torch.cat((surro_train_push_forward, surro_train_push_forward_new), 0)

            surro_val_domain = torch.cat((surro_val_domain, surro_val_domain_new), 0)
            surro_val_push_forward = torch.cat((surro_val_push_forward, surro_val_push_forward_new), 0)
            print(f"iter:{iteration+1} tr. data θ: {surro_train_domain.shape}, x: {surro_train_push_forward.shape}")
            print(f"iter:{iteration+1} val data θ: {surro_val_domain.shape}, x: {surro_val_push_forward.shape}")
        
        else:
            (surro_train_push_forward, surro_train_domain), (surro_val_push_forward, surro_val_domain) = train_val_split(
                surro_push_forward, surro_domain
            )
            # print(f"iter:{iteration+1} tr. data θ: {surro_train_domain.shape}, x: {surro_train_push_forward.shape}")
            # print(f"iter:{iteration+1} val data θ: {surro_val_domain.shape}, x: {surro_val_push_forward.shape}")

    train_dataset = create_dataloader(surro_train_push_forward, surro_train_domain)
    val_dataset = create_dataloader(surro_val_push_forward, surro_val_domain)

    train_loss, val_loss = fit_conditional_normalizing_flow(
        network=surrogate,
        optimizer=surro_optimizer,
        training_dataset=train_dataset,
        validation_dataset=val_dataset,
        nb_epochs=300, #//NUMBER_OF_ITERATIONS, ## shall we cut down on epochs? no we need surrogate to converge
        early_stopping_patience=10_000,
        print_every=50, # don't print often
    )

    train_losses.extend(train_loss); val_losses.extend(val_loss) # concatenate losses across iterations
    # evaluate the current surrogate
    surrogate.eval()
    with torch.no_grad():
        # surrogate generated observations from the above 2 sets of 10k normalized thetas
        gt_surrogate = surrogate.sample(context=gt_source.to(device))  # forward pass produces standardized x-s
        gt_surrogate_two = surrogate.sample(context=gt_source_two.to(device))

    current_c2st = np.mean(c2st_scores(gt_simulator.cpu(), gt_surrogate.cpu())) # both are standardized
    current_swd = sliced_wasserstein_distance(gt_simulator.to(device), gt_surrogate,
                                          num_projections=4096, device=device).item()

    surro_c2sts[iteration] = current_c2st
    surro_swds[iteration] = current_swd
    print(f"Surrogate vs Simulator y-space C2ST AUC after iteration {iteration+1}: {current_c2st}")
    print(f"Surrogate vs Simulator y-space SWD AUC after iteration {iteration+1}: {current_swd}")

    torch.cuda.empty_cache() # needed!!
    """
        source fitting
    """
    
    schedule = schedule_org
    if cfg.sequential.lambda_stays_same:
        schedule = first_iteration_schedule if iteration==0 else next_iters_schedule

    train_source_loss = train_source(
        data=xo_stats_norm,  # note that we are passing standardized x-s to fit the source
        source_model=source,
        simulator=surrogate, ##### ALWAYS PASSING SURROGATE HERE FOR NAIVE SEQUENTIAL METHOD!!!
        optimizer=optimizer,
        entro_dist=None,
        entro_lambda=schedule,
        wasser_p=cfg.source.wasserstein_order,
        wasser_np=cfg.source.wasserstein_slices,
        use_log_sw=cfg.source.use_log_sw,
        num_chunks=cfg.source.num_chunks,
        epochs=cfg.source.pretraining_steps
        + cfg.source.linear_decay_steps
        + cfg.source.lambda_steps,
        min_epochs_x_chus=cfg.source.pretraining_steps + cfg.source.linear_decay_steps,
        early_stopping_patience=cfg.source.early_stopping_patience,
        device=device,
        print_every=1000,
    )
    train_source_losses.extend(train_source_loss)

    # Evaluate trained source model
    source.eval()
    surrogate.eval()
    with torch.no_grad():
        estimated_source = source.sample(10_000) # so we are assumung the source generates thetas in [-1, 1], because it is trained on standardized x-s
        moved_estimated_source = scale_tensor(   # then move it to the actual range if we wish to pass it through the simulator
            estimated_source.cpu(),
            -torch.ones(source_dim),
            torch.ones(source_dim),
            PRIOR_MIN,
            PRIOR_MAX,
        )
        surro_estimated_pf = surrogate.sample(10_000, estimated_source) # surrogate expects normalized source, standardized x-s
    
    surro_c2st = np.mean(c2st_scores(surro_estimated_pf.cpu(), gt_surrogate_two.cpu())) # both standardized
    print(f"y c2st AUC on surrogate: {surro_c2st}")
    nns_idx = torch.zeros(estimated_source.shape[0])
    for i, test_theta in enumerate(estimated_source):
        test = test_theta.unsqueeze(0)
        #print(test.shape)
        dist = torch.norm(params_gpu - test, dim=1, p=2)
        #print(dist.shape)
        nns_idx[i] = dist.argmin().item()
        # print(i)

    nns_estimated_pf = xs[nns_idx.long()]
    nns_c2st = np.mean(c2st_scores(nns_estimated_pf.cpu(), gt_simulator_two.cpu()))
    print(f"y c2st AUC (NNS estimate): {nns_c2st}")

    source_simu_pf_c2sts[iteration] = nns_c2st
    source_surro_pf_c2sts[iteration] = surro_c2st

    # plot true and learned sources
    fig_source = corner(
        gt_source_two.cpu().numpy(), # old θ-s from the actual prior theta
        color="black",
        bins=20,
        hist_kwargs={"density": True},
        plot_contours=False,
        plot_density=False,
        labels=theta_labels,
        # plot_datapoints=False,
    )
    corner(
        estimated_source.cpu().numpy(),
        fig=fig_source,
        color="red",
        bins=20,
        hist_kwargs={"density": True},
        plot_contours=False,
        plot_density=False,
        labels=theta_labels,
        # plot_datapoints=False,
    )
    fig_source.suptitle(f'source after iter: {iteration+1}/{cfg.sequential.number_of_iterations}')
    
    save_fig(
        fig_source,
        file_name=f"{cfg.base.tag}_source_fig_iter_{iteration+1}.pdf",
        folder=cfg.base.folder,
        base_path=cfg.base.base_path,
    )
    plt.close(fig_source)

    # Plot pairplots in observation space with surrogate
    fig_surro = corner(
        gt_surrogate_two.cpu().numpy(), # surrogate generated x-s from ground truth θ-s
        color="black",
        bins=20,
        hist_kwargs={"density": True},
        plot_contours=False,
        plot_density=False,
        labels=y_labels,
        # plot_datapoints=False,
    )
    corner(
        surro_estimated_pf.cpu().numpy(), # surrogate generated x-s from source generated θ-s
        fig=fig_surro,
        color="red",
        bins=20,
        hist_kwargs={"density": True},
        plot_contours=False,
        plot_density=False,
        labels=y_labels,
        # plot_datapoints=False,
    )
    fig_surro.suptitle(f'surro est pf after iter: {iteration+1}/{cfg.sequential.number_of_iterations}')
    save_fig(
        fig_surro,
        file_name=f"{cfg.base.tag}_surrogate_fig_iter_{iteration+1}.pdf",
        folder=cfg.base.folder,
        base_path=cfg.base.base_path,
    )
    plt.close(fig_surro)

    # Plot pairplots in observation space with true simulator
    fig_simu = corner(
        gt_simulator_two.cpu().numpy(),
        color="black",
        bins=20,
        hist_kwargs={"density": True},
        plot_contours=False,
        plot_density=False,
        labels=y_labels,
        # plot_datapoints=False,
    )
    corner(
        nns_estimated_pf.cpu().numpy(),
        fig=fig_simu,
        color="red",
        bins=20,
        hist_kwargs={"density": True},
        plot_contours=False,
        plot_density=False,
        labels=y_labels,
        # plot_datapoints=False,
    )
    fig_simu.suptitle(f'nns est pf after iter: {iteration+1}/{cfg.sequential.number_of_iterations}')
    save_fig(
        fig_simu,
        file_name=f"{cfg.base.tag}_nns_fig_iter_{iteration+1}.pdf",
        folder=cfg.base.folder,
        base_path=cfg.base.base_path,
    )
    plt.close(fig_simu)
    

    

    estimated_source_kole = kozachenko_leonenko_estimator(estimated_source, on_torus=False).item()
    print(f"Estimated source entropy estimate: {estimated_source_kole}")
    source_entropies[iteration] = estimated_source_kole

    ## update the source for next iteration of updating the surrogate
    surro_train_domain_distribution = source
    print("--__--"*10)

    torch.cuda.empty_cache()



results_df = pd.DataFrame(
    {
        "budgets": BUDGET_PER_ITERATION, # this will be wrong collate_prev_iter_data is set to True
        "surro_c2sts": surro_c2sts,
        "surro_swds": surro_swds,

        # "source_simu_pf_swds": source_simu_pf_swds,
        "source_simu_pf_c2sts": source_simu_pf_c2sts,
        "source_surro_pf_c2sts": source_surro_pf_c2sts,

        # "source_c2sts": source_c2sts,
        "source_entropies": source_entropies,
    }
)

results_df.to_csv(os.path.join(cfg.base.base_path, cfg.base.folder, f"{cfg.base.tag}_results_df.csv"), index=False)

# save everything as a dataframe?
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)
train_source_losses = np.array(train_source_losses)

save_state_dict(
    source,
    f"{cfg.base.tag}_learned_final_source.pt",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

save_state_dict(
    surrogate,
    f"{cfg.base.tag}_final_surrogate.pt",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

save_numpy_csv(
    train_losses,
    file_name=f"{cfg.base.tag}_train_losses_surro.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

save_numpy_csv(
    val_losses,
    file_name=f"{cfg.base.tag}_val_losses_surro.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

save_numpy_csv(
    train_source_losses,
    file_name=f"{cfg.base.tag}_source_training_losses.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)