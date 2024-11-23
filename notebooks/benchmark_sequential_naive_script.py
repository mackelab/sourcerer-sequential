import os
from time import time_ns; from datetime import datetime
import numpy as np; import pandas as pd; import matplotlib.pyplot as plt
import torch
from corner import corner
from omegaconf import OmegaConf as OC

from sourcerer.fit_surrogate import (
    create_train_val_dataloaders,
    fit_conditional_normalizing_flow,
    generate_data_for_surrogate,
    train_val_split,
    create_dataloader
)

from sourcerer.likelihood_estimator import train_lml_source
from sourcerer.real_nvp import (
    Sampler,
    RealNVPs,
    TemperedUniform, # a little slanted uniform to make log-density well defined
    kozachenko_leonenko_estimator,
)

from sourcerer.sbi_classifier_two_sample_test import c2st_scores
from sourcerer.simulators import (
    InverseKinematicsSimulator,
    LotkaVolterraSimulator,
    SIRSimulator,
    SLCPSimulator,
    TwoMoonsSimulator,
    GaussianMixtureSimulator,
)
from sourcerer.sliced_wasserstein import sliced_wasserstein_distance
from sourcerer.utils import (
    save_cfg_as_yaml,
    save_fig,
    save_numpy_csv,
    save_state_dict,
    script_or_command_line_cfg,
    set_seed,
)
from sourcerer.wasserstein_estimator import train_source

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f'run id: {run_id}')
print(f"using {device}")

def get_simulator(cfg):
    if cfg.simulator.self == "two_moons":
        return TwoMoonsSimulator()
    elif cfg.simulator.self == "inverse_kinematics":
        return InverseKinematicsSimulator()
    elif cfg.simulator.self == "slcp":
        return SLCPSimulator()
    elif cfg.simulator.self == "sir":
        return SIRSimulator()
    elif cfg.simulator.self == "lotka_volterra":
        return LotkaVolterraSimulator()
    elif cfg.simulator.self == "gaussian_mixture":
        return GaussianMixtureSimulator()
    else:
        raise ValueError
    

# Define config
# NOTE: These overrides only take effect if this script is run interactively
simulator_str = "inverse_kinematics"
local_overrides = [
    f"base.tag=debug_{run_id}",
    f"base.folder=sequential",# _{run_id}",
    f"simulator={simulator_str}",
    "surrogate.self=train_surrogate",
    "+surrogate.num_training_samples=15000",

"surrogate.flow_length=8",
"surrogate.hidden_layer_dim=50",

"+surrogate.nb_epochs=1000", # do we need to increase it to make sure the surrogates converge at each iteration?
"+surrogate.surrogate_lr=0.0001",
"+surrogate.surrogate_weight_decay=0.00005",

"+surrogate.early_stopping_patience=50",

# "source.fin_lambda=0.062", # only needed for gm simulator

# "sequential.total_simulation_budget=10000",
# "sequential.number_of_iterations=10",

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

# '''
# save the above config file in ../results_sourcerer/sequential directory
save_cfg_as_yaml(
    cfg,
    f"{cfg.base.tag}_cfg_{run_id}.yaml",
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

# Define Simulator and reference domain. Train a surrogate/load an existing surrogate if necessary.
simulator = get_simulator(cfg)
simulator = simulator.to(device)

## this is the initial distribution to sample theta from to train the surrogate
box_domain = TemperedUniform(
    cfg.simulator.box_domain_lower,
    cfg.simulator.box_domain_upper,
    simulator.xdim,
    device=device,
)

# notation here: θ (which is x here), and x (which is x here) 
print(f"θ/parameter dimension: {simulator.xdim}")
print(f"x/observation dimension: {simulator.ydim}")
theta_labels = [fr"$\theta_{i+1}$" for i in range(simulator.xdim)]
y_labels = [fr"$x_{i+1}$" for i in range(simulator.ydim)]

surrogate = RealNVPs(
    flow_length=cfg.surrogate.flow_length, # 1 RealNVP layer
    data_dim=simulator.ydim,               # this is x ~ p(x|theta)
    context_dim=simulator.xdim,            # this is theta, okay so context dim is theta's dimension
    hidden_layer_dim=cfg.surrogate.hidden_layer_dim # 5,
)
surrogate = surrogate.to(device)

surro_optimizer = torch.optim.Adam(
    surrogate.parameters(),
    lr=cfg.surrogate.surrogate_lr, # 0.0001
    weight_decay=cfg.surrogate.surrogate_weight_decay, # 0.00005
)

assert cfg.surrogate.self == "train_surrogate"

## used for evaluation of surrogate and estimated source later
gt_source = simulator.sample_prior(cfg.source.num_obs)          # 10k thetas from true source distribution
gt_source_two = simulator.sample_prior(cfg.source.num_eval_obs) # 10k thetas for validation

gt_source_kole = kozachenko_leonenko_estimator(gt_source_two, on_torus=False).item() # true source entropy, just compute and save
print(f"Ground truth source entropy estimate: {gt_source_kole}")
save_numpy_csv(
    np.array([gt_source_kole]),
    file_name=f"{cfg.base.tag}_gt_source_kole.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)


with torch.no_grad():
    # simulator generated observations/pfs from the above 2 sets of 10k thetas
    gt_simulator = simulator.sample(context=gt_source)  # should this also be included in the budget?
    gt_simulator_two = simulator.sample(context=gt_source_two)

# print("Sliced wassertein distance on groundtruth y:")
expected_distance = sliced_wasserstein_distance(
    gt_simulator[: cfg.source.num_eval_obs],  # from theta1
    gt_simulator_two,                         # from theta2 (both ground truth)
    num_projections=4096,
    device=device,
).item()


print(f"Sliced wassertein distance on groundtruth y (expected/baseline swd): {expected_distance}, log: {np.log(expected_distance)}")
save_numpy_csv(np.array([expected_distance]), file_name=f"{cfg.base.tag}_expected_swd.csv",
               folder=cfg.base.folder, base_path=cfg.base.base_path)


## define source model
if cfg.source_model.self == "sampler":
    # easier to scale, used in the paper
    # note that Sampler() model does not have log-density evaluation
    print("init sampler source")
    source = Sampler(
        xdim=simulator.xdim,
        input_noise_dim=simulator.xdim,
        hidden_layer_dim=cfg.source_model.hidden_layer_dim,
        num_hidden_layers=cfg.source_model.num_hidden_layers,
        low=cfg.simulator.box_domain_lower,
        high=cfg.simulator.box_domain_upper,
    )
elif cfg.source_model.self == "real_nvp":
    source = RealNVPs(
        data_dim=simulator.xdim,
        context_dim=0,  # here we are not using it as a conditioning mechanism!
        hidden_layer_dim=cfg.source_model.hidden_layer_dim,
        flow_length=cfg.source_model.flow_length,
        low=cfg.simulator.box_domain_lower,
        high=cfg.simulator.box_domain_upper,
    )
else:
    raise ValueError

source = source.to(device)

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
    
# print(BUDGET_PER_ITERATION)

""" naive sequential method for loop starting from here
    update surrogate --> train source --> store metrics and visualizations
"""


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

# start loop here, don't initialize surrogate again!
print(f"naive sequential method on {cfg.simulator.self} with {cfg.sequential.number_of_iterations} iters and total budget: {cfg.sequential.total_simulation_budget}")
print(f"aggregating data? {cfg.sequential.collate_last_iter_data}, varying lambda over its? {not cfg.sequential.lambda_stays_same}")
for iteration in range(cfg.sequential.number_of_iterations):
    print(f"ITER: {iteration+1}/{cfg.sequential.number_of_iterations} fitting surrogate with {BUDGET_PER_ITERATION[iteration]} samples")
    surro_domain, surro_push_forward = generate_data_for_surrogate(simulator,
                                                                   surro_train_domain_distribution,
                                                                   number=BUDGET_PER_ITERATION[iteration]
                                                                  )
    if iteration == 0:
        (surro_train_push_forward, surro_train_domain), (surro_val_push_forward, surro_val_domain) = train_val_split(
            surro_push_forward, surro_domain
        )
        # print(f"first iter tr. data θ: {surro_train_domain.shape}, x: {surro_train_push_forward.shape}")
        # print(f"first iter val data θ: {surro_val_domain.shape}, x: {surro_val_push_forward.shape}")
    
    else:
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
        nb_epochs=cfg.surrogate.nb_epochs, #//NUMBER_OF_ITERATIONS, ## shall we cut down on epochs? no we need surrogate to converge
        early_stopping_patience=cfg.surrogate.early_stopping_patience,
        print_every=5000, # don't print often
    )

    train_losses.extend(train_loss); val_losses.extend(val_loss) # concatenate losses across iterations
    # evaluate the current surrogate
    surrogate.eval()
    with torch.no_grad():
        # surrogate generated observations from the above 2 sets of 10k thetas
        gt_surrogate = surrogate.sample(size=cfg.source.num_obs, context=gt_source)
        gt_surrogate_two = surrogate.sample(size=cfg.source.num_eval_obs, context=gt_source_two)
    
    current_c2st = np.mean(c2st_scores(gt_simulator.cpu(), gt_surrogate.cpu()))
    current_swd = sliced_wasserstein_distance(gt_simulator, gt_surrogate,
                                              num_projections=4096, device=device
                                             ).item()
    surro_c2sts[iteration] = current_c2st
    surro_swds[iteration] = current_swd
    print(f"Surrogate vs Simulator y-space C2ST AUC after iteration {iteration+1}: {current_c2st}")
    print(f"Surrogate vs Simulator y-space SWD AUC after iteration {iteration+1}: {current_swd}")
    
    schedule = schedule_org
    if cfg.sequential.lambda_stays_same:
        schedule = first_iteration_schedule if iteration==0 else next_iters_schedule
        
    train_source_loss = train_source(
        data=gt_simulator, # observations (y) sampled using simulator.sample(context=gt_source)
        source_model=source,
        simulator=surrogate,  ##### ALWAYS PASSING SURROGATE HERE FOR NAIVE SEQUENTIAL METHOD!!!
        optimizer=optimizer,
        entro_dist=None,   # default uniform is used
        kld_samples=cfg.source.num_kole_samples, # 512
        entro_lambda=schedule, # careful! this is modified according to cfg.sequential.lambda_stays_same
        wasser_p=cfg.source.wasserstein_order,   # 2
        wasser_np=cfg.source.wasserstein_slices, # 500
        use_log_sw=cfg.source.use_log_sw,   # True => log_or_id = torch.log
        num_chunks=cfg.source.num_chunks,   # 1
        epochs=cfg.source.pretraining_steps # 0
        + cfg.source.linear_decay_steps     # 500
        + cfg.source.lambda_steps,          # 3000
        min_epochs_x_chus=cfg.source.pretraining_steps + cfg.source.linear_decay_steps,
        early_stopping_patience=cfg.source.early_stopping_patience, # 500
        device=device,
        print_every=5000,
    )
    train_source_losses.extend(train_source_loss)

    # Evaluate trained source model
    source.eval()
    surrogate.eval()
    with torch.no_grad():
        # sample some θ-s from the estimated source
        estimated_source = source.sample(cfg.source.num_eval_obs)
        surro_estimated_pf = surrogate.sample(cfg.source.num_eval_obs, estimated_source) #pushforward those θ-s through surrogate
        simu_estimated_pf = simulator.sample(estimated_source) #pushforward those θ-s through the real simulator (!)


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

    # Avoid plotting very large corner plots
    plot_ss = slice(None)
    if cfg.simulator.self == "sir":
        plot_ss = slice(1, 50, 4)
    elif cfg.simulator.self == "lotka_volterra":
        plot_ss = slice(10, 20)
    # Plot pairplots in observation space with surrogate
    fig_surro = corner(
        gt_surrogate_two.cpu().numpy()[:, plot_ss], # surrogate generated x-s from ground truth θ-s
        color="black",
        bins=20,
        hist_kwargs={"density": True},
        plot_contours=False,
        plot_density=False,
        labels=y_labels,
        # plot_datapoints=False,
    )
    corner(
        surro_estimated_pf.cpu().numpy()[:, plot_ss], # surrogate generated x-s from source generated θ-s
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
        gt_simulator_two.cpu().numpy()[:, plot_ss],
        color="black",
        bins=20,
        hist_kwargs={"density": True},
        plot_contours=False,
        plot_density=False,
        labels=y_labels,
        # plot_datapoints=False,
    )
    corner(
        simu_estimated_pf.cpu().numpy()[:, plot_ss],
        fig=fig_simu,
        color="red",
        bins=20,
        hist_kwargs={"density": True},
        plot_contours=False,
        plot_density=False,
        labels=y_labels,
        # plot_datapoints=False,
    )
    fig_simu.suptitle(f'simu est pf after iter: {iteration+1}/{cfg.sequential.number_of_iterations}')
    save_fig(
        fig_simu,
        file_name=f"{cfg.base.tag}_simulator_fig_iter_{iteration+1}.pdf",
        folder=cfg.base.folder,
        base_path=cfg.base.base_path,
    )
    plt.close(fig_simu)

    # source metrics in terms of pushforwards: Classifier two sample tests
    simu_c2st = np.mean(c2st_scores(simu_estimated_pf.cpu(), gt_simulator_two.cpu()))
    surro_c2st = np.mean(c2st_scores(surro_estimated_pf.cpu(), gt_surrogate_two.cpu()))

    # swd computation
    num_repeat = 10
    simu_dists = np.zeros(num_repeat)
    for idx in range(10):
        with torch.no_grad():
            # pushforward from simulator on a new & different θ from estimated source
            simu_est_pf_add = simulator.sample(source.sample(cfg.source.num_eval_obs))
        simu_dists[idx] = sliced_wasserstein_distance(
            simu_est_pf_add,   # pushforward from simulator using estimated θ
            gt_simulator_two,  # pushforward from simulator using ground truth θ
            num_projections=4096,
            device=device,
        )
    simu_swd, simu_swd_std = np.mean(simu_dists), np.std(simu_dists)
    
    source_simu_pf_c2sts[iteration] = simu_c2st
    source_surro_pf_c2sts[iteration] = surro_c2st
    source_simu_pf_swds[iteration] = simu_swd
    print(f"y c2st AUC on simulator: {simu_c2st}")
    print(f"y swd on simulator: {simu_swd} +- {simu_swd_std}")
    print(f"y c2st AUC on surrogate: {surro_c2st}")


    # source metrics: c2st and entropy
    source_c2st = np.mean(
        c2st_scores( # 2 `sets` of samples of thetas, from the estimated and the ground truth sources
            estimated_source.cpu(),
            gt_source_two.cpu(),
        )
    )
    source_c2sts[iteration] = source_c2st
    
    # print("Estimate source entropies")
    estimated_source_kole = kozachenko_leonenko_estimator(estimated_source, on_torus=False).item()
    source_entropies[iteration] = estimated_source_kole

    print(f"Source c2st AUC: {source_c2st}")
    print(f"Ground truth source entropy estimate: {gt_source_kole}")
    print(f"Estimated source entropy estimate: {estimated_source_kole}")


    ## update the source for next iteration of updating the surrogate
    surro_train_domain_distribution = source
    print("--__--"*10)

""" naive sequential method for loop ends here
    save the statistics
"""


results_df = pd.DataFrame(
    {
        "budgets": BUDGET_PER_ITERATION, # this will be wrong collate_prev_iter_data is set to True
        "surro_c2sts": surro_c2sts,
        "surro_swds": surro_swds,

        "source_simu_pf_swds": source_simu_pf_swds,
        "source_simu_pf_c2sts": source_simu_pf_c2sts,
        "source_surro_pf_c2sts": source_surro_pf_c2sts,

        "source_c2sts": source_c2sts,
        "source_entropies": source_entropies,
    }
)

results_df.to_csv(os.path.join(cfg.base.base_path, cfg.base.folder, f"{cfg.base.tag}_results_df.csv"), index=False)

# save everything as a dataframe?
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)
train_source_losses = np.array(train_source_losses)

# save_state_dict(
#     source,
#     f"{cfg.base.tag}_learned_final_source.pt",
#     folder=cfg.base.folder,
#     base_path=cfg.base.base_path,
# )

# save_state_dict(
#     surrogate,
#     f"{cfg.base.tag}_final_surrogate.pt",
#     folder=cfg.base.folder,
#     base_path=cfg.base.base_path,
# )

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
# '''