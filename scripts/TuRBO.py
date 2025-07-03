import math
from dataclasses import dataclass

import torch
from scripts.TuRBO_sim import simulate
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize, standardize
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood

# --- Minimal TuRBO helper (aus Tutorial) -----------------------------------------
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0  # wird im __post_init__ gesetzt
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        # Mindestens ceil(dim / batch_size) Failures erlauben
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )


def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    if Y_next.max() > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.failure_counter += 1
        state.success_counter = 0

    if state.success_counter >= state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, Y_next.max().item())
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


def generate_batch(state: TurboState, model, X, Y, batch_size: int, n_candidates: int = 5000):
    dim, dtype, device = X.shape[-1], X.dtype, X.device
    # best observed center in normalized space
    x_center = X[Y.argmax(), :].clone()

    # gewichte Länge pro Achse
    ls = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = ls / ls.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    sobol = SobolEngine(dim, scramble=True)
    X_cand = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    X_cand = tr_lb + (tr_ub - tr_lb) * X_cand

    # Max Posterior Sampling für Batch-Kandidaten
    from botorch.generation import MaxPosteriorSampling
    ts = MaxPosteriorSampling(model=model, replacement=False)
    return ts(X_cand, num_samples=batch_size)
# -------------------------------------------------------------------------------

# --- Hyperparameter-Bounds (16 Parameter) --------------------------------------
bounds = torch.tensor([
    [   # lower bounds
        1, 1, 1e-4, 1, 10,   1e-3, 0.0, 1e-5, 1e-3, # QC, QL, MU, DVTHETA_MAX, N, T_HORIZON, ALPHA_INTERP, BW_RAMP, RAMP_TIME
        1, 0.4, 0.15, 0.3, 0.05, 1e-6, 1e-6, 1e-6 # BARRIER_WEIGHT, TUNNEL_WIDTH, TUNNEL_WIDTH_GATE, NARROW_DIST, GATE_FLAT_DIST, R_VTHETA, REG_THRUST, REG_INPUTS
    ],
    [   # upper bounds
        50,  50,   10,  4,  40,   1.0,  1.0,  0.2, 2.0,
        50,  0.6,   0.4,  0.7,  0.3,  1.0,  1.0,  1.0
    ]
], dtype=torch.double)
num_dims = bounds.shape[1]
# -------------------------------------------------------------------------------

# --- Custom Start-Param-Dicts --------------------------------------------
# Definiere Start-Konfigurationen als Liste von Dictionaries:
hyperparam_names = [
    'QC','QL','MU','DVTHETA_MAX','N','T_HORIZON','ALPHA_INTERP','BW_RAMP', 'RAMP_TIME',
    'BARRIER_WEIGHT','TUNNEL_WIDTH','TUNNEL_WIDTH_GATE','NARROW_DIST',
    'GATE_FLAT_DIST','R_VTHETA','REG_THRUST','REG_INPUTS'
]

start_param_dicts = [
    # Beispiel 1: Baseline
    {
        'QC': 10, 'QL': 40, 'MU': 10, 'DVTHETA_MAX': 1.0,
        'N': 20, 'T_HORIZON': 0.7, 'ALPHA_INTERP': 0.1, 'BW_RAMP': 1, 'RAMP_TIME' : 1.6,
        'BARRIER_WEIGHT': 10, 'TUNNEL_WIDTH': 0.4, 'TUNNEL_WIDTH_GATE': 0.15,
        'NARROW_DIST': 0.7, 'GATE_FLAT_DIST': 0.1, 'R_VTHETA': 1.0,
        'REG_THRUST': 1.0e-4, 'REG_INPUTS': 9.0e-2
    },
]

# Umwandeln in Tensor (n_start × num_dims)
start_points = torch.tensor(
    [[d[name] for name in hyperparam_names] for d in start_param_dicts],
    dtype=torch.double
)
# ---------------------------------------------------------------------------


def evaluate_controller(params: torch.Tensor, n_eval: int = 3) -> float:
    """
    Ruft simulate() mit allen Hyperparametern auf.
    Reward:
      - alle 4 Gates geschafft: reward = -time_finished
      - Gate(s) verfehlt:      reward = -1e6 - 1e5*(fehlende Gates)
    """
    hp = params.tolist()
    rewards = []
    for _ in range(n_eval):
        param_dict = dict(    
            QC =               hp[0],
            QL =              hp[1],
            MU =               hp[2],
            DVTHETA_MAX =    hp[3],
            N            =   int(round(hp[4])),
            T_HORIZON     =  hp[5],
            ALPHA_INTERP  =  hp[6],
            BW_RAMP       = hp[7],
            RAMP_TIME     = hp[8],
            BARRIER_WEIGHT = hp[9],
            TUNNEL_WIDTH   = hp[10],
            TUNNEL_WIDTH_GATE= hp[11],
            NARROW_DIST     =hp[12],
            GATE_FLAT_DIST = hp[13],
            R_VTHETA       = hp[14],
            REG_THRUST     = hp[15],
            REG_INPUTS     = hp[16],
        )
        # Debug: show which parameters are being evaluated
        print("PARAM_DICT:")
        for k, v in param_dict.items():
            print(f"  {k:18} = {v}")
        time_finished, gates_passed = simulate(
            n_runs=2,
            gui=False,
            visualize=False,
            PARAM_DICT=param_dict
        )
        # Debug: print raw simulator outputs
        print(gates_passed)
        print(time_finished)
        for i in range(len(time_finished)):
            if time_finished[i] is None:
                time_finished[i] = 1e4
        # Aggregate lists if simulate returns lists
        if isinstance(time_finished, (list, tuple)):
            avg_time = sum(time_finished) / len(time_finished)
        else:
            avg_time = time_finished
        if isinstance(gates_passed, (list, tuple)):
            # penalize total missing gates across runs
            total_missing = sum(max(0, 4 - g) for g in gates_passed)
            if total_missing > 0:
                reward = -1e6 - 1e5 * total_missing
            else:
                reward = -avg_time
        else:
            if gates_passed < 4:
                reward = -1e6 - 1e5 * (4 - gates_passed)
            else:
                reward = -avg_time
        rewards.append(reward)
        # Debug: print current average reward
        print(float(sum(rewards) / len(rewards)))
    return float(sum(rewards) / len(rewards))

# --- Initial Design ------------------------------------------------------------
torch.manual_seed(0)

n_init = max(2, 4 * num_dims)
num_start = start_points.size(0)
num_rest = n_init - num_start
if num_rest > 0:
    # Sobol samples come back as (num_rest, 1, num_dims), reshape to (num_rest, num_dims)
    rest = draw_sobol_samples(bounds=bounds, n=num_rest, q=1, seed=0).reshape(num_rest, num_dims)
    X_init = torch.cat([start_points, rest], dim=0)
else:
    X_init = start_points
Y_init = torch.tensor([evaluate_controller(x) for x in X_init],
                      dtype=torch.double).unsqueeze(-1)

train_X = normalize(X_init, bounds)
train_Y = standardize(Y_init)
# -------------------------------------------------------------------------------

# --- TuRBO State & Loop --------------------------------------------------------
state = TurboState(dim=num_dims, batch_size=1)

n_iter = 60
for itr in range(n_iter):
    # GP anpassen
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # nächste Kandidaten
    X_next_norm = generate_batch(state, gp, train_X, train_Y, batch_size=1)
    X_next = unnormalize(X_next_norm.squeeze(0), bounds)

    # simuliere und reward
    y_val = evaluate_controller(X_next)
    Y_next = torch.tensor([[y_val]], dtype=torch.double)

    # TR-Status updaten
    state = update_state(state, Y_next)

    # neue Daten anhängen
    train_X = torch.cat([train_X, X_next_norm], dim=0)
    train_Y = torch.cat([
        train_Y,
        standardize(Y_next, train_Y.mean(), train_Y.std())
    ], dim=0)

    if state.restart_triggered:
        print(f"Trust region collapsed at iteration {itr}, restart triggered.")
        break

# bestes Ergebnis
best_idx = train_Y.argmax()
best_params = unnormalize(train_X[best_idx], bounds)
print("Best hyperparameters found:\n", best_params)