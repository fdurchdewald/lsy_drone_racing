import math
import multiprocessing as mp
import os
import pathlib
import shutil
import tempfile
from dataclasses import dataclass

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, standardize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

import lsy_drone_racing.control.mpccpp_move as mpcc_move
from lsy_drone_racing.control.debug_utils import get_logger as logger
from scripts.TuRBO_sim import simulate

_orig_create = mpcc_move.create_ocp_solver
def _create_ocp_solver_patched(Tf, N, verbose=False):
    suffix = os.getenv("ACADOS_JSON_SUFFIX", "")
    return _orig_create(Tf, N, verbose=verbose, json_file=f"lsy_example_mpc{suffix}.json")
mpcc_move.create_ocp_solver = _create_ocp_solver_patched

log = logger('TuRBO')
# --- Minimal TuRBO helper (aus Tutorial) -----------------------------------------
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.4
    length_min: float = 0.5 ** 5
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0  # wird im __post_init__ gesetzt
    success_counter: int = 0
    success_tolerance: int = 5
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
    ls = model.covar_module.lengthscale.squeeze().detach()
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

# --- Hyperparameter-Bounds (24 Parameter) --------------------------------------
# Order MUST match hyperparam_names below. The last 8 dims are 4 OBSTACLE_RADIUS
# values and 4 per-gate TUNNEL_WIDTH_GATE values.
bounds = torch.tensor([
    [   # lower bounds
        10,     # QC
        10,     # QL
        1e-4,  # MU
        1.5,   # DVTHETA_MAX
        1e-4,  # R_VTHETA
        -1e-2,  # REG_THRUST
        1e-4,  # REG_INPUTS
    ],
    [   # upper bounds
        50,    # QC
        80,    # QL
        100,    # MU
        3.0,   # DVTHETA_MAX
        2.0,  # R_VTHETA
        1.0,   # REG_THRUST
        1.0,   # REG_INPUTS
    ]
], dtype=torch.double)
num_dims = bounds.shape[1]
# -------------------------------------------------------------------------------

# --- Custom Start-Param-Dicts --------------------------------------------
# Definiere Start-Konfigurationen als Liste von Dictionaries:
hyperparam_names = [
    'QC','QL','MU','DVTHETA_MAX','R_VTHETA','REG_THRUST','REG_INPUTS',
]

start_param_dicts = [
    {
        'QC':
        17.815010170135714,
        'QL': 
        57.757025392584005,
        'MU': 
        74.50459495671505,
        'DVTHETA_MAX':
        1.7512476467771976,
        'R_VTHETA':
        0.3482765306459784,
        'REG_THRUST':
        0.05364509113357147,
        'REG_INPUTS':
        0.12248400348396848,
        },
]

# Umwandeln in Tensor (n_start × num_dims)
start_points = torch.tensor(
    [[d[name] for name in hyperparam_names] for d in start_param_dicts],
    dtype=torch.double
)
# ---------------------------------------------------------------------------
def _single_run(param_dict):
    """Führt genau einen Simulator-Run aus und gibt das Ergebnis als Tupel zurück.
    Wird von ProcessPoolExecutor in einem separaten Prozess ausgeführt.
    """
    t_finished, g_passed, d_z, c_mass = simulate(
        n_runs=1, gui=False, visualize=False, PARAM_DICT=param_dict
    )
    # simulate() gibt Listen zurück → auf einzelne Werte reduzieren
    return (
        t_finished[0] if isinstance(t_finished, list) else t_finished,
        g_passed[0]   if isinstance(g_passed, list)   else g_passed,
        d_z[0]        if isinstance(d_z, list)        else d_z,
        c_mass,
    )

_BASE_REPO = pathlib.Path(__file__).resolve().parent  # = repo root

def _init_worker():
    """Wird **einmal** in jedem Fork ausgeführt.
    1.   Kopiert das Verzeichnis  <repo>/c_generated_code →  ein Temp-Dir
    2.   Setzt ACADOS_TEMPLATE_PATH und wechselt dorthin (os.chdir)
    3.   Hinterlegt PID-Suffix für die JSON-Datei im Env (wird später gelesen)
    """
    pid = os.getpid()
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix=f"acados_{pid}_"))
    # 1) komplette Code-Schablone kopieren (nur ~300 kB)
    src = _BASE_REPO / "c_generated_code"
    if src.exists():
        shutil.copytree(src, tmpdir / "c_generated_code")
    # 2) Umgebungs­variable für acados_template
    os.environ["ACADOS_TEMPLATE_PATH"] = str(tmpdir)
    # 3) JSON-Suffix merken
    os.environ["ACADOS_JSON_SUFFIX"] = f"_{pid}"
    # 4) in Temp-Dir wechseln, damit alle Relativ­pfade dort landen
    os.chdir(tmpdir)

# --- Parallele Evaluations-Funktion ------------------------------------------
def evaluate_controller(params: torch.Tensor, n_runs: int = 20) -> float:
    """Bewertet einen Hyperparameter-Vektor parallel über `n_runs` Simulator-Aufrufe.
    Alle Logs/Debug-Ausgaben entsprechen der bisherigen seriellen Implementierung.
    """
    # -------------------------------------------------------------
    # 1) Hyperparameter-Vektor → param_dict wie bisher
    # -------------------------------------------------------------
    hp = params.tolist()
    param_dict = {
        'QC': hp[0],  'QL': hp[1],  'MU': hp[2],        'DVTHETA_MAX': hp[3], 'R_VTHETA': hp[4],
        'REG_THRUST': hp[5],       'REG_INPUTS': hp[6],
       
    }

    # Debug-Ausgabe wie gehabt
    print("PARAM_DICT:")
    for k, v in param_dict.items():
        print(f"  {k:18} = {v}")

    results = POOL.map(_single_run, [param_dict]*n_runs)

    # -------------------------------------------------------------
    # 3) Ergebnisse aggregieren – identisch zur Original-Logik
    # -------------------------------------------------------------
    time_finished, gates_passed, dist_z = [], [], []
    current_mass = None                           # wird pro Run zurückgegeben

    for t, g, d, m in results:
        time_finished.append(t)
        gates_passed.append(g)
        dist_z.append(d)
        current_mass = m                          # identisch in allen Runs

    # Debug-Prints
    print(f"gates passed: {gates_passed}")
    print(f"time finished: {time_finished}")
    print(f"current mass: {current_mass}")

    valid_times = [t for t in time_finished if t is not None]
    avg_valid_time = (sum(valid_times) / len(valid_times)) if valid_times else float("inf")
    if valid_times:
        print(f"Average Time: {avg_valid_time}")

    not_finished = 0
    best_times = []
    for i in time_finished:
        if i is not None:
            best_times.append(i)
        if i is None:
            not_finished += 1
            if not_finished > (0.4*len(time_finished)):
                best_times.append(15)


    avg_time = sum(best_times) / len(best_times)
    reward = -avg_time
    print(f"Reward: {reward}")

    finished_ratio = len(valid_times) / n_runs
    log.debug(
        f"reward {reward}, Average Time: {avg_valid_time}, Finished: {finished_ratio}, "
        f"Time: {time_finished}, used parameters: {param_dict}, "
        f"current mass: {current_mass} dist z: {dist_z}"
    )

    return reward

# -------------------------------------------------------
#  Globales Handle – wird in main() befüllt
POOL = None
# -------------------------------------------------------

def main():
    global POOL

    # ---------- Pool erst hier anlegen (spawn-sicher) ----
    CTX = mp.get_context("spawn")
    POOL = CTX.Pool(
        processes=4,
        initializer=_init_worker,
    )

    try:
        torch.manual_seed(0)

        n_init = max(2, 5* num_dims) 
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
        train_Y_raw = Y_init.clone()
        # -------------------------------------------------------------------------------

        # --- TuRBO State & Loop --------------------------------------------------------
        state = TurboState(dim=num_dims, batch_size=2)

        n_iter = 1000
        for itr in range(n_iter):
            # GP anpassen
            gp = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Generiere eine Batch von Kandidaten entsprechend state.batch_size
            X_next_norm = generate_batch(state, gp, train_X, train_Y, batch_size=state.batch_size)

            # Unnormalize alle Kandidaten (Form: (batch_size, dim))
            X_next = unnormalize(X_next_norm, bounds)

            # Jede Kandidaten-Konfiguration evaluieren
            Y_list = []
            for cand in X_next:
                y_val = evaluate_controller(cand)
                Y_list.append([y_val])
            Y_next = torch.tensor(Y_list, dtype=torch.double)

            log.debug(f"Batch iteration {itr}: rewards {Y_next.view(-1).tolist()}, current TR length {state.length}")

            # TR-Status updaten
            state = update_state(state, Y_next)

            # neue Daten anhängen (RAW und anschließend neu standardisieren)
            train_X = torch.cat([train_X, X_next_norm], dim=0)
            train_Y_raw = torch.cat([train_Y_raw, Y_next], dim=0)
            train_Y = standardize(train_Y_raw)
            log.debug(f"Best raw reward so far: {train_Y_raw.max().item():.4f}")

            if state.restart_triggered:
                print(f"Trust region collapsed at iteration {itr}, restart triggered.")
                break

        # bestes Ergebnis
        best_idx = train_Y.argmax()
        best_params = unnormalize(train_X[best_idx], bounds)
        print("Best hyperparameters found:\n", best_params)
    finally:
        # ---------- sauber schließen & Temp-Dirs räumen ----------
        POOL.close()
        POOL.join()
        _cleanup_acados_tmpdirs()

def _cleanup_acados_tmpdirs():
    tmp_root = pathlib.Path(tempfile.gettempdir())
    for p in tmp_root.glob("acados_*"):
        shutil.rmtree(p, ignore_errors=True)

if __name__ == "__main__":
    main()
