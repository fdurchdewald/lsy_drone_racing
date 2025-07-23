"""Configuration parameters for the MPC controller in drone racing.

This module defines the MPCC_CFG dictionary containing all relevant
tuning parameters for the controller, including weights, tunnel widths,
obstacle radii, and other settings.
"""

MPCC_CFG = dict(
    QC=17,                  # contouring cost
    QL=57,                  # lagging cost
    MU=74,                  # progress cost
    DVTHETA_MAX=1.792,      # maximum change in vtheta
    N=20,                   # number of control intervals
    T_HORIZON=1.0,          # horizon time in seconds
    ALPHA_INTERP=0.8,       # smoothing factor for tunnel width interpolation: 0=no movement, 1=full shift
    RAMP_TIME=0.5,          # start ramp time for the controller
    BW_RAMP=0.2,            # ramp time for barrier weight
    BARRIER_WEIGHT=63,      # barrier weight
    TUNNEL_WIDTH=0.542,     # nominal tunnel width
    NARROW_DIST=0.71,       # distance (m) at which tunnel starts to narrow
    GATE_FLAT_DIST=0.16306670286442024,     # distance (m) at which gate is flat
    R_VTHETA=0.3482765306459784,            # quadratic penalty on vtheta
    REG_THRUST=0.07364509113357147,         # regularization on thrust
    REG_INPUTS=0.15248400348396848,         # regularization on inputs
    OBSTACLE_RADIUS=[0.13, 0.14, 0.1, 0.1], # radius of obstacles in meters
    TUNNEL_WIDTH_GATE = [0.2707038115931886, 0.10141761868934414, 0.13493788046375668, 0.18311715476279598],  # reduced width of gate opening
)