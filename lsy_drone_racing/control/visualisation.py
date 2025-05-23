"""
visualisation.py: 3D-Plot von Gates (seitliche Öffnungen), Obstacles als Säulen bis zum Boden,
und geplanter Trajektorie mit markierten Waypoints für Level 2 inklusive randomisierter Obstacles,
symmetrischem Grid, Visualisierung der Gate-Randomisierung und Gate-Mittelpunkte
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Hardcoded Level 2 configuration ---
# Gate-Mittelpunkte (nominal)
gates = np.array([
    [0.45, -0.5, 0.56],
    [1.0, -1.05, 1.11],
    [0.0, 1.0, 0.56],
    [-0.5, 0.0, 1.11],
])
# Gate-RPY (roll, pitch, yaw)
gates_rpy = np.array([
    [0.0, 0.0, 2.35],
    [0.0, 0.0, -0.78],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 3.14],
])
# Randomization ranges for gates
gate_pos_min = np.array([-0.15, -0.15, -0.1])
gate_pos_max = np.array([ 0.15,  0.15,  0.1])
gate_rpy_min = np.array([0.0, 0.0, -0.1])
gate_rpy_max = np.array([0.0, 0.0,  0.1])

# Hindernisse (nominal)
obstacles = np.array([
    [1.0, 0.0, 1.4],
    [0.5, -1.0, 1.4],
    [0.0, 1.5, 1.4],
    [-0.5, 0.5, 1.4],
])
# Randomization range for obstacles (uniform)
obs_min = np.array([-0.15, -0.15, -0.05])
obs_max = np.array([ 0.15,  0.15,  0.05])

# Waypoints für Spline
waypoints = np.array([
    [1.09, 1.4, 0.1],
    [0.8, 1.0, 0.2],
    [0.6, -0.35, 0.56],
    #[0.45, -0.5, 0.56],
    [0.3, -0.65, 0.56],
    [0.2, -1.3, 0.65],
    [0.85, -1.2, 1.11],
    #[1.0, -1.05, 1.11],
    [1.1, -0.5, 1.11],
    [0.1, 0.5, 0.65],
    [0.0, 1.0, 0.56],
    [0.0, 1.2, 0.525],
    [0.0, 1.2, 1.1],
    [-0.25, 0.5, 1.1],
    [-0.5, 0.0, 1.1],
    [-0.5, -0.5, 1.1],
])

# Erzeuge CubicSpline
ts_way = np.linspace(0, 1, len(waypoints))
cs_x = CubicSpline(ts_way, waypoints[:, 0])
cs_y = CubicSpline(ts_way, waypoints[:, 1])
cs_z = CubicSpline(ts_way, waypoints[:, 2])
# Spline-Pfad mit hoher Auflösung
ts_fine = np.linspace(0, 1, 500)
spline_pts = np.vstack([cs_x(ts_fine), cs_y(ts_fine), cs_z(ts_fine)]).T

# Gate-Halbmaße
half_width = 0.225  # in Meter
half_height = 0.225  # in Meter (Gate-Höhe)
# Hindernis-Basishöhe
ground_z = 0.0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 1) Plot Gates als korrekt orientierte vertikale Quadrate
for center, rpy in zip(gates, gates_rpy):
    rot = R.from_euler('xyz', rpy)
    x_axis = rot.apply([1, 0, 0])  # Breite local x
    z_axis = rot.apply([0, 0, 1])  # Höhe  local z
    corners = []
    for sx in [+1, -1]:
        for sz in [+1, -1]:
            corner = center + sx * half_width * x_axis + sz * half_height * z_axis
            corners.append(corner)
    corners = np.array(corners)
    order = [0, 2, 3, 1, 0]
    sq = corners[order]
    ax.plot(sq[:,0], sq[:,1], sq[:,2], 'b-')

# 2) Gate-Mittelpunkte als Punkte
ax.scatter(gates[:,0], gates[:,1], gates[:,2], marker='o', c='k', s=50, label='Gate centers')

# 3) Visualisierung der Gate-Position-Randomisierung als halbtransparenten Würfel um jeden Mittelpunkt
for center in gates:
    x0, y0, z0 = center + gate_pos_min
    x1, y1, z1 = center + gate_pos_max
    verts = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
    ]
    pc = Poly3DCollection(verts, facecolors='cyan', linewidths=0.5, edgecolors='cyan', alpha=0.2)
    ax.add_collection3d(pc)

# 4) Visualisierung der Gate-Rotation-Randomisierung durch mehrere Probe-Quadrate
n_samples = 8
for center, base_rpy in zip(gates, gates_rpy):
    for yaw_offset in np.linspace(gate_rpy_min[2], gate_rpy_max[2], n_samples):
        rpy = base_rpy + np.array([0.0,0.0,yaw_offset])
        rot = R.from_euler('xyz', rpy)
        x_axis = rot.apply([1, 0, 0])
        z_axis = rot.apply([0, 0, 1])
        corners = np.array([center + sx*half_width*x_axis + sz*half_height*z_axis
                            for sx in [+1,-1] for sz in [+1,-1]])
        sq = corners[[0,2,3,1,0]]
        ax.plot(sq[:,0], sq[:,1], sq[:,2], color='cyan', alpha=0.3)

# 5) Plot Obstacles als Säulen bis zum Boden
for obs in obstacles:
    x, y, z = obs
    ax.plot([x, x], [y, y], [ground_z, z], color='r', linewidth=4)

# 6) Plot randomization regions for obstacles as translucent cubes from ground to max
for obs in obstacles:
    x0, y0 = obs[0] + obs_min[0], obs[1] + obs_min[1]
    x1, y1 = obs[0] + obs_max[0], obs[1] + obs_max[1]
    z0 = ground_z
    z1 = obs[2] + obs_max[2]
    verts = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
    ]
    pc = Poly3DCollection(verts, facecolors='r', linewidths=0.5, edgecolors='r', alpha=0.1)
    ax.add_collection3d(pc)

# 7) Plot Spline
ax.plot(spline_pts[:,0], spline_pts[:,1], spline_pts[:,2], 'g-', label='Spline Path')
# 8) Mark Waypoints
ax.scatter(waypoints[:,0], waypoints[:,1], waypoints[:,2], marker='^', s=60, c='g', label='Waypoints')

# 9) Symmetrisches Grid / gleiche Achsenlängen
all_x = np.concatenate([spline_pts[:,0], gates[:,0], obstacles[:,0]])
all_y = np.concatenate([spline_pts[:,1], gates[:,1], obstacles[:,1]])
all_z = np.concatenate([spline_pts[:,2], gates[:,2], obstacles[:,2], [ground_z]])
dx, dy, dz = np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)
max_range = max(dx, dy, dz)
mid_x, mid_y, mid_z = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2, (all_z.max()+all_z.min())/2
ax.set_xlim(mid_x-max_range/2, mid_x+max_range/2)
ax.set_ylim(mid_y-max_range/2, mid_y+max_range/2)
ax.set_zlim(mid_z-max_range/2, mid_z+max_range/2)
ax.set_box_aspect([1,1,1])  # für Matplotlib ≥3.3

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Level 2: Gates inkl. Randomisierung, Obstacles & Spline mit Waypoints')
ax.legend()
plt.show()
