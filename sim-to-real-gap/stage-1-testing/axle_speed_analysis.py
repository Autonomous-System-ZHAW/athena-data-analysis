import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------
rootdir = "data-sim/"
try:  # als Skript: Ordner relativ zum Skript suchen (im Notebook gilt der Notebook-Ordner)
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), rootdir)
except NameError:
    pass

# Frame, dessen Bewegung die Zeitstempel bestimmt. Nur /tf-Nachrichten, die diesen
# Frame enthalten, ergeben eine Zeile - andere Publisher auf /tf (Räder, Federung)
# mit eigenen Stempeln können die Zeitachse so nicht verfälschen.
MOVING_FRAME = "Chassis"

# Fester Frame, in dem gemessen wird. Muss Ground Truth aus der Sim sein
# (z. B. "world"), NICHT die Odometrie des Ackermann-Controllers - sonst misst
# man nur, was der Controller selbst annimmt.
FIXED_FRAME = "world"
AXLES = ["back_axle", "front_axle"]

groups = ["left", "straight", "right"]
STEER = {"left": 0.36, "straight": 0.0, "right": -0.36}  # rad
V_CMD = 1.0  # vorgegebene Geschwindigkeit [m/s]
SKIP_S = 2.0  # Anfahrphase am Anfang ignorieren [s]
K = 5  # Halbbreite der Differenz für die Geschwindigkeit (glättet Rauschen)

titles = {g: f"{g}: {STEER[g]} rad, {V_CMD} m/s" for g in groups}


# ---------------------------------------------------------------------------
# TF-Hilfsfunktionen
# ---------------------------------------------------------------------------
def clean(name):
    return name.lstrip("/")


def tf_matrix(tr):
    """geometry_msgs/Transform -> 4x4 Matrix (Parent <- Child)."""
    t, q = tr.translation, tr.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        # Normieren: ein nicht normiertes Quaternion würde Positionen strecken
        x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def pose_in(frame, target, edges):
    """Pose von `frame` im Frame `target`: den Baum Kind -> Parent hochlaufen
    und die Transformationen aufmultiplizieren. None, wenn nicht verbunden."""
    T = np.eye(4)
    seen = set()
    while frame != target:
        if frame not in edges or frame in seen:
            return None
        seen.add(frame)
        parent, M = edges[frame]
        T = M @ T
        frame = parent
    return T


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------
def load_axles(path):
    """Liest /tf und /tf_static und gibt pro TF-Update die Position
    beider Achsen im FIXED_FRAME zurück."""
    edges = {}  # child -> (parent, 4x4)
    rows = []
    warned = set()
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, msg in reader.iter_decoded_messages(
            topics=["/tf", "/tf_static"]
        ):
            stamp = None
            for tr in msg.transforms:
                child = clean(tr.child_frame_id)
                parent = clean(tr.header.frame_id)
                q = tr.transform.rotation
                n = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
                if abs(n - 1) > 1e-3 and child not in warned:
                    warned.add(child)
                    print(
                        f"  WARNUNG: {parent} -> {child} hat ein nicht normiertes Quaternion "
                        f"(Länge {n:.3f}: x={q.x:.3f} y={q.y:.3f} z={q.z:.3f} w={q.w:.3f})"
                    )
                edges[child] = (parent, tf_matrix(tr.transform))
                if child == MOVING_FRAME:
                    stamp = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9

            if channel.topic != "/tf" or stamp is None:
                continue

            row = {"t": stamp}
            for a in AXLES:
                T = pose_in(a, FIXED_FRAME, edges)
                if T is None:
                    break
                row[f"{a}_x"], row[f"{a}_y"] = T[0, 3], T[1, 3]
                row[f"{a}_yaw"] = np.arctan2(T[1, 0], T[0, 0])
            else:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        # Hilft beim Debuggen: welche Verbindungen gibt es überhaupt?
        print("  Keine Achs-Posen gefunden. TF-Kanten (child <- parent):")
        for child, (parent, _) in sorted(edges.items()):
            print(f"    {child} <- {parent}")
        return df
    return df.sort_values("t").drop_duplicates(subset="t").reset_index(drop=True)


def normalize_start(df):
    """Start der Hinterachse auf (0, 0), Startrichtung +x.
    Beide Achsen bekommen dieselbe Transformation, damit die Geometrie erhalten bleibt.
    """
    df = df.copy()
    x0, y0 = df["back_axle_x"].iloc[0], df["back_axle_y"].iloc[0]
    yaw0 = df["back_axle_yaw"].iloc[0]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    for a in AXLES:
        dx, dy = df[f"{a}_x"] - x0, df[f"{a}_y"] - y0
        df[f"{a}_x"] = c * dx - s * dy
        df[f"{a}_y"] = s * dx + c * dy
    df["t"] = df["t"] - df["t"].iloc[0]
    return df


def add_speeds(df):
    """Geschwindigkeit = Weg / Zeit über ein Fenster von 2K Samples."""
    t = df["t"].to_numpy()
    for a in AXLES:
        x, y = df[f"{a}_x"].to_numpy(), df[f"{a}_y"].to_numpy()
        v = np.full(len(df), np.nan)
        if len(df) > 2 * K:
            dist = np.hypot(x[2 * K :] - x[: -2 * K], y[2 * K :] - y[: -2 * K])
            dt = t[2 * K :] - t[: -2 * K]
            v[K:-K] = dist / dt
        df[f"{a}_v"] = v
    return df


# ---------------------------------------------------------------------------
# Dateien einlesen  (Namen wie "stage-1-test-right-2-1")
# ---------------------------------------------------------------------------
dfs = {g: {} for g in groups}
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if not file.endswith(".mcap"):
            continue
        name = os.path.splitext(file)[0]
        if not name.startswith("stage-1-test"):  # nur die neuen Aufnahmen
            continue

        m = re.search(r"(left|straight|right)-(\d+(?:-\d+)*)", name)
        if not m:
            continue
        group, run = m.group(1), m.group(2)  # run z. B. "2-1"

        print(f"{name}:")
        df = load_axles(os.path.join(subdir, file))
        if df.empty:
            continue
        print(f"  {len(df)} TF-Updates")
        dfs[group][run] = add_speeds(normalize_start(df))


# ---------------------------------------------------------------------------
# Auswertung: welche Achse fährt mit V_CMD?
# ---------------------------------------------------------------------------
summary = []
for group in groups:
    for run, df in sorted(dfs[group].items()):
        steady = df[df["t"] > SKIP_S]
        vb, vf = steady["back_axle_v"].mean(), steady["front_axle_v"].mean()
        weg = np.hypot(np.diff(df["back_axle_x"]), np.diff(df["back_axle_y"])).sum()
        summary.append(
            {
                "group": group,
                "run": run,
                "Weg hinten [m]": weg,
                "Dauer [s]": df["t"].iloc[-1] - df["t"].iloc[0],
                "v_back": vb,
                "v_front": vf,
                "ratio front/back": vf / vb,
                "erwartet (1/cos)": 1 / np.cos(STEER[group]),
                "näher an V_CMD": (
                    "back" if abs(vb - V_CMD) < abs(vf - V_CMD) else "front"
                ),
            }
        )
summary = pd.DataFrame(summary)
print()
print(summary.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Plots: oben Bahnen, unten Geschwindigkeit über Zeit
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
colors = plt.cm.tab10.colors

for col, group in enumerate(groups):
    ax_xy, ax_v = axes[0, col], axes[1, col]
    for i, run in enumerate(sorted(dfs[group])):
        df = dfs[group][run]
        c = colors[i % len(colors)]

        ax_xy.plot(
            df["back_axle_x"],
            df["back_axle_y"],
            "-",
            color=c,
            lw=1,
            label=f"{run} hinten",
        )
        ax_xy.plot(
            df["front_axle_x"],
            df["front_axle_y"],
            "--",
            color=c,
            lw=1,
            label=f"{run} vorne",
        )
        ax_xy.scatter(0, 0, color="green", s=30, zorder=3)

        ax_v.plot(df["t"], df["back_axle_v"], "-", color=c, lw=1, label=f"{run} hinten")
        ax_v.plot(
            df["t"], df["front_axle_v"], "--", color=c, lw=1, label=f"{run} vorne"
        )

    ax_xy.set_title(titles[group])
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True)

    ax_v.axhline(V_CMD, color="k", lw=0.8, ls=":")
    ax_v.axvline(SKIP_S, color="gray", lw=0.8, ls=":")
    ax_v.set_xlabel("t [s]")
    ax_v.set_ylabel("v [m/s]")
    ax_v.grid(True)

    if dfs[group]:
        ax_xy.legend(fontsize=8)
        ax_v.legend(fontsize=8)

plt.tight_layout()
plt.show()
