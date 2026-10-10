import os

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# Ordner relativ zum Skript, egal von wo aus es gestartet wird
rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data-sim/")


def clean(name):
    return name.lstrip("/")


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def read(path):
    """Gibt zwei Arrays zurück, Spalten: Stempel, Log-Zeit, x, y.
    odom: aus /odom (Controller), tf: Verbindung odom -> Chassis aus /tf."""
    odom, tf = [], []
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, msg in reader.iter_decoded_messages(topics=["/odom", "/tf"]):
            log_t = message.log_time / 1e9
            if channel.topic == "/odom":
                p = msg.pose.pose.position
                odom.append((stamp(msg.header), log_t, p.x, p.y))
            else:
                for tr in msg.transforms:
                    if clean(tr.header.frame_id) == "odom" and clean(tr.child_frame_id) == "Chassis":
                        t = tr.transform.translation
                        tf.append((stamp(tr.header), log_t, t.x, t.y))
    return np.array(odom), np.array(tf)


def summary(name, a):
    if len(a) < 2:
        print(f"  {name:5s}: keine Daten")
        return
    a = a[np.argsort(a[:, 0], kind="stable")]
    dist = np.hypot(np.diff(a[:, 2]), np.diff(a[:, 3])).sum()   # zurückgelegter Weg
    d_stamp = a[-1, 0] - a[0, 0]
    d_log = a[:, 1].max() - a[:, 1].min()
    n_dup = len(a) - len(np.unique(a[:, 0]))                     # gleiche Stempel mehrfach?
    print(f"  {name:5s}: Weg {dist:6.2f} m | Stempel {d_stamp:6.2f} s -> {dist / d_stamp:4.2f} m/s"
          f" | Log {d_log:6.2f} s -> {dist / d_log:4.2f} m/s"
          f" | {len(a)} Nachr. ({n_dup} doppelte Stempel)"
          f" | Ende ({a[-1, 2]:.2f}, {a[-1, 3]:.2f})")


print(f"Suche in: {rootdir}\n")
for subdir, _, files in os.walk(rootdir):
    for file in sorted(files):
        if not (file.startswith("stage-1-test") and file.endswith(".mcap")):
            continue
        odom, tf = read(os.path.join(subdir, file))
        if len(tf) == 0:
            continue   # Dateien ohne odom -> Chassis in /tf überspringen
        print(file)
        summary("/odom", odom)
        summary("TF", tf)
        print()
