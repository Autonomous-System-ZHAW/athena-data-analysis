import os

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

rootdir = "data-sim/"
FRAME = "back_axle"


def clean(name):
    return name.lstrip("/")


def print_chain(path, frame=FRAME):
    """Zeigt die Kette von `frame` bis zur Wurzel und ob sich jede Verbindung bewegt."""
    edges, moves = {}, {}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(
            topics=["/tf", "/tf_static"]
        ):
            for tr in msg.transforms:
                c, p = clean(tr.child_frame_id), clean(tr.header.frame_id)
                pos = (
                    round(tr.transform.translation.x, 3),
                    round(tr.transform.translation.y, 3),
                )
                edges[c] = p
                moves.setdefault(c, set()).add(pos)

    print(os.path.basename(path))
    while frame in edges:
        state = "bewegt sich" if len(moves[frame]) > 1 else "fest"
        print(f"  {edges[frame]} -> {frame}: {state}")
        frame = edges[frame]
    print()


# Ordner relativ zum Skript, egal von wo aus es gestartet wird
rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), rootdir)
print(f"Suche in: {rootdir}")
if not os.path.isdir(rootdir):
    print("  Ordner existiert nicht! rootdir oben im Skript anpassen.")

# Alle neuen Aufnahmen prüfen
found = 0
for subdir, _, files in os.walk(rootdir):
    for file in sorted(files):
        if file.startswith("stage-1-test") and file.endswith(".mcap"):
            found += 1
            print_chain(os.path.join(subdir, file))

if found == 0:
    print(
        "  Keine Dateien gefunden, die mit 'stage-1-test' beginnen und auf .mcap enden."
    )
