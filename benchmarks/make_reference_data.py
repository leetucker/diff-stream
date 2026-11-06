from pathlib import Path

from diff_stream.simulate import simulate_panel

path = Path(__file__).resolve().parents[1] / "tests" / "data" / "reference_panel.csv"
path.parent.mkdir(exist_ok=True)
simulate_panel(240, 5, seed=794).write_csv(path)
print(path)
