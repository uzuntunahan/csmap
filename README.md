# CS2 Demo to 2D Replay (AWPy)

This project extracts rich data from a CS2 `.dem` file with AWPy and renders it in `replay_tool.html` as a 2D replay.

## What Is Extracted

- Player state: name, steamid, team, position `(X, Y, Z)`, health, armor, weapon, and money if available
- Events: kills, damages, shots, smokes, infernos, and bomb events
- Bomb data:
- Event records such as `plant`, `defuse`, and `explode`
- Bomb carrier route (`bomb_carrier_path`) when `has_bomb` is available
- Utility data:
- Grenade trajectories (`grenades`)
- Grenade landing points (`grenade_landings`)

## Data Extraction Options

### Option 1: Notebook (step by step)

Use `demo_to_data.ipynb` if you want to inspect the data while exporting.

1. Open `demo_to_data.ipynb` in VS Code.
2. Select a Python 3 kernel.
3. Run cells in order.
4. The notebook generates `demo_data.json` in the final export cell.

### Option 2: Command Line Script (single run)

Use `extract_demo_data.py` for repeatable one command exports.

```bash
py -m pip install awpy polars
py extract_demo_data.py vita-auro.dem -o demo_data.json --tick-sample 4 --grenade-sample 2
```

Arguments:

- `--tick-sample`: keeps one of every N ticks to reduce output size
- `--grenade-sample`: sparsifies grenade trajectory rows
- `--verbose`: enables AWPy parse logs

Note: On Windows, prefer `py` instead of `python`.

## Notebook vs Script

Both produce `demo_data.json`, but they are not exactly the same workflow.

- Notebook:
- Interactive analysis plus export
- Better for exploring tables and verifying data manually

- Script:
- One shot export from CLI
- Better for automation and consistent output
- Uses tick sampling that preserves all player rows for kept ticks

## Replay Tool Usage

1. Open `replay_tool.html` in a browser.
2. Load `demo_data.json`.
3. Select rounds, scrub with the tick slider, or press play.

## Visual Features in Replay Tool

- Bomb carrier route as a dashed path
- Grenade landing markers by utility type
- Round timeline panel for kill and bomb events
- Player trail view for recent movement
- Optional map image layer from `meta.map_image`
- Per map alignment controls (scale and X/Y offset) stored in localStorage

## Optional Map Image

- In export data, set `meta.map_image` to a local path or URL
- Example: `maps/de_dust2.png`
- Toggle map visibility with the `Map` checkbox in the controls

## Troubleshooting

- Extra props such as `cash` or `has_bomb` may be unavailable depending on demo or AWPy version
- The script falls back to base player props if extended parsing fails

### Python Not Found on Windows

1. Check launcher:

```bash
py --version
```

2. Install packages and run script:

```bash
py -m pip install awpy polars
py extract_demo_data.py vita-auro.dem -o demo_data.json
```

3. If needed, create a clean environment with Anaconda or another Python environment manager.
