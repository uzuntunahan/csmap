import argparse
import json
import math
from pathlib import Path
from typing import Any

import polars as pl
from awpy import Demo


BASE_PLAYER_PROPS = [
    "X",
    "Y",
    "Z",
    "health",
    "armor_value",
    "has_helmet",
    "has_defuser",
    "inventory",
    "current_equip_value",
    "team_name",
    "is_alive",
    "pitch",
    "yaw",
    "active_weapon",
]

EXTRA_PLAYER_PROPS = [
    "cash",
    "has_bomb",
]

DEFAULT_MAP_CALIBRATION = {
    "scale": 1.0,
    "offset_x": 0,
    "offset_y": 0,
}

MAP_CALIBRATION_PRESETS: dict[str, dict[str, float | int]] = {
    # Keep neutral defaults unless a map specific correction is verified.
    "de_dust2": {"scale": 1.0, "offset_x": 0, "offset_y": 0},
    "de_inferno": {"scale": 1.0, "offset_x": 0, "offset_y": 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract rich CS2 replay data from a .dem file using AWPy.",
    )
    parser.add_argument("demo_file", help="Path to CS2 .dem file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: <demo_file_name>.json)",
    )
    parser.add_argument(
        "--tick-sample",
        type=int,
        default=4,
        help="Keep one of every N ticks (default: 4)",
    )
    parser.add_argument(
        "--grenade-sample",
        type=int,
        default=2,
        help="Keep one of every N grenade trajectory rows (default: 2)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable AWPy verbose parsing",
    )
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    return False


def safe_value(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, dict):
        return {k: safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_value(v) for v in value]
    return value


def df_to_records(df: pl.DataFrame | None, sample_every: int = 1) -> list[dict[str, Any]]:
    if df is None or df.is_empty():
        return []
    if sample_every > 1:
        df = df[::sample_every]
    return [{k: safe_value(v) for k, v in row.items()} for row in df.to_dicts()]


def sample_ticks_preserving_players(ticks_df: pl.DataFrame | None, sample_every: int) -> pl.DataFrame | None:
    """Sample by unique tick values, then keep all player rows for kept ticks.

    Row slicing like df[::N] drops most players within a tick because ticks table has
    one row per player per tick. This keeps roster integrity for replay rendering.
    """
    if ticks_df is None or ticks_df.is_empty() or sample_every <= 1:
        return ticks_df
    if "tick" not in ticks_df.columns:
        return ticks_df

    kept_ticks = (
        ticks_df
        .select("tick")
        .unique(maintain_order=True)
        .sort("tick")
        .with_row_index("_i")
        .filter((pl.col("_i") % sample_every) == 0)
        .select("tick")
    )

    return ticks_df.join(kept_ticks, on="tick", how="inner")


def compute_map_bounds(ticks_df: pl.DataFrame | None, pad: int = 200) -> dict[str, float] | None:
    if ticks_df is None or ticks_df.is_empty() or "X" not in ticks_df.columns or "Y" not in ticks_df.columns:
        return None

    xs = [x for x in ticks_df["X"].to_list() if not is_missing(x)]
    ys = [y for y in ticks_df["Y"].to_list() if not is_missing(y)]
    if not xs or not ys:
        return None

    return {
        "min_x": float(min(xs) - pad),
        "max_x": float(max(xs) + pad),
        "min_y": float(min(ys) - pad),
        "max_y": float(max(ys) + pad),
    }


def detect_map_image(map_name: str) -> str | None:
    if not map_name or map_name == "unknown":
        return None
    candidate = Path("maps") / f"{map_name}.png"
    if candidate.exists():
        return f"maps/{map_name}.png"
    return None


def detect_map_calibration(map_name: str) -> dict[str, Any]:
    base = dict(DEFAULT_MAP_CALIBRATION)

    preset = MAP_CALIBRATION_PRESETS.get(map_name)
    if preset:
        base.update(preset)
        source = "preset"
    else:
        source = "default"

    sidecar = Path("maps") / f"{map_name}.calibration.json"
    if map_name and map_name != "unknown" and sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            scale = float(data.get("scale", base["scale"]))
            offset_x = int(data.get("offset_x", base["offset_x"]))
            offset_y = int(data.get("offset_y", base["offset_y"]))
            base.update({"scale": scale, "offset_x": offset_x, "offset_y": offset_y})
            source = "sidecar"
        except Exception as exc:
            print(f"Warning: failed to parse calibration sidecar '{sidecar}': {exc}")

    return {
        **base,
        "source": source,
    }


def has_cols(df: pl.DataFrame | None, cols: list[str]) -> bool:
    if df is None:
        return False
    return all(c in df.columns for c in cols)


def make_player_kills(kills_df: pl.DataFrame | None) -> list[dict[str, Any]]:
    if kills_df is None or kills_df.is_empty() or "attacker_steamid" not in kills_df.columns:
        return []

    cols = [c for c in ["attacker_steamid", "attacker_name", "attacker_team_name"] if c in kills_df.columns]
    grouped = (
        kills_df
        .filter(pl.col("attacker_steamid").is_not_null())
        .group_by(cols)
        .agg(pl.len().alias("kills"))
        .sort("kills", descending=True)
    )
    return grouped.to_dicts()


def make_player_stats(kills_df: pl.DataFrame | None) -> list[dict[str, Any]]:
    """Build per-player K/D/A stats from parsed kill events.

    Values are derived only from kill rows exported by AWPy:
    - kills   <- attacker_steamid
    - deaths  <- victim_steamid
    - assists <- assister_steamid
    """
    if kills_df is None or kills_df.is_empty():
        return []

    stats: dict[str, dict[str, Any]] = {}

    def ensure_player(
        steamid_value: Any,
        name_value: Any = None,
        team_value: Any = None,
    ) -> dict[str, Any] | None:
        if steamid_value is None:
            return None

        sid = str(steamid_value)
        if sid not in stats:
            stats[sid] = {
                "steamid": sid,
                "name": None,
                "team_name": None,
                "kills": 0,
                "deaths": 0,
                "assists": 0,
            }

        row = stats[sid]
        if row["name"] in (None, "") and name_value not in (None, ""):
            row["name"] = name_value
        if row["team_name"] in (None, "") and team_value not in (None, ""):
            row["team_name"] = team_value
        return row

    for k in kills_df.to_dicts():
        attacker = ensure_player(
            k.get("attacker_steamid"),
            k.get("attacker_name"),
            k.get("attacker_team_name") or k.get("attacker_side"),
        )
        if attacker is not None:
            attacker["kills"] += 1

        victim = ensure_player(
            k.get("victim_steamid"),
            k.get("victim_name"),
            k.get("victim_team_name") or k.get("victim_side"),
        )
        if victim is not None:
            victim["deaths"] += 1

        assister = ensure_player(
            k.get("assister_steamid"),
            k.get("assister_name"),
            k.get("assister_team_name") or k.get("assister_side"),
        )
        if assister is not None:
            assister["assists"] += 1

    rows = list(stats.values())
    rows.sort(key=lambda r: (r["kills"], -r["deaths"], r["assists"]), reverse=True)
    return rows


def make_grenade_landings(grenades_df: pl.DataFrame | None) -> list[dict[str, Any]]:
    if grenades_df is None or grenades_df.is_empty():
        return []

    required = ["round_num", "entity_id", "tick", "X", "Y", "Z"]
    if not has_cols(grenades_df, required):
        return []

    base = grenades_df.sort("tick")
    first_rows = base.group_by(["round_num", "entity_id"]).first()
    last_rows = base.group_by(["round_num", "entity_id"]).last()

    start_cols = [c for c in ["round_num", "entity_id", "tick", "thrower_name", "thrower_steamid", "grenade_type"] if c in first_rows.columns]
    end_cols = [c for c in ["round_num", "entity_id", "tick", "X", "Y", "Z"] if c in last_rows.columns]

    starts = first_rows.select(start_cols).rename({"tick": "start_tick"})
    ends = last_rows.select(end_cols).rename({"tick": "land_tick", "X": "land_X", "Y": "land_Y", "Z": "land_Z"})

    joined = starts.join(ends, on=["round_num", "entity_id"], how="inner")
    return joined.to_dicts()


def make_bomb_carrier_path(ticks_df: pl.DataFrame | None) -> list[dict[str, Any]]:
    if ticks_df is None or ticks_df.is_empty() or "has_bomb" not in ticks_df.columns:
        return []

    needed = [c for c in ["tick", "round_num", "steamid", "name", "X", "Y", "Z", "has_bomb"] if c in ticks_df.columns]
    if "tick" not in needed or "round_num" not in needed:
        return []

    carriers = (
        ticks_df
        .filter(pl.col("has_bomb") == True)
        .select(needed)
        .sort(["round_num", "tick"])
    )

    if carriers.is_empty():
        return []
    return carriers.to_dicts()


def parse_demo(args: argparse.Namespace) -> Demo:
    demo = Demo(args.demo_file, verbose=args.verbose)

    requested = BASE_PLAYER_PROPS + EXTRA_PLAYER_PROPS
    try:
        demo.parse(player_props=requested)
    except Exception as exc:
        print("Warning: parse with extended props failed, retrying with base props.")
        print(f"Reason: {exc}")
        demo = Demo(args.demo_file, verbose=args.verbose)
        demo.parse(player_props=BASE_PLAYER_PROPS)

    return demo


def build_export(demo: Demo, args: argparse.Namespace) -> dict[str, Any]:
    tick_sample = max(1, args.tick_sample)
    grenade_sample = max(1, args.grenade_sample)

    ticks_sampled = sample_ticks_preserving_players(demo.ticks, tick_sample)
    grenades_sampled = (
        demo.grenades[::grenade_sample]
        if demo.grenades is not None and not demo.grenades.is_empty()
        else demo.grenades
    )

    bomb_carrier_path = make_bomb_carrier_path(demo.ticks)
    grenade_landings = make_grenade_landings(demo.grenades)
    player_kills = make_player_kills(demo.kills)
    player_stats = make_player_stats(demo.kills)
    map_name = demo.header.get("map_name", "unknown")
    map_bounds = compute_map_bounds(demo.ticks)
    map_image = detect_map_image(map_name)
    map_calibration = detect_map_calibration(map_name)

    return {
        "meta": {
            "map_name": map_name,
            "server_name": demo.header.get("server_name", "unknown"),
            "demo_file": str(args.demo_file),
            "demo_version_name": demo.header.get("demo_version_name", "unknown"),
            "tick_sample": tick_sample,
            "grenade_sample": grenade_sample,
            "map_bounds": map_bounds,
            "map_image": map_image,
            "map_calibration": map_calibration,
            "has_bomb_carrier_path": len(bomb_carrier_path) > 0,
        },
        "rounds": df_to_records(demo.rounds),
        "ticks": df_to_records(ticks_sampled),
        "kills": df_to_records(demo.kills),
        "damages": df_to_records(demo.damages),
        "bomb": df_to_records(demo.bomb),
        "grenades": df_to_records(grenades_sampled),
        "smokes": df_to_records(demo.smokes),
        "infernos": df_to_records(demo.infernos),
        "shots": df_to_records(demo.shots),
        "bomb_carrier_path": [{k: safe_value(v) for k, v in row.items()} for row in bomb_carrier_path],
        "grenade_landings": [{k: safe_value(v) for k, v in row.items()} for row in grenade_landings],
        "player_kills": [{k: safe_value(v) for k, v in row.items()} for row in player_kills],
        "player_stats": [{k: safe_value(v) for k, v in row.items()} for row in player_stats],
    }


def main() -> None:
    args = parse_args()

    demo_path = Path(args.demo_file)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    output_path = Path(args.output) if args.output else demo_path.with_suffix(".json")

    print(f"Parsing demo: {demo_path}")
    demo = parse_demo(args)

    print("Building export payload...")
    payload = build_export(demo, args)

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Export complete: {output_path}")
    print(f"Map: {payload['meta']['map_name']}")
    print("Record counts:")
    for key in [
        "rounds",
        "ticks",
        "kills",
        "damages",
        "bomb",
        "grenades",
        "smokes",
        "infernos",
        "shots",
        "bomb_carrier_path",
        "grenade_landings",
        "player_kills",
        "player_stats",
    ]:
        print(f"  {key:<17} {len(payload.get(key, [])):,}")


if __name__ == "__main__":
    main()
