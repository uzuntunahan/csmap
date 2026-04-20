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


def normalize_side(value: Any) -> str | None:
    s = str(value or "").strip().lower()
    if s in {"ct", "counter-terrorist", "counter_terrorist", "counterterrorist"}:
        return "ct"
    if s in {"t", "terrorist", "terrorists"}:
        return "t"
    if "counter" in s or s.startswith("ct"):
        return "ct"
    if "terror" in s or s == "t-side":
        return "t"
    return None


def first_non_missing(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and not is_missing(row.get(key)):
            return row.get(key)
    return None


def side_from_row(row: dict[str, Any], keys: list[str]) -> str | None:
    return normalize_side(first_non_missing(row, keys))


def categorize_weapon(weapon: Any) -> str:
    w = str(weapon or "").lower().replace("weapon_", "")
    if not w:
        return "unknown"
    if "awp" in w or "ssg08" in w or "scar20" in w or "g3sg1" in w:
        return "sniper"
    if "ak47" in w or "m4" in w or "famas" in w or "galil" in w or "aug" in w or "sg553" in w:
        return "rifle"
    if "deagle" in w or "usp" in w or "glock" in w or "p250" in w or "five" in w or "cz75" in w:
        return "pistol"
    if "nova" in w or "xm1014" in w or "mag7" in w or "sawedoff" in w:
        return "shotgun"
    if "mp" in w or "p90" in w or "bizon" in w or "ump" in w or "mac10" in w:
        return "smg"
    if "m249" in w or "negev" in w:
        return "lmg"
    return "other"


def make_round_features(
    rounds_df: pl.DataFrame | None,
    ticks_df: pl.DataFrame | None,
    kills_df: pl.DataFrame | None,
    bomb_df: pl.DataFrame | None,
    grenades_df: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    if rounds_df is None or rounds_df.is_empty():
        return []

    round_rows = rounds_df.to_dicts()
    all_ticks = [] if ticks_df is None or ticks_df.is_empty() else ticks_df.to_dicts()
    all_kills = [] if kills_df is None or kills_df.is_empty() else kills_df.to_dicts()
    all_bomb = [] if bomb_df is None or bomb_df.is_empty() else bomb_df.to_dicts()
    all_grenades = [] if grenades_df is None or grenades_df.is_empty() else grenades_df.to_dicts()

    features: list[dict[str, Any]] = []

    for r in round_rows:
        rnum = r.get("round_num")
        freeze_end = int(r.get("freeze_end") or 0)
        round_end = int(r.get("end") or freeze_end)

        round_kills = sorted(
            [k for k in all_kills if k.get("round_num") == rnum],
            key=lambda x: x.get("tick") or 0,
        )
        round_bomb = sorted(
            [b for b in all_bomb if b.get("round_num") == rnum],
            key=lambda x: x.get("tick") or 0,
        )
        round_grenades = [g for g in all_grenades if g.get("round_num") == rnum]
        round_ticks = [
            t for t in all_ticks
            if t.get("round_num") == rnum and (t.get("tick") or 0) >= freeze_end and (t.get("tick") or 0) <= round_end
        ]

        # Entry and trade signals.
        first_kill = round_kills[0] if round_kills else None
        first_kill_team = (
            side_from_row(first_kill, ["attacker_team_name", "attacker_side", "attacker_team"])
            if first_kill else None
        )
        first_kill_tick = int(first_kill.get("tick") or 0) if first_kill else None
        first_kill_offset = (first_kill_tick - freeze_end) if first_kill_tick is not None else None
        first_kill_headshot = bool(first_kill.get("is_headshot") or first_kill.get("headshot")) if first_kill else False
        first_kill_weapon_bucket = categorize_weapon(first_kill.get("weapon") if first_kill else None)

        first_trade_happened = False
        first_trade_delay_ticks = None
        if first_kill:
            victim_sid = first_kill.get("victim_steamid")
            entry_tick = int(first_kill.get("tick") or 0)
            trade_window_end = entry_tick + 8 * 64
            for k in round_kills[1:]:
                ktick = int(k.get("tick") or 0)
                if ktick > trade_window_end:
                    break
                if victim_sid is not None and k.get("attacker_steamid") == victim_sid:
                    first_trade_happened = True
                    first_trade_delay_ticks = ktick - entry_tick
                    break

        # Kill split by side.
        kills_ct = 0
        kills_t = 0
        for k in round_kills:
            atk_side = side_from_row(k, ["attacker_team_name", "attacker_side", "attacker_team"])
            if atk_side == "ct":
                kills_ct += 1
            elif atk_side == "t":
                kills_t += 1

        # Utility usage split by thrower side.
        util_counts = {
            "ct_smokes": 0,
            "t_smokes": 0,
            "ct_flashes": 0,
            "t_flashes": 0,
            "ct_molotovs": 0,
            "t_molotovs": 0,
        }
        for g in round_grenades:
            gtype = str(g.get("grenade_type") or "").lower()
            thrower_side = side_from_row(g, ["thrower_team_name", "thrower_side", "team_name", "side"])
            if thrower_side not in {"ct", "t"}:
                continue
            key_prefix = "ct" if thrower_side == "ct" else "t"
            if gtype == "smoke":
                util_counts[f"{key_prefix}_smokes"] += 1
            elif gtype == "flashbang":
                util_counts[f"{key_prefix}_flashes"] += 1
            elif gtype in {"molotov", "incgrenade"}:
                util_counts[f"{key_prefix}_molotovs"] += 1

        # Bomb timeline signals.
        plants = [b for b in round_bomb if str(b.get("event") or "").lower() == "plant"]
        defuses = [b for b in round_bomb if str(b.get("event") or "").lower() == "defuse"]
        explodes = [b for b in round_bomb if str(b.get("event") or "").lower() == "explode"]
        plant_event = plants[-1] if plants else None
        plant_tick = int(plant_event.get("tick") or 0) if plant_event else None
        plant_site = str(plant_event.get("bombsite") or "") if plant_event else ""
        post_plant_ticks = (round_end - plant_tick) if plant_tick is not None else None

        # Economy at round start (first observed row per player after freeze_end).
        first_tick_by_player: dict[str, dict[str, Any]] = {}
        for t in sorted(round_ticks, key=lambda x: x.get("tick") or 0):
            sid = t.get("steamid")
            if sid is None:
                continue
            sid_key = str(sid)
            if sid_key not in first_tick_by_player:
                first_tick_by_player[sid_key] = t

        ct_cash_vals: list[float] = []
        t_cash_vals: list[float] = []
        ct_equip_vals: list[float] = []
        t_equip_vals: list[float] = []
        for row in first_tick_by_player.values():
            side = side_from_row(row, ["team_name", "side"])
            if side not in {"ct", "t"}:
                continue
            cash = row.get("cash")
            equip = row.get("current_equip_value")
            if side == "ct":
                if not is_missing(cash):
                    ct_cash_vals.append(float(cash))
                if not is_missing(equip):
                    ct_equip_vals.append(float(equip))
            else:
                if not is_missing(cash):
                    t_cash_vals.append(float(cash))
                if not is_missing(equip):
                    t_equip_vals.append(float(equip))

        def avg_or_none(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 2) if values else None

        features.append({
            "round_num": rnum,
            "winner": normalize_side(r.get("winner")) or str(r.get("winner") or "").lower(),
            "reason": str(r.get("reason") or ""),
            "duration_ticks": max(0, round_end - freeze_end),
            "first_kill_team": first_kill_team,
            "first_kill_tick_offset": first_kill_offset,
            "first_kill_headshot": first_kill_headshot,
            "first_kill_weapon_bucket": first_kill_weapon_bucket,
            "first_trade_happened": first_trade_happened,
            "first_trade_delay_ticks": first_trade_delay_ticks,
            "kills_ct": kills_ct,
            "kills_t": kills_t,
            "ct_smokes": util_counts["ct_smokes"],
            "t_smokes": util_counts["t_smokes"],
            "ct_flashes": util_counts["ct_flashes"],
            "t_flashes": util_counts["t_flashes"],
            "ct_molotovs": util_counts["ct_molotovs"],
            "t_molotovs": util_counts["t_molotovs"],
            "plant_happened": plant_event is not None,
            "plant_site": plant_site,
            "plant_tick_offset": (plant_tick - freeze_end) if plant_tick is not None else None,
            "defused": len(defuses) > 0,
            "exploded": len(explodes) > 0,
            "post_plant_ticks": post_plant_ticks,
            "ct_avg_cash_start": avg_or_none(ct_cash_vals),
            "t_avg_cash_start": avg_or_none(t_cash_vals),
            "ct_avg_equip_start": avg_or_none(ct_equip_vals),
            "t_avg_equip_start": avg_or_none(t_equip_vals),
        })

    return features


def make_round_rule_analysis(round_features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []

    for f in round_features:
        winner = normalize_side(f.get("winner"))
        entry = normalize_side(f.get("first_kill_team"))
        reason = str(f.get("reason") or "").lower()
        rules: list[dict[str, str]] = []
        duration_ticks = int(f.get("duration_ticks") or 0)
        first_kill_offset = f.get("first_kill_tick_offset")
        ct_smokes = int(f.get("ct_smokes") or 0)
        t_smokes = int(f.get("t_smokes") or 0)
        ct_flashes = int(f.get("ct_flashes") or 0)
        t_flashes = int(f.get("t_flashes") or 0)
        ct_molotovs = int(f.get("ct_molotovs") or 0)
        t_molotovs = int(f.get("t_molotovs") or 0)
        kills_ct = int(f.get("kills_ct") or 0)
        kills_t = int(f.get("kills_t") or 0)

        def add_rule(
            code: str,
            severity: str,
            text: str,
            side: str = "ct_t",
            judgement: str = "neutral",
        ) -> None:
            normalized_side = side if side in {"ct", "t", "ct_t"} else "ct_t"
            side_label = "CT" if normalized_side == "ct" else "T" if normalized_side == "t" else "CT-T"
            normalized_judgement = judgement if judgement in {"good", "fault", "neutral"} else "neutral"
            rules.append({
                "code": code,
                "severity": severity,
                "side": normalized_side,
                "judgement": normalized_judgement,
                "text": f"[{side_label}] {text}",
            })

        if entry in {"ct", "t"} and winner in {"ct", "t"}:
            if entry == winner:
                add_rule("R001", "info", "Entry kill alan taraf roundu kapatti (entry conversion pozitif).", side=entry, judgement="good")
            else:
                add_rule("R002", "high", "Entry kill avantaji rounda tasinamadi (conversion zayif).", side=entry, judgement="fault")

        if f.get("first_trade_happened") is True:
            delay = f.get("first_trade_delay_ticks")
            if isinstance(delay, (int, float)) and delay <= 256:
                add_rule("R003", "info", "Hizli trade var; spacing ve destek koordinasyonu iyi.", side=entry or "ct_t", judgement="good")
            else:
                add_rule("R004", "medium", "Trade gec geldi; ilk temas sonrasi alan kontrolu zayiflayabilir.", side=entry or "ct_t", judgement="fault")
        elif entry in {"ct", "t"}:
            add_rule("R005", "medium", "Entry sonrasi hizli trade yok; 5v4 avantaj korunmaliydi.", side=entry, judgement="fault")

        if f.get("plant_happened") is True:
            plant_offset = f.get("plant_tick_offset")
            post_plant = f.get("post_plant_ticks")
            if isinstance(plant_offset, (int, float)) and plant_offset > 1200:
                add_rule("R006", "medium", "Plant gec geldi; execute zamani sikisti.", side="t", judgement="fault")
            if winner == "t" and isinstance(post_plant, (int, float)) and post_plant >= 320:
                add_rule("R007", "info", "Post-plant suresi yeterli; afterplant disiplinli oynanmis.", side="t", judgement="good")
            if winner == "ct" and isinstance(post_plant, (int, float)) and post_plant <= 320:
                add_rule("R008", "high", "Plant sonrasi hizli kayip var; post-plant pozisyonlari kirilgan.", side="t", judgement="fault")
        else:
            if winner == "ct":
                add_rule("R009", "info", "Plant engellenmis; round kontrolu CT tarafinda kalmis.", side="ct", judgement="good")

        ct_utility = ct_smokes + ct_flashes + ct_molotovs
        t_utility = t_smokes + t_flashes + t_molotovs
        if ct_utility - t_utility >= 3:
            add_rule("R010", "info", "CT utility temposu ustun; alan yavaslatma basarili.", side="ct", judgement="good")
        elif t_utility - ct_utility >= 3:
            add_rule("R011", "info", "T utility temposu ustun; execute hazirligi guclu.", side="t", judgement="good")

        ct_cash = f.get("ct_avg_cash_start")
        t_cash = f.get("t_avg_cash_start")
        if isinstance(ct_cash, (int, float)) and isinstance(t_cash, (int, float)):
            if ct_cash - t_cash >= 1500 and winner == "t":
                add_rule("R012", "high", "Ekonomi avantaji CT tarafinda olmasina ragmen round kaybedildi.", side="ct", judgement="fault")
            if t_cash - ct_cash >= 1500 and winner == "ct":
                add_rule("R013", "high", "Ekonomi avantaji T tarafinda olmasina ragmen round kaybedildi.", side="t", judgement="fault")

        if reason == "explode":
            add_rule("R014", "medium", "Bomb explode: retake gec kalmis veya utility yetersiz kalmis olabilir.", side="ct", judgement="fault")
        if reason == "defuse":
            add_rule("R015", "info", "Defuse roundu: retake zamanlamasi ve trade zinciri calismis.", side="ct", judgement="good")

        # R016-R030: extended pace/execute/upset rules.
        if isinstance(first_kill_offset, (int, float)) and first_kill_offset >= 45 * 64:
            add_rule("R016", "medium", "Ilk temas gec geldi; tempo dusuk ve bilgi oyunu agir basmis.", side="ct_t", judgement="neutral")

        if isinstance(first_kill_offset, (int, float)) and first_kill_offset <= 12 * 64 and entry in {"ct", "t"}:
            if winner == entry:
                add_rule("R017", "info", "Erken temas avantaji hizli sekilde skora cevrildi.", side=entry, judgement="good")
            elif winner in {"ct", "t"}:
                add_rule("R018", "medium", "Erken agresyon avantaji korunamadi; tempo kontrolu kaybedildi.", side=entry, judgement="fault")

        if f.get("plant_happened") is True and t_smokes == 0:
            add_rule("R019", "high", "Plant roundunda T smoke kullanimi yok; execute katmani eksik.", side="t", judgement="fault")

        if f.get("plant_happened") is True and (t_smokes + t_molotovs) < 2:
            add_rule("R020", "medium", "Plant oncesi alan acma utilitysi zayif; post-plant kirilganligi artmis olabilir.", side="t", judgement="fault")

        if reason == "defuse" and ct_utility < 2:
            add_rule("R021", "info", "Dusuk utility ile defuse alindi; CT retake bireysel karar kalitesi yuksek.", side="ct", judgement="good")

        if reason == "explode" and ct_utility >= (t_utility + 3):
            add_rule("R022", "high", "CT utility ustunlugune ragmen bomb explode; utility degeri skora donusmedi.", side="ct", judgement="fault")

        if winner == "t" and kills_t >= 5 and duration_ticks <= 25 * 64:
            add_rule("R023", "info", "Hizli T roundu: execute temiz ve round kapanisi hizli.", side="t", judgement="good")

        if winner == "ct" and not f.get("plant_happened") and duration_ticks <= 22 * 64:
            add_rule("R024", "info", "Plant verilmeden erken CT kapanisi; map kontrolu guclu.", side="ct", judgement="good")

        if isinstance(ct_cash, (int, float)) and isinstance(t_cash, (int, float)):
            if ct_cash >= (t_cash + 1500) and winner == "t":
                add_rule("R025", "info", "T tarafi ekonomik dezavantajdan round cikardi (upset win).", side="t", judgement="good")
            if t_cash >= (ct_cash + 1500) and winner == "ct":
                add_rule("R026", "info", "CT tarafi ekonomik dezavantajdan round cikardi (upset win).", side="ct", judgement="good")

        post_plant = f.get("post_plant_ticks")
        if f.get("plant_happened") is True and isinstance(post_plant, (int, float)):
            if winner == "ct" and post_plant <= 160:
                add_rule("R027", "high", "Plant sonrasi cok hizli dusus; T afterplant setup dagilmis.", side="t", judgement="fault")
            if winner == "ct" and post_plant >= 700:
                add_rule("R028", "info", "Uzun post-plant savasi sonunda CT roundu aldi; retake sabri guclu.", side="ct", judgement="good")

        if (ct_flashes + t_flashes) == 0 and duration_ticks >= 40 * 64:
            add_rule("R029", "medium", "Uzun rounda ragmen flash kullanimi yok; bilgi ve giris senkronu sinirli kalmis olabilir.", side="ct_t", judgement="fault")

        if abs(ct_utility - t_utility) <= 1 and winner in {"ct", "t"}:
            add_rule("R030", "info", "Utility dengesi yakin; round sonucu daha cok duel ve pozisyon kalitesiyle belirlenmis.", side="ct_t", judgement="neutral")

        if not rules:
            add_rule("R000", "info", "Bu round icin belirgin bir kritik sinyal yakalanmadi.", side="ct_t", judgement="neutral")

        score = sum(3 if r["severity"] == "high" else 2 if r["severity"] == "medium" else 1 for r in rules)
        analyses.append({
            "round_num": f.get("round_num"),
            "score": score,
            "summary": f"Round {f.get('round_num')}: {len(rules)} sinyal uretildi.",
            "rules": rules,
        })

    return analyses


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


def make_round_player_analysis(
    rounds_df: pl.DataFrame | None,
    ticks_df: pl.DataFrame | None,
    kills_df: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    if rounds_df is None or rounds_df.is_empty():
        return []

    round_rows = rounds_df.to_dicts()
    all_ticks = [] if ticks_df is None or ticks_df.is_empty() else ticks_df.to_dicts()
    all_kills = [] if kills_df is None or kills_df.is_empty() else kills_df.to_dicts()
    trade_window_ticks = 8 * 64
    out: list[dict[str, Any]] = []

    for r in round_rows:
        rnum = r.get("round_num")
        round_ticks = [t for t in all_ticks if t.get("round_num") == rnum]
        round_kills = sorted(
            [k for k in all_kills if k.get("round_num") == rnum],
            key=lambda x: x.get("tick") or 0,
        )

        players: dict[str, dict[str, Any]] = {}

        def ensure_player(
            steamid_value: Any,
            name_value: Any = None,
            side_value: Any = None,
        ) -> dict[str, Any] | None:
            if steamid_value is None:
                return None
            sid = str(steamid_value)
            if sid not in players:
                players[sid] = {
                    "steamid": sid,
                    "name": name_value or sid,
                    "side": normalize_side(side_value),
                    "kills": 0,
                    "deaths": 0,
                    "assists": 0,
                    "entry_kill": 0,
                    "entry_death": 0,
                    "trade_kills": 0,
                    "untraded_deaths": 0,
                    "multikill": 0,
                }
            row = players[sid]
            if (not row.get("name")) and name_value:
                row["name"] = name_value
            if row.get("side") is None:
                row["side"] = normalize_side(side_value)
            return row

        # Seed known participants from ticks to avoid missing zero-kill players.
        for t in round_ticks:
            ensure_player(t.get("steamid"), t.get("name"), first_non_missing(t, ["team_name", "side"]))

        # Entry kill/death markers.
        if round_kills:
            fk = round_kills[0]
            fk_att = ensure_player(
                fk.get("attacker_steamid"),
                fk.get("attacker_name"),
                first_non_missing(fk, ["attacker_team_name", "attacker_side", "attacker_team"]),
            )
            fk_vic = ensure_player(
                fk.get("victim_steamid"),
                fk.get("victim_name"),
                first_non_missing(fk, ["victim_team_name", "victim_side", "victim_team"]),
            )
            if fk_att:
                fk_att["entry_kill"] += 1
            if fk_vic:
                fk_vic["entry_death"] += 1

        # Core K/D/A accumulation.
        for k in round_kills:
            atk = ensure_player(
                k.get("attacker_steamid"),
                k.get("attacker_name"),
                first_non_missing(k, ["attacker_team_name", "attacker_side", "attacker_team"]),
            )
            vic = ensure_player(
                k.get("victim_steamid"),
                k.get("victim_name"),
                first_non_missing(k, ["victim_team_name", "victim_side", "victim_team"]),
            )
            ast = ensure_player(
                k.get("assister_steamid"),
                k.get("assister_name"),
                first_non_missing(k, ["assister_team_name", "assister_side", "assister_team"]),
            )

            if atk is not None:
                atk["kills"] += 1
            if vic is not None:
                vic["deaths"] += 1
            if ast is not None:
                ast["assists"] += 1

        # Multi-kill count per player in round.
        for row in players.values():
            row["multikill"] = row["kills"]

        # Build death events for trade checks.
        death_events: list[dict[str, Any]] = []
        for k in round_kills:
            vic_sid = k.get("victim_steamid")
            atk_sid = k.get("attacker_steamid")
            tick = int(k.get("tick") or 0)
            vic_side = side_from_row(k, ["victim_team_name", "victim_side", "victim_team"])
            atk_side = side_from_row(k, ["attacker_team_name", "attacker_side", "attacker_team"])
            death_events.append({
                "tick": tick,
                "victim_sid": str(vic_sid) if vic_sid is not None else None,
                "attacker_sid": str(atk_sid) if atk_sid is not None else None,
                "victim_side": vic_side,
                "attacker_side": atk_side,
            })

        # Trade kills: attacker avenges recent teammate death by killing that killer.
        for k in round_kills:
            atk_sid = k.get("attacker_steamid")
            vic_sid = k.get("victim_steamid")
            if atk_sid is None or vic_sid is None:
                continue
            atk_key = str(atk_sid)
            atk_side = side_from_row(k, ["attacker_team_name", "attacker_side", "attacker_team"])
            ktick = int(k.get("tick") or 0)

            is_trade = False
            for d in death_events:
                if d["tick"] >= ktick:
                    continue
                if ktick - d["tick"] > trade_window_ticks:
                    continue
                if d["victim_side"] != atk_side:
                    continue
                if d["attacker_sid"] != str(vic_sid):
                    continue
                is_trade = True
                break

            if is_trade and atk_key in players:
                players[atk_key]["trade_kills"] += 1

        # Untraded deaths: no same-side revenge on killer within trade window.
        for d in death_events:
            victim_sid = d["victim_sid"]
            attacker_sid = d["attacker_sid"]
            dside = d["victim_side"]
            dtick = d["tick"]
            if victim_sid is None or attacker_sid is None or dside is None:
                continue

            traded = False
            for k in round_kills:
                ktick = int(k.get("tick") or 0)
                if ktick <= dtick:
                    continue
                if ktick - dtick > trade_window_ticks:
                    break
                k_att_side = side_from_row(k, ["attacker_team_name", "attacker_side", "attacker_team"])
                if k_att_side != dside:
                    continue
                if str(k.get("victim_steamid")) == attacker_sid:
                    traded = True
                    break

            if not traded and victim_sid in players:
                players[victim_sid]["untraded_deaths"] += 1

        # Score + textual signals.
        player_rows: list[dict[str, Any]] = []
        for p in players.values():
            impact = (
                p["kills"] * 2.0
                + p["assists"] * 1.0
                + p["trade_kills"] * 1.5
                + p["entry_kill"] * 2.0
                + max(0, p["multikill"] - 1) * 1.5
            )
            fault = (
                p["deaths"] * 1.0
                + p["untraded_deaths"] * 2.0
                + p["entry_death"] * 1.5
            )
            net = impact - fault

            judgement = "neutral"
            if net >= 2.0:
                judgement = "good"
            elif net <= -2.0:
                judgement = "fault"

            signals: list[str] = []
            if p["entry_kill"] > 0:
                signals.append("Entry kill aldi")
            if p["entry_death"] > 0:
                signals.append("Entry duel kaybi")
            if p["trade_kills"] > 0:
                signals.append(f"Trade kill x{p['trade_kills']}")
            if p["untraded_deaths"] > 0:
                signals.append(f"Untraded death x{p['untraded_deaths']}")
            if p["kills"] >= 2:
                signals.append(f"Multi-kill x{p['kills']}")

            player_rows.append({
                "steamid": p["steamid"],
                "name": p["name"],
                "side": p.get("side") or "ct_t",
                "kills": p["kills"],
                "deaths": p["deaths"],
                "assists": p["assists"],
                "impact_score": round(impact, 2),
                "fault_score": round(fault, 2),
                "net_score": round(net, 2),
                "judgement": judgement,
                "signals": signals[:5],
            })

        player_rows.sort(key=lambda x: x["net_score"], reverse=True)

        top_impact = player_rows[0] if player_rows else None
        top_fault = None
        if player_rows:
            worst = sorted(player_rows, key=lambda x: x["net_score"])[0]
            if worst["net_score"] < 0:
                top_fault = worst

        out.append({
            "round_num": rnum,
            "top_impact_player": top_impact,
            "top_fault_player": top_fault,
            "players": player_rows,
        })

    return out


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
    round_features = make_round_features(demo.rounds, demo.ticks, demo.kills, demo.bomb, demo.grenades)
    round_rule_analysis = make_round_rule_analysis(round_features)
    round_player_analysis = make_round_player_analysis(demo.rounds, demo.ticks, demo.kills)
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
        "round_features": [{k: safe_value(v) for k, v in row.items()} for row in round_features],
        "round_rule_analysis": [{k: safe_value(v) for k, v in row.items()} for row in round_rule_analysis],
        "round_player_analysis": [{k: safe_value(v) for k, v in row.items()} for row in round_player_analysis],
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
        "round_features",
        "round_rule_analysis",
        "round_player_analysis",
    ]:
        print(f"  {key:<17} {len(payload.get(key, [])):,}")


if __name__ == "__main__":
    main()
