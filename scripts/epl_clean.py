#!/usr/bin/env python3
"""
EPL data-cleaning pipeline  (seasons 2021-22 ... 2025-26)
=========================================================

Reads the 15 FBref-style EPL CSVs plus the (Premier League / E0) Matches.csv and writes
analysis-ready CSVs.  The raw files are never modified.

Run:
    python epl_clean.py --raw-dir /path/to/epl_csvs --matches /path/to/Matches.csv --out ./out

Steps (each one is a function below, run in this order because later steps depend on earlier ones):

  1  LOAD            read everything as text, repair the header, trim whitespace, blank -> NULL
  2  STRUCTURE       drop repeated header rows inside the player/keeper files, drop junk columns
  3  TEAM NAMES      strip the "vs " prefix in opponent files; one canonical club-name list
  4  TYPES           turn text into integers / decimals
  5  PEOPLE          players + keepers: split Nation, split Pos, add player_key and flags
  6  PENALTIES       rebuild the missing 2025-26 PKcon, drop the unreliable PKwon
  7  LEAGUE TABLES   parse Notes + Top Team Scorer, add points_deduction, own-goal columns
  8  MATCHES         merge split Div column, ISO dates, canonical names, bad shots, sparse odds
  9  COLUMN NAMES    snake_case for SQL / Power BI / Tableau, with a raw -> clean map
 10  WRITE + REPORT  CSV output, cleaning log, column map, remaining-null report

Design rules: nothing is silently invented (a value that cannot be known stays NULL), every
change is written to reports/cleaning_log.csv, and anything unexpected stops the run.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- configuration
RAW_FILES = {  # logical name -> raw file name (note the raw typo "oppenent")
    "players_stats": "epl_players_stats_2021_2026.csv",
    "players_shooting": "epl_players_shooting_2021_2026.csv",
    "keepers_stats": "epl_keepers_stats_2021_2026.csv",
    "squad_stats": "epl_squad_stats_2021_2026.csv",
    "squad_shooting": "epl_squad_shooting_2021_2026.csv",
    "squad_goalkeeping": "epl_squad_goalkeeping_2021_2026.csv",
    "squad_misc": "epl_squad_misc_2021_2026.csv",
    "squad_playingtime": "epl_squad_playingtime_2021_2026.csv",
    "opponent_stats": "epl_opponent_stats_2021_2026.csv",
    "opponent_shooting": "epl_oppenent_shooting_2021_2026.csv",
    "opponent_goalkeeping": "epl_oppenent_goalkeeping_2021_2026.csv",
    "opponent_misc": "epl_oppenent_misc_2021_2026.csv",
    "opponent_playingtime": "epl_oppenent_playingtime_2021_2026.csv",
    "rank_overall": "epl_rank_overall_2021_2026.csv",
    "rank_home_away": "epl_rank_home_away_2021_2026.csv",
}
MATCHES_OUT = "epl_matches_2021_2026.csv"
MATCHES_ODDS_OUT = "epl_matches_odds_extra_2021_2026.csv"

PEOPLE = ["players_stats", "players_shooting", "keepers_stats"]
TEAM_TABLES = [k for k in RAW_FILES if k not in PEOPLE]
OPPONENT_TABLES = [k for k in RAW_FILES if k.startswith("opponent_")]
EXPECTED_TEAMS_PER_SEASON = 20
ODDS_MIN_FILL = 0.95   # an odds column is kept in the main matches file only if >= 95% filled in EVERY season

# Matches.csv name -> name used in all the FBref-style files
MATCHES_TEAM_MAP = {
    "Man City": "Manchester City", "Man United": "Manchester Utd", "Nott'm Forest": "Nottingham",
    "Leeds": "Leeds United", "Leicester": "Leicester City", "Norwich": "Norwich City",
    "Ipswich": "Ipswich Town", "Luton": "Luton Town",
}
# readable names for the match-result / match-stat columns (odds columns get generic snake_case)
MATCHES_RENAME = {
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "ft_home_goals", "FTAG": "ft_away_goals", "FTR": "ft_result",
    "HTHG": "ht_home_goals", "HTAG": "ht_away_goals", "HTR": "ht_result",
    "HS": "home_shots", "AS": "away_shots", "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HF": "home_fouls", "AF": "away_fouls", "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellows", "AY": "away_yellows", "HR": "home_reds", "AR": "away_reds",
}
MATCH_CORE = ["Season", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
              "HTR", "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]


# --------------------------------------------------------------------------- helpers
class Ctx:
    """Holds the data frames and the audit trail while the pipeline runs."""

    def __init__(self) -> None:
        self.frames: dict[str, pd.DataFrame] = {}
        self.orig_cols: dict[str, list[str]] = {}
        self.dropped: dict[str, list[tuple[str, str]]] = {}
        self.log: list[dict] = []
        self.canonical_teams: set[str] = set()

    def note(self, step: int, dataset: str, action: str, n: int | None = None, detail: str = "") -> None:
        self.log.append({"step": step, "dataset": dataset, "action": action, "rows_affected": n, "detail": detail})

    def drop(self, name: str, cols: list[str], reason: str) -> None:
        df = self.frames[name]
        present = [c for c in cols if c in df.columns]
        self.frames[name] = df.drop(columns=present)
        for c in present:
            self.dropped.setdefault(name, []).append((c, reason))


def clean_header(c: str) -> str:
    """Remove stray BOM bytes (also in their mojibake form 'ï»¿') and whitespace from a header."""
    return c.replace("﻿", "").replace("ï»¿", "").strip()


def snake(col: str) -> str:
    """Raw header -> SQL/BI-friendly snake_case.   'Per 90 Minutes_G+A-PK' -> 'per_90_minutes_g_plus_a_minus_pk'."""
    special = {"# Pl": "n_players", "90s": "playing_time_90s"}  # '90s' = same measure as 'Playing Time_90s'
    if col in special:
        return special[col]
    s = col.strip()
    s = s.replace("+/-", " plus_minus ").replace("+", " plus ").replace("-", " minus ")
    s = s.replace("%", " pct ").replace("/", " per ").replace(">", " over ").replace("<", " under ").replace(".", "_")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return ("x_" + s) if s[0].isdigit() else s


def season_start(season: pd.Series) -> pd.Series:
    return season.str[:4].astype(int)


# --------------------------------------------------------------------------- step 1: load
def read_text_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def coalesce_duplicate_headers(df: pd.DataFrame, ctx: Ctx, name: str) -> pd.DataFrame:
    """Matches.csv has the division code in two columns ('Div' and a BOM-polluted 'ï»¿Div'), each filled for
    different seasons. After cleaning the header both are called 'Div' -> merge them into one column."""
    cleaned = [clean_header(c) for c in df.columns]
    out: dict[str, pd.Series] = {}
    for raw, new in zip(df.columns, cleaned):
        s = df[raw]
        if new not in out:
            out[new] = s
            continue
        both = (out[new] != "") & (s != "")
        if (out[new][both] != s[both]).any():
            raise ValueError(f"{name}: duplicate columns {new!r} disagree")
        merged = out[new].where(out[new] != "", s)
        ctx.note(1, name, f"merged duplicate column '{new}'", int((s != "").sum()),
                 f"second copy supplied {int(((out[new] == '') & (s != '')).sum())} values")
        out[new] = merged
    return pd.DataFrame(out)


def normalise_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Trim whitespace in every cell and turn empty strings into NULL."""
    changed = 0
    for c in df.columns:
        s = df[c]
        t = s.str.strip()
        changed += int((s != t).sum())
        df[c] = t.where(t != "", np.nan)
    return df, changed


def step1_load(ctx: Ctx, raw_dir: Path, matches_path: Path) -> None:
    for name, fn in RAW_FILES.items():
        df = read_text_csv(raw_dir / fn)
        ctx.orig_cols[name] = list(df.columns)
        df, changed = normalise_cells(df)
        ctx.frames[name] = df
        ctx.note(1, name, "loaded", len(df), f"{df.shape[1]} columns; {changed} cells trimmed")
    m = read_text_csv(matches_path)
    ctx.orig_cols["matches"] = list(m.columns)
    m = coalesce_duplicate_headers(m, ctx, "matches")
    m, changed = normalise_cells(m)
    ctx.frames["matches"] = m
    ctx.note(1, "matches", "loaded", len(m), f"{m.shape[1]} columns; {changed} cells trimmed (e.g. 'J Gillett ')")


# --------------------------------------------------------------------------- step 2: structure
def step2_structure(ctx: Ctx) -> None:
    for name in PEOPLE:
        df = ctx.frames[name]
        # FBref repeats the column-header row every ~26 rows; those rows have Player == 'Player'.
        hdr = df["Player"] == "Player"
        ctx.note(2, name, "dropped repeated header rows", int(hdr.sum()))
        df = df[~hdr].reset_index(drop=True)
        ctx.frames[name] = df
        # Rk is a per-season row counter (alphabetical), 'Matches' is the text of a hyperlink.
        ctx.drop(name, ["Rk"], "row counter, not data")
        ctx.drop(name, ["Matches"], "hyperlink text, constant 'Matches'")


# --------------------------------------------------------------------------- step 3: team names
def step3_team_names(ctx: Ctx) -> None:
    for name in OPPONENT_TABLES:
        df = ctx.frames[name]
        if not df["Squad"].str.startswith("vs ").all():
            raise ValueError(f"{name}: expected every Squad to start with 'vs '")
        df["Squad"] = df["Squad"].str.removeprefix("vs ")
        ctx.note(3, name, "removed 'vs ' prefix from Squad", len(df),
                 "row now describes what the opponents did AGAINST this squad")
    ctx.canonical_teams = set(ctx.frames["squad_stats"]["Squad"])
    for name in TEAM_TABLES:
        df = ctx.frames[name]
        if set(df["Squad"]) != ctx.canonical_teams:
            raise ValueError(f"{name}: club names differ from canonical set: {set(df['Squad']) ^ ctx.canonical_teams}")
        per_season = df.groupby("Season")["Squad"].nunique()
        if not (per_season == EXPECTED_TEAMS_PER_SEASON).all():
            raise ValueError(f"{name}: not 20 clubs in every season: {per_season.to_dict()}")
        if df.duplicated(["Squad", "Season"]).any():
            raise ValueError(f"{name}: duplicate (Squad, Season)")
    for name in PEOPLE:
        extra = set(ctx.frames[name]["Squad"]) - ctx.canonical_teams
        if extra:
            raise ValueError(f"{name}: unknown clubs {extra}")
        ctx.frames[name]["Squad"] = ctx.frames[name]["Squad"].str.replace(r"\s+", " ", regex=True)
        ctx.frames[name]["Player"] = ctx.frames[name]["Player"].str.replace(r"\s+", " ", regex=True)
    ctx.note(3, "all team tables", "club names verified", None,
             f"{len(ctx.canonical_teams)} canonical clubs, 20 per season, (Squad, Season) unique")


# --------------------------------------------------------------------------- step 4: types
def cast_numeric(df: pd.DataFrame, skip: set[str] = frozenset()) -> int:
    """A column becomes numeric only if EVERY non-null value is a number.  It becomes an integer column
    only if every value is written as an integer in the source ('38' yes, '38.0' no) -> '90s' stays decimal."""
    n = 0
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        nn = s.dropna()
        if nn.empty or pd.to_numeric(nn, errors="coerce").isna().any():
            continue
        if nn.str.fullmatch(r"-?\d+").all():
            df[c] = pd.to_numeric(s).astype("Int64")
        else:
            df[c] = pd.to_numeric(s).astype("float64")
        n += 1
    return n


def step4_types(ctx: Ctx) -> None:
    for name in PEOPLE + TEAM_TABLES:
        n = cast_numeric(ctx.frames[name])
        ctx.note(4, name, "text -> numeric", n, "columns converted")


# --------------------------------------------------------------------------- step 5: players & keepers
def step5_people(ctx: Ctx) -> None:
    for name in PEOPLE:
        df = ctx.frames[name]
        # Nation: 'eng ENG' = lowercase flag code + 3-letter code (1:1 mapping, checked) -> keep the 3-letter code.
        nat = df["Nation"].str.extract(r"^[a-z\-]+\s+([A-Z]{3})$")[0]
        if (nat.isna() != df["Nation"].isna()).any():
            raise ValueError(f"{name}: unexpected Nation format")
        ctx.note(5, name, "Nation -> 3-letter code", int(nat.notna().sum()),
                 f"{int(nat.isna().sum())} blank Nation stay NULL (cannot be recovered from other seasons)")
        df["Nation"] = nat
        # Pos: order is meaningful in FBref ('FW,MF' = mainly forward) -> primary / secondary position.
        parts = df["Pos"].str.split(",", expand=True)
        df["pos_primary"] = parts[0]
        df["pos_secondary"] = parts[1] if parts.shape[1] > 1 else np.nan
        ctx.note(5, name, "Pos split into pos_primary / pos_secondary", int(df["pos_secondary"].notna().sum()),
                 "rows with two positions; original Pos kept")
        # player_key: names alone are NOT unique ('Joshua King' = two different people, born 1992 and 2007).
        df["player_key"] = df["Player"] + "|" + df["Born"].astype("string").fillna("NA")
        if df.duplicated(["player_key", "Squad", "Season"]).any():
            raise ValueError(f"{name}: duplicate (player_key, Squad, Season)")
        nat_per_key = df.dropna(subset=["Nation"]).groupby("player_key")["Nation"].nunique()
        if (nat_per_key > 1).any():
            raise ValueError(f"{name}: one player_key with several nationalities")
        # flags (rows are kept, analysts decide)
        c90 = "Playing Time_90s" if "Playing Time_90s" in df.columns else "90s"
        df["low_minutes_flag"] = (df[c90] < 1).astype(int)
        df["multi_club_season_flag"] = (df.groupby(["Season", "player_key"])["Squad"].transform("nunique") > 1).astype(int)
        ctx.note(5, name, "added low_minutes_flag (<1 full 90)", int(df["low_minutes_flag"].sum()),
                 "per-90 rates are unstable on these rows")
        ctx.note(5, name, "added multi_club_season_flag", int(df["multi_club_season_flag"].sum()),
                 "mid-season transfer: one row per club")
        # sanity (information only): Age should equal season start - Born, or one less
        d = season_start(df["Season"]) - df["Born"] - df["Age"]
        off = int((~d.isin([0, 1]) & d.notna()).sum())
        ctx.note(5, name, "age/born consistency check", off, "rows where Age is inconsistent with Born (expected 0)")
        ctx.frames[name] = df
    a, b = ctx.frames["players_stats"], ctx.frames["players_shooting"]
    k = ["Player", "Squad", "Season", "Born"]
    aligned = a[k].astype("string").fillna("NA").equals(b[k].astype("string").fillna("NA"))
    ctx.note(5, "players_shooting", "row alignment with players_stats", len(b),
             "identical keys in identical order" if aligned else "NOT aligned - merge on keys, not position")
    if not aligned:
        raise ValueError("players_shooting is not row-aligned with players_stats")


# --------------------------------------------------------------------------- step 6: penalties
def rebuild_pkcon(misc: pd.DataFrame, pkatt_source: pd.DataFrame, label: str, ctx: Ctx) -> pd.DataFrame:
    """PKcon (penalties conceded) is blank for all of 2025-26 but it is exactly the opposing side's PKatt
    (verified 80/80 reported rows in each file), so the blanks are rebuilt from a complete column."""
    m = misc.merge(pkatt_source.rename(columns={"Performance_PKatt": "_pkatt"}), on=["Squad", "Season"],
                   how="left", validate="1:1")
    missing = m["Performance_PKcon"].isna()
    if m["_pkatt"].isna().any():
        raise ValueError(f"{label}: PKatt source incomplete")
    if (m.loc[~missing, "Performance_PKcon"] != m.loc[~missing, "_pkatt"]).any():
        raise ValueError(f"{label}: reported PKcon disagrees with PKatt - rule invalid")
    m["pkcon_source"] = np.where(missing, "derived_from_pkatt", "reported")
    m["Performance_PKcon"] = m["Performance_PKcon"].fillna(m["_pkatt"]).astype("Int64")
    ctx.note(6, label, "rebuilt Performance_PKcon", int(missing.sum()),
             f"seasons: {sorted(m.loc[missing, 'Season'].unique())}; reported values verified equal to PKatt")
    return m.drop(columns="_pkatt")


def step6_penalties(ctx: Ctx) -> None:
    f = ctx.frames
    pk = lambda n: f[n][["Squad", "Season", "Performance_PKatt"]]
    # squad_misc[X].PKcon  = opponent_stats[X].PKatt   (penalties attempted AGAINST X)
    f["squad_misc"] = rebuild_pkcon(f["squad_misc"], pk("opponent_stats"), "squad_misc", ctx)
    # opponent_misc[vs X].PKcon = squad_stats[X].PKatt (penalties X attempted)
    f["opponent_misc"] = rebuild_pkcon(f["opponent_misc"], pk("squad_stats"), "opponent_misc", ctx)
    # PKwon cannot be rebuilt and is under-counted in every season (league totals 74-94 vs 83-107 real penalties)
    for n in ("squad_misc", "opponent_misc"):
        ctx.drop(n, ["Performance_PKwon"], "unreliable (under-counted) and blank for 2025-26; use PKatt/PK")


# --------------------------------------------------------------------------- step 7: league tables
def step7_league_tables(ctx: Ctx) -> None:
    f = ctx.frames
    ro = f["rank_overall"]

    # --- Notes -> deduction + qualification
    notes = ro["Notes"]
    ro["points_deduction"] = pd.to_numeric(notes.str.extract(r"(\d+)-point deduction")[0]).fillna(0).astype("Int64")
    clean = notes.str.replace(r"\s+\d+$", "", regex=True).str.replace("Europa Conference League", "Conference League")
    c = clean.fillna("")
    ro["qualification"] = np.select(
        [c.str.contains("Champions League"), c.str.contains("Conference League"),
         c.str.contains("Europa League"), c.str.contains("Relegated")],
        ["Champions League", "Conference League", "Europa League", "Relegated"], default="")
    ro["qualification_route"] = np.select(
        [c.str.contains("via league finish"), c.str.contains("via cup win")], ["league finish", "cup win"], default="")
    for col in ("qualification", "qualification_route"):
        ro[col] = ro[col].replace("", np.nan)
    ro["Notes"] = clean
    ctx.note(7, "rank_overall", "parsed Notes -> points_deduction/qualification/route", int((ro["points_deduction"] > 0).sum()),
             "deductions: " + "; ".join(f"{r.Squad} {r.Season} -{r.points_deduction}" for r in ro[ro["points_deduction"] > 0].itertuples()))
    # Pts already includes the deduction -> also keep the pre-deduction points (= 3W + D)
    ro["pts_before_deduction"] = (ro["Pts"] + ro["points_deduction"]).astype("Int64")

    # --- Top Team Scorer 'Mohamed Salah - 23' -> name + goals
    ts = ro["Top Team Scorer"].str.extract(r"^(.*\S)\s+-\s+(\d+)$")
    if ts.isna().any().any():
        raise ValueError("Top Team Scorer: unexpected format")
    ro["top_scorer_name"], ro["top_scorer_goals"] = ts[0], ts[1].astype("Int64")
    # ties are listed in one cell ('A, B'); FBref truncates long tie lists with '...' -> flag them, do not guess names
    ro["top_scorer_joint"] = ro["top_scorer_name"].str.contains(r",|\.\.\.$").astype(int)
    trunc = int(ro["top_scorer_name"].str.endswith("...").sum())
    ro = ro.drop(columns="Top Team Scorer")
    ctx.dropped.setdefault("rank_overall", []).append(("Top Team Scorer", "split into top_scorer_name / top_scorer_goals"))
    ctx.note(7, "rank_overall", "split Top Team Scorer + top_scorer_joint flag", int(ro["top_scorer_joint"].sum()),
             f"tied top scorers share one cell; {trunc} lists are truncated by the source with '...'")

    # --- own goals: keep BOTH goal definitions and link them (decision: keep both)
    #   GF (league table) = player goals + own goals scored by the opponent in this team's favour
    sq = f["squad_stats"][["Squad", "Season", "Performance_Gls"]].rename(columns={"Performance_Gls": "_gls"})
    op = f["opponent_stats"][["Squad", "Season", "Performance_Gls"]].rename(columns={"Performance_Gls": "_opp_gls"})
    ro = ro.merge(sq, on=["Squad", "Season"], how="left", validate="1:1").merge(op, on=["Squad", "Season"], how="left", validate="1:1")
    ro["og_for"] = (ro["GF"] - ro["_gls"]).astype("Int64")        # own goals by opponents that count for this team
    ro["og_against"] = (ro["GA"] - ro["_opp_gls"]).astype("Int64")  # own goals by this team that count against it
    if (ro[["og_for", "og_against"]] < 0).any().any():
        raise ValueError("negative own-goal difference")
    ro = ro.drop(columns=["_gls", "_opp_gls"])
    ctx.note(7, "rank_overall", "added og_for / og_against", len(ro),
             "GF = Gls + og_for ; GA = opponent Gls + og_against (both goal definitions kept)")
    f["rank_overall"] = ro

    # --- home/away table: same deduction column, so points can be reconciled
    ha = f["rank_home_away"].merge(ro[["Squad", "Season", "points_deduction"]], on=["Squad", "Season"], how="left", validate="1:1")
    f["rank_home_away"] = ha
    ctx.note(7, "rank_home_away", "added points_deduction", int((ha["points_deduction"] > 0).sum()),
             "Home_Pts + Away_Pts = overall Pts + points_deduction")


# --------------------------------------------------------------------------- step 8: matches
def step8_matches(ctx: Ctx) -> None:
    m = ctx.frames["matches"]
    # division: every row must be E0 (Premier League); the two columns are then redundant
    if set(m["League"].dropna()) != {"E0"} or set(m["Div"].dropna()) != {"E0"} or m["Div"].isna().any():
        raise ValueError("Matches.csv is not purely E0 (Premier League)")
    m = m.drop(columns=["League", "Div"])
    ctx.dropped["matches"] = [("League", "constant 'E0'"), ("Div", "constant 'E0' (merged from two columns)")]
    ctx.note(8, "matches", "verified all rows are E0, dropped League/Div", len(m), "")

    # dates / referee / team names
    m["Date"] = pd.to_datetime(m["Date"], format="%d/%m/%Y", errors="raise").dt.strftime("%Y-%m-%d")
    m["Referee"] = m["Referee"].str.replace(r"\s+", " ", regex=True)
    for c in ("HomeTeam", "AwayTeam"):
        m[c] = m[c].replace(MATCHES_TEAM_MAP)
        unknown = set(m[c]) - ctx.canonical_teams
        if unknown:
            raise ValueError(f"Matches {c}: unmapped clubs {unknown}")
    ctx.note(8, "matches", "dates -> ISO; team names mapped to canonical", len(m),
             f"{len(MATCHES_TEAM_MAP)} clubs renamed: " + ", ".join(f"{k}->{v}" for k, v in MATCHES_TEAM_MAP.items()))

    cast_numeric(m)
    m = m.copy()  # defragment after ~165 column-wise assignments

    # impossible shots: shots on target can never exceed shots; which of the two is wrong is unknowable -> NULL both
    for side, team in (("H", "HomeTeam"), ("A", "AwayTeam")):
        bad = m[f"{side}ST"] > m[f"{side}S"]
        for r in m[bad].itertuples():
            ctx.note(8, "matches", f"{side} shots set to NULL (SoT > shots)", 1,
                     f"{r.Date} {r.HomeTeam} v {r.AwayTeam}: {side}S={getattr(r, side + 'S')}, {side}ST={getattr(r, side + 'ST')}")
        m.loc[bad, [f"{side}S", f"{side}ST"]] = pd.NA

    # odds: keep a column in the main file only if >= 95% filled in EVERY season; the rest go to a side file
    odds = [c for c in m.columns if c not in MATCH_CORE]
    fill = m[odds].notna().groupby(m["Season"]).mean().min()
    keep = [c for c in odds if fill[c] >= ODDS_MIN_FILL]
    sparse = [c for c in odds if c not in keep]
    ctx.note(8, "matches", "odds columns kept in main file", len(keep), f">= {ODDS_MIN_FILL:.0%} filled in every season")
    ctx.note(8, "matches", "odds columns moved to side file", len(sparse), "bookmakers present in only some seasons")

    # stable id, deterministic: season + order of kick-off
    m = m.sort_values(["Date", "Time", "HomeTeam"], kind="stable").reset_index(drop=True)
    m.insert(0, "match_id", season_start(m["Season"]).astype(str) + "-" + (m.groupby("Season").cumcount() + 1).astype(str).str.zfill(3))
    extra = m[["match_id"] + sparse].copy()
    ctx.frames["matches"] = m[["match_id"] + MATCH_CORE + keep]
    ctx.frames["matches_odds_extra"] = extra
    ctx.dropped["matches"] += [(c, f"moved to {MATCHES_ODDS_OUT} (sparse odds)") for c in sparse]
    ctx.orig_cols["matches_odds_extra"] = ["match_id"] + sparse


# --------------------------------------------------------------------------- step 9: column names
def step9_rename(ctx: Ctx) -> list[dict]:
    rows = []
    for name, df in ctx.frames.items():
        new_cols = [MATCHES_RENAME.get(c, snake(c)) if name == "matches" else snake(c) for c in df.columns]
        if len(set(new_cols)) != len(new_cols):
            raise ValueError(f"{name}: snake_case produced duplicate column names")
        mapping = dict(zip(df.columns, new_cols))
        orig = ctx.orig_cols.get(name, [])
        dropped = dict(ctx.dropped.get(name, []))
        for raw in orig:
            raw_clean = clean_header(raw)
            if raw_clean in mapping:
                rows.append({"dataset": name, "raw_column": raw, "clean_column": mapping[raw_clean], "status": "kept"})
            elif raw_clean in dropped:
                rows.append({"dataset": name, "raw_column": raw, "clean_column": "", "status": "dropped/moved: " + dropped[raw_clean]})
            elif name == "matches" and raw != raw_clean:
                continue  # the polluted duplicate header, merged into 'Div' (dropped)
            else:
                raise ValueError(f"{name}: raw column {raw!r} unaccounted for")
        for c in df.columns:
            if c not in [clean_header(r) for r in orig]:
                rows.append({"dataset": name, "raw_column": "", "clean_column": mapping[c], "status": "derived"})
        ctx.frames[name] = df.rename(columns=mapping)
    return rows


# --------------------------------------------------------------------------- step 10: write + report
NULL_REASON_RULES = {
    "players_stats": {"nation": "blank in source (3 players)", "age": "blank in source", "born": "blank in source",
                      "pos_secondary": "single-position player"},
    "players_shooting": {"nation": "blank in source", "age": "blank in source", "born": "blank in source",
                         "pos_secondary": "single-position player",
                         "standard_sot_pct": "player took no shots", "standard_g_per_sh": "player took no shots",
                         "standard_g_per_sot": "player had no shots on target"},
    "keepers_stats": {"performance_save_pct": "no shots on target faced", "performance_cs_pct": "played < 60 minutes",
                      "penalty_kicks_save_pct": "no penalty on target faced", "pos_secondary": "single-position player"},
    "squad_goalkeeping": {"penalty_kicks_save_pct": "no penalty on target faced"},
    "opponent_goalkeeping": {"penalty_kicks_save_pct": "no penalty on target faced"},
    "rank_overall": {"notes": "no European place / relegation / deduction", "qualification": "no European place or relegation",
                     "qualification_route": "route not stated"},
    "matches": {"away_shots": "SoT > shots, nulled", "away_shots_on_target": "SoT > shots, nulled",
                "home_shots": "SoT > shots, nulled", "home_shots_on_target": "SoT > shots, nulled"},
}


def null_report(ctx: Ctx) -> pd.DataFrame:
    rows = []
    core_clean = {MATCHES_RENAME.get(c, snake(c)) for c in MATCH_CORE} | {"match_id"}
    for name, df in ctx.frames.items():
        if name == "matches_odds_extra":
            continue
        for c in df.columns:
            n = int(df[c].isna().sum())
            if not n:
                continue
            reason = NULL_REASON_RULES.get(name, {}).get(c)
            if reason is None and name == "matches" and c not in core_clean:
                reason = "bookmaker did not quote this match (<=5% of rows)"
            rows.append({"dataset": name, "column": c, "null_count": n, "reason": reason or "UNEXPECTED"})
    return pd.DataFrame(rows)


def out_name(name: str) -> str:
    if name == "matches":
        return MATCHES_OUT
    if name == "matches_odds_extra":
        return MATCHES_ODDS_OUT
    return RAW_FILES[name].replace("oppenent", "opponent")


def step10_write(ctx: Ctx, out: Path, colmap: list[dict]) -> None:
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    for name, df in ctx.frames.items():
        df.to_csv(out / "clean" / out_name(name), index=False, encoding="utf-8")
        ctx.note(10, name, "written", len(df), f"{out_name(name)}  ({df.shape[1]} columns)")
    pd.DataFrame(colmap).to_csv(out / "reports" / "column_map.csv", index=False)
    nr = null_report(ctx)
    nr.to_csv(out / "reports" / "remaining_nulls.csv", index=False)
    pd.DataFrame(ctx.log).to_csv(out / "reports" / "cleaning_log.csv", index=False)
    bad = nr[nr["reason"] == "UNEXPECTED"]
    if len(bad):
        print("WARNING - unexplained NULLs:\n", bad.to_string(index=False))
        sys.exit(2)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path, help="folder holding the 15 EPL CSVs")
    ap.add_argument("--matches", required=True, type=Path, help="the Premier League (E0) Matches.csv")
    ap.add_argument("--out", required=True, type=Path, help="output folder (clean/ and reports/ are created)")
    a = ap.parse_args()

    ctx = Ctx()
    steps = [("1  load", lambda: step1_load(ctx, a.raw_dir, a.matches)),
             ("2  structure", lambda: step2_structure(ctx)),
             ("3  team names", lambda: step3_team_names(ctx)),
             ("4  types", lambda: step4_types(ctx)),
             ("5  players & keepers", lambda: step5_people(ctx)),
             ("6  penalties", lambda: step6_penalties(ctx)),
             ("7  league tables", lambda: step7_league_tables(ctx)),
             ("8  matches", lambda: step8_matches(ctx))]
    for label, fn in steps:
        fn()
        print(f"step {label:<22} ok")
    colmap = step9_rename(ctx)
    print(f"step {'9  column names':<22} ok")
    step10_write(ctx, a.out, colmap)
    print(f"step {'10 write + report':<22} ok  -> {a.out}")


if __name__ == "__main__":
    main()
