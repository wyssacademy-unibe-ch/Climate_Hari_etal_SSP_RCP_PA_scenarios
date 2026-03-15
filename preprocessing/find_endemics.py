#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np

# --------------------------
# Helpers
# --------------------------


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names:
      - remove BOM, strip spaces, lowercase
      - collapse internal whitespace -> single space
      - replace spaces with underscores
    """
    new_cols = []
    for c in df.columns:
        cc = str(c)
        # strip potential BOM
        cc = cc.encode("utf-8").decode("utf-8-sig")
        cc = cc.strip().lower()
        cc = re.sub(r"\s+", " ", cc)  # collapse many spaces
        cc = cc.replace(" ", "_")
        new_cols.append(cc)
    df.columns = new_cols
    return df


def canon_key(name: str) -> str:
    """Case/format‑robust key for joining names."""
    if pd.isna(name):
        return ""
    s = str(name).lower().strip().replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_modeled_name_from_file(fname: str) -> str:
    """
    Extract species name from filenames like:
      Abavorana_luctuosa_GAM_dispersal.csv.xz -> 'Abavorana luctuosa'
    """
    base = os.path.basename(fname)
    base = re.sub(r"\.(csv(\.(xz|gz))?|xz|gz)$", "", base, flags=re.I)
    m = re.match(r"(.+?)_GAM", base, flags=re.I)
    species_underscore = m.group(1) if m else base.split("_GAM")[0]
    raw = re.sub(r"_+", " ", species_underscore).strip()
    return raw

# --------------------------
# Load reference tables
# --------------------------


def load_endemics(endemic_csv: str, synonyms_csv: str):
    """
    endemic_ssp.csv → must contain (case-insensitive): className, sci_name, ID
    synonyms_endemics.csv → must contain: ID, sci_name (or sci_nam), synonym
    Returns: (df_end, df_synK)
    """
    # Endemics
    df_end = pd.read_csv(endemic_csv)
    df_end = normalize_cols(df_end)

    # --- FIX 1 START: Handle 'classNam' typo in endemic_ssp.csv ---
    # After normalize_cols, 'classNam' becomes 'classnam'. We rename it to 'classname'.
    if "classname" not in df_end.columns and "classnam" in df_end.columns:
        print(
            "[NOTE] Renaming 'classnam' to 'classname' to fix header typo in endemic_ssp.csv.")
        df_end.rename(columns={"classnam": "classname"}, inplace=True)
    # --- FIX 1 END ---

    required_end = {"classname", "sci_name", "id"}
    missing_end = required_end - set(df_end.columns)
    if missing_end:
        raise ValueError(f"[endemic_ssp.csv] missing columns after normalization: {missing_end}\n"
                         f"Columns seen: {df_end.columns.tolist()}")

    # Keep only target classes
    df_end = df_end[df_end["classname"].str.upper().isin(
        {"AMPHIBIA", "MAMMALIA", "AVES"})].copy()

    # Canonical key
    df_end["sci_name"] = df_end["sci_name"].astype(str).str.strip()
    df_end["key_canon"] = df_end["sci_name"].map(canon_key)

    # Store df_end columns needed for the merge later
    end_cols = df_end[["id", "classname", "sci_name", "key_canon"]].copy()

    # Synonyms
    df_syn = pd.read_csv(synonyms_csv)
    df_syn = normalize_cols(df_syn)
    # Accept sci_name or sci_nam (typo-safe)
    if "sci_name" not in df_syn.columns and "sci_nam" in df_syn.columns:
        df_syn.rename(columns={"sci_nam": "sci_name"}, inplace=True)

    required_syn = {"id", "sci_name", "synonym"}
    missing_syn = required_syn - set(df_syn.columns)
    if missing_syn:
        raise ValueError(f"[synonyms_endemics.csv] missing columns after normalization: {missing_syn}\n"
                         f"Columns seen: {df_syn.columns.tolist()}")

    df_syn["synonym"] = df_syn["synonym"].astype(str).str.strip()
    df_syn["key_syn"] = df_syn["synonym"].map(canon_key)

    # Store df_syn columns needed for the merge later
    syn_cols = df_syn[["id", "sci_name", "key_syn"]].copy()

    # --- FIX 2 START: Explicit column management for df_synK merge ---
    # Join synonyms (syn_cols) → endemics (end_cols) to inherit class and canonical name
    # We use suffixes to manage the 'sci_name' conflict.
    df_synK = syn_cols.merge(
        end_cols,
        on="id",
        how="left",
        suffixes=("_syn", "_end")
    )

    # Rename the canonical scientific name (from endemic source) back to 'sci_name'
    df_synK.rename(columns={"sci_name_end": "sci_name"}, inplace=True)

    # Drop the redundant synonym name and key_canon, as they are not needed in the match function
    # 'classname' and 'id' survived correctly. 'key_syn' survived correctly.
    df_synK.drop(columns=["sci_name_syn", "key_canon"], inplace=True)

    # Final check: df_synK should contain: ['id', 'key_syn', 'classname', 'sci_name']

    # Only keep the three classes (this step is redundant since end_cols was filtered, but kept for safety)
    df_synK = df_synK[df_synK["classname"].str.upper().isin(
        {"AMPHIBIA", "MAMMALIA", "AVES"})].copy()
    # --- FIX 2 END ---

    return df_end, df_synK

# --------------------------
# Scan modeled files
# --------------------------


def scan_modeled_species(taxon_label: str, folder: str) -> pd.DataFrame:
    tax2class = {"Amphibians": "AMPHIBIA",
                 "Mammals": "MAMMALIA", "Bird": "AVES"}
    taxon_class = tax2class.get(taxon_label, taxon_label.upper())

    if folder is None or not os.path.isdir(folder):
        print(f"[WARN] Folder not found for {taxon_label}: {folder}")
        return pd.DataFrame(columns=["taxon", "taxon_class", "modeled_name", "key", "file_path"])

    patterns = ["*.csv", "*.csv.xz", "*.csv.gz"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))

    rows = []
    for f in sorted(set(files)):
        modeled = normalize_modeled_name_from_file(f)
        rows.append(dict(
            taxon=taxon_label,
            taxon_class=taxon_class,
            modeled_name=modeled,
            key=canon_key(modeled),
            file_path=f
        ))
    return pd.DataFrame(rows)

# --------------------------
# Match
# --------------------------
# --------------------------
# Match (Corrected Section)
# --------------------------


def match_modeled_to_endemics(df_mod, df_end, df_synK):
    # 1) canonical (No change here)
    m1 = df_mod.merge(
        df_end[["id", "classname", "sci_name", "key_canon"]],
        left_on="key", right_on="key_canon", how="left"
    )
    m1["match_source"] = np.where(m1["id"].notna(), "canonical", pd.NA)

    # 2) synonyms for the ones still unmatched
    um = m1[m1["id"].isna()].copy()
    if not um.empty:
        syn_join = um.merge(
            df_synK[["id", "classname", "sci_name", "key_syn"]],
            left_on="key", right_on="key_syn", how="left",
            # --- FIX: Use suffixes to rename the required columns from df_synK (right table) ---
            # The right table's columns (df_synK) will retain the original names
            suffixes=("_old", "")
            # The left table's columns (um, which are NaN) get the '_old' suffix
        )

        # After the merge with suffixes:
        # The new, valid ID is now 'id' (from df_synK).
        # The old, NaN ID is now 'id_old' (from um).

        # We need to drop the redundant columns from the left table (um)
        # that were merged in (they only contained the NaN values).
        syn_join.drop(columns=["id_old", "classname_old",
                      "sci_name_old"], inplace=True, errors='ignore')

        # Now, the column 'id' exists again, and the match logic works:
        syn_join.loc[syn_join["id"].notna(), "match_source"] = "synonym"
        m1 = pd.concat([m1[m1["id"].notna()], syn_join], ignore_index=True)

    # Keep only the right class
    m_ok = m1[m1["id"].notna()].copy()
    m_ok = m_ok[m_ok["classname"].str.upper(
    ) == m_ok["taxon_class"].str.upper()]

    matched = m_ok[[
        "taxon", "taxon_class", "modeled_name", "sci_name", "id", "classname", "match_source", "file_path"
    ]].rename(columns={
        "sci_name": "endemic_canonical_name",
        "classname": "endemic_class",
        "id": "endemic_id"
    }).sort_values(["taxon", "modeled_name"])

    # unmatched list (for curation)
    unmatched = df_mod.merge(
        matched[["file_path"]], on="file_path", how="left", indicator=True)
    unmatched = unmatched[unmatched["_merge"] ==
                          "left_only"].drop(columns=["_merge"]).copy()
    unmatched = unmatched[["taxon", "taxon_class", "modeled_name",
                           "file_path"]].sort_values(["taxon", "modeled_name"])

    return matched, unmatched

# --------------------------
# CLI
# --------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Check which modeled species are endemics (canonical + synonyms).")
    ap.add_argument("--amph", required=True,
                    help="Folder with Amphibians projections (CSV/.xz/.gz)")
    ap.add_argument("--mamm", required=True,
                    help="Folder with Mammals projections (CSV/.xz/.gz)")
    ap.add_argument("--bird", required=True,
                    help="Folder with Bird projections (CSV/.xz/.gz)")
    ap.add_argument("--endemic", required=True, help="endemic_ssp.csv")
    ap.add_argument("--synonyms", required=True, help="synonyms_endemics.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    print(">> Loading reference tables ...")
    df_end, df_synK = load_endemics(args.endemic, args.synonyms)
    print(f"   Endemic rows (AMPHIBIA/MAMMALIA/AVES): {len(df_end):,}")
    print(f"   Synonyms rows (joined): {len(df_synK):,}")

    print(">> Scanning modeled species ...")
    df_amph = scan_modeled_species("Amphibians", args.amph)
    df_mamm = scan_modeled_species("Mammals",    args.mamm)
    df_bird = scan_modeled_species("Bird",       args.bird)
    df_all = pd.concat([df_amph, df_mamm, df_bird], ignore_index=True)
    print(
        f"   Modeled files: {len(df_all):,} | Unique modeled names: {df_all['modeled_name'].nunique():,}")

    print(">> Matching modeled species to endemics (canonical → synonyms) ...")
    matched, unmatched = match_modeled_to_endemics(df_all, df_end, df_synK)

    out_matches = os.path.join(args.outdir, "modeled_endemics_matches.csv")
    out_unmatched = os.path.join(args.outdir, "modeled_species_unmatched.csv")
    matched.to_csv(out_matches, index=False)
    unmatched.to_csv(out_unmatched, index=False)
    print(f"[OK] wrote: {out_matches}   (rows={len(matched):,})")
    print(f"[OK] wrote: {out_unmatched}   (rows={len(unmatched):,})")

    # Summary

    # Summary per taxon
    if not matched.empty:
        summ = matched.groupby("taxon").agg(
            modeled_endemics=("modeled_name", "nunique"),
            files=("file_path", "count")
        ).reset_index()
        print("\n[SUMMARY: modeled endemics per taxon]")
        for _, r in summ.iterrows():
            print(
                f"   {r['taxon']}: {int(r['modeled_endemics'])} species (from {int(r['files'])} files)")
    else:
        print("\n[SUMMARY] No modeled species matched as endemics.")

    if not unmatched.empty:
        miss = unmatched.groupby("taxon")["modeled_name"].nunique(
        ).reset_index(name="unmatched_species")
        print("\n[NOTE] Unmatched modeled species (no endemic match):")
        for _, r in miss.iterrows():
            print(f"   {r['taxon']}: {int(r['unmatched_species'])} species")
    else:
        print("\n[NOTE] All modeled species were matched as endemics.")


if __name__ == "__main__":
    main()
