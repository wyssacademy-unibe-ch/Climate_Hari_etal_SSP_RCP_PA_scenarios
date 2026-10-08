#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helper for paired RCP2.6 versus RCP6.0 climate uncertainty.

Import and call `print_paired_climate_mitigation(member_df, SDM_CONFIGS, GCMS)`
after `member_df` has been created in the main contribution script.
"""

import numpy as np
import pandas as pd


def print_paired_climate_mitigation(member_df, sdm_configs, gcms):
    """Print RCP2.6 minus RCP6.0 climate contribution, mean +/- paired SD."""

    required_columns = {
        "Scenario",
        "SDM_configuration",
        "GCM",
        "Member",
        "PA_configuration",
        "Climate",
    }
    missing = required_columns.difference(member_df.columns)
    if missing:
        raise KeyError(
            "member_df is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Climate values are repeated for every PA configuration. Confirm that
    # the repeated values agree, then retain one value per pathway and member.
    consistency = (
        member_df
        .groupby(
            ["Scenario", "SDM_configuration", "GCM", "Member"],
            as_index=False,
        )
        .agg(
            Climate_min=("Climate", "min"),
            Climate_max=("Climate", "max"),
        )
    )

    inconsistent = consistency.loc[
        ~np.isclose(
            consistency["Climate_min"],
            consistency["Climate_max"],
            rtol=0.0,
            atol=1e-12,
        )
    ]
    if not inconsistent.empty:
        raise ValueError(
            "Climate contributions differ among PA rows for matched members:\n"
            + inconsistent.to_string(index=False)
        )

    climate_members = (
        member_df[
            ["Scenario", "SDM_configuration", "GCM", "Member", "Climate"]
        ]
        .drop_duplicates(
            subset=["Scenario", "SDM_configuration", "GCM", "Member"]
        )
        .copy()
    )

    paired = (
        climate_members
        .pivot(
            index=["SDM_configuration", "GCM", "Member"],
            columns="Scenario",
            values="Climate",
        )
        .reset_index()
    )

    for scenario in ["rcp26", "rcp60"]:
        if scenario not in paired.columns:
            raise KeyError(f"No member-level values were found for {scenario}.")

    paired = paired.dropna(subset=["rcp26", "rcp60"]).copy()
    if paired.empty:
        raise RuntimeError("No complete RCP2.6-RCP6.0 member pairs were found.")

    # Positive values are projected richness loss avoided under RCP2.6.
    paired["Climate_mitigation"] = paired["rcp26"] - paired["rcp60"]

    rcp26_mean = float(paired["rcp26"].mean())
    rcp26_sd = float(paired["rcp26"].std(ddof=1))
    rcp60_mean = float(paired["rcp60"].mean())
    rcp60_sd = float(paired["rcp60"].std(ddof=1))
    mitigation_mean = float(paired["Climate_mitigation"].mean())
    mitigation_sd = float(paired["Climate_mitigation"].std(ddof=1))

    n_pairs = int(paired["Climate_mitigation"].count())
    expected_pairs = len(sdm_configs) * len(gcms)

    print("\n" + "=" * 100)
    print("PAIRED CLIMATE-MITIGATION CONTRIBUTION")
    print("RCP2.6 climate contribution minus RCP6.0 climate contribution")
    print("=" * 100)

    for _, row in paired.iterrows():
        print(
            f"{row['Member']}: "
            f"RCP2.6 = {row['rcp26']:+.4f}%, "
            f"RCP6.0 = {row['rcp60']:+.4f}%, "
            f"avoided loss = {row['Climate_mitigation']:+.4f} pp"
        )

    print("\nSummary across matched GCM-SDM members")
    print(f"  RCP2.6 climate contribution: {rcp26_mean:+.2f}% ± {rcp26_sd:.2f}%")
    print(f"  RCP6.0 climate contribution: {rcp60_mean:+.2f}% ± {rcp60_sd:.2f}%")
    print(
        "  Climate-mitigation contribution: "
        f"{mitigation_mean:+.2f} ± {mitigation_sd:.2f} percentage points"
    )
    print(f"  Paired members: {n_pairs} of {expected_pairs}")

    if n_pairs != expected_pairs:
        print("  WARNING: the paired summary uses an incomplete member set.")

    return paired
