"""
skim_probe_tree.py
==================
Reads the tag-and-probe J/psi ROOT file, applies the same event and probe
selections as efficiency_nn_v2.py, and writes a new flat ROOT tree with one
entry per selected probe muon:

    tree/pt       – probe pT  [GeV]
    tree/eta      – probe eta  (signed, |eta| < 1.5 by construction)
    tree/dxy_sig  – probe |dxy / sigma_xy|
    tree/pass     – 1 if probe fired the HLT, 0 otherwise  (int32)
"""

import numpy as np
import uproot

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# INPUT_FILE  = "hb_2018_24may2026.root"
# # OUTPUT_FILE = "probe_skim_hb_2018_24may2026.root"
# # OUTPUT_FILE = "probe_skim_hb_2018_27aug2026.root"
# OUTPUT_FILE = "probe_skim_hb_2018_31aug2026_v2.root"

INPUT_FILE  = "charmonium_2018_24may2026.root"
# OUTPUT_FILE = "probe_skim_charmonium_2018_24may2026.root"
# OUTPUT_FILE = "probe_skim_charmonium_2018_27aug2026.root"
OUTPUT_FILE = "probe_skim_charmonium_2018_31aug2026.root"


# ──────────────────────────────────────────────────────────────────────────────
# READ
# ──────────────────────────────────────────────────────────────────────────────

branches = [
    "run", "lumi", "event",
    "cos2d", "lxy",
    "mu1_pt", "mu2_pt",
    "mu1_bs_dxy_sig", "mu2_bs_dxy_sig",
    "mu1_bs_dxy_e", "mu2_bs_dxy_e",
    "mu1_bs_dxy", "mu2_bs_dxy",
    "mu1_eta", "mu2_eta",
    "mass", "vtx_prob", "charge", "dr_12",
    "mu1_id_medium", "mu2_id_medium",

    "mu1_HLT_Mu7p5_Track3p5_Jpsi_probe",
    "mu2_HLT_Mu7p5_Track3p5_Jpsi_probe",
    "mu1_HLT_Mu7p5_Track3p5_Jpsi_tag",
    "mu2_HLT_Mu7p5_Track3p5_Jpsi_tag",

    # prescale OR flags (any > 0 → event is on a relevant path)
    "HLT_Mu7_IP4_ps",   "HLT_Mu8_IP3_ps",   "HLT_Mu8_IP5_ps",
    "HLT_Mu8_IP6_ps",   "HLT_Mu8p5_IP3p5_ps","HLT_Mu9_IP4_ps",
    "HLT_Mu9_IP5_ps",   "HLT_Mu9_IP6_ps",   "HLT_Mu10p5_IP3p5_ps",
    "HLT_Mu12_IP6_ps",

    # mu1 HLT tag flags
    "mu1_HLT_Mu7_IP4_tag",    "mu1_HLT_Mu8_IP3_tag",
    "mu1_HLT_Mu8_IP5_tag",    "mu1_HLT_Mu8_IP6_tag",
    "mu1_HLT_Mu8p5_IP3p5_tag","mu1_HLT_Mu9_IP4_tag",
    "mu1_HLT_Mu9_IP5_tag",    "mu1_HLT_Mu9_IP6_tag",
    "mu1_HLT_Mu10p5_IP3p5_tag","mu1_HLT_Mu12_IP6_tag",

    # mu2 HLT tag flags
    "mu2_HLT_Mu7_IP4_tag",    "mu2_HLT_Mu8_IP3_tag",
    "mu2_HLT_Mu8_IP5_tag",    "mu2_HLT_Mu8_IP6_tag",
    "mu2_HLT_Mu8p5_IP3p5_tag","mu2_HLT_Mu9_IP4_tag",
    "mu2_HLT_Mu9_IP5_tag",    "mu2_HLT_Mu9_IP6_tag",
    "mu2_HLT_Mu10p5_IP3p5_tag","mu2_HLT_Mu12_IP6_tag",
]

print(f"Opening {INPUT_FILE} …")
f    = uproot.open(INPUT_FILE)
tree = f["tree"]

print("Reading branches …")
arrays = tree.arrays(branches, library="np")

def v(name):
    return arrays[name]

# ──────────────────────────────────────────────────────────────────────────────
# SELECTIONS
# ──────────────────────────────────────────────────────────────────────────────

# Any prescaled path fired
ps_or = (
    (v("HLT_Mu7_IP4_ps")      > 0) | (v("HLT_Mu8_IP3_ps")      > 0) |
    (v("HLT_Mu8_IP5_ps")      > 0) | (v("HLT_Mu8_IP6_ps")      > 0) |
    (v("HLT_Mu8p5_IP3p5_ps")  > 0) | (v("HLT_Mu9_IP4_ps")      > 0) |
    (v("HLT_Mu9_IP5_ps")      > 0) | (v("HLT_Mu9_IP6_ps")      > 0) |
    (v("HLT_Mu10p5_IP3p5_ps") > 0) | (v("HLT_Mu12_IP6_ps")     > 0)
)

# "Fired HLT" flag for mu1 and mu2 (used both as label and as tag)
mu1_fired = (
    (v("mu1_HLT_Mu7_IP4_tag")        > 0.5) | (v("mu1_HLT_Mu8_IP3_tag")        > 0.5) |
    (v("mu1_HLT_Mu8_IP5_tag")        > 0.5) | (v("mu1_HLT_Mu8_IP6_tag")        > 0.5) |
    (v("mu1_HLT_Mu8p5_IP3p5_tag")    > 0.5) | (v("mu1_HLT_Mu9_IP4_tag")        > 0.5) |
    (v("mu1_HLT_Mu9_IP5_tag")        > 0.5) | (v("mu1_HLT_Mu9_IP6_tag")        > 0.5) |
    (v("mu1_HLT_Mu10p5_IP3p5_tag")   > 0.5) | (v("mu1_HLT_Mu12_IP6_tag")       > 0.5)
)

mu2_fired = (
    (v("mu2_HLT_Mu7_IP4_tag")        > 0.5) | (v("mu2_HLT_Mu8_IP3_tag")        > 0.5) |
    (v("mu2_HLT_Mu8_IP5_tag")        > 0.5) | (v("mu2_HLT_Mu8_IP6_tag")        > 0.5) |
    (v("mu2_HLT_Mu8p5_IP3p5_tag")    > 0.5) | (v("mu2_HLT_Mu9_IP4_tag")        > 0.5) |
    (v("mu2_HLT_Mu9_IP5_tag")        > 0.5) | (v("mu2_HLT_Mu9_IP6_tag")        > 0.5) |
    (v("mu2_HLT_Mu10p5_IP3p5_tag")   > 0.5) | (v("mu2_HLT_Mu12_IP6_tag")       > 0.5)
)

# Di-muon quality cuts (same as the NN training)
quality = (
#     (np.abs(v("mass") - 3.0969) < 0.1) &
    (v("vtx_prob") > 0.01) &
    (v("mu2_pt") > 6) &
    (v("charge") == 0) &
    (v("dr_12")  > 0.12) &
    (v("cos2d")  > 0.9) &
#     (v("run")  >= 320673) &
    (v("lxy") * v("cos2d") * 3.0969 >= 0.008) & # 80 micron pseudo proper decay length
    ps_or
)

# mu1 as probe: mu2 is the tag, mu1 is the probe
mask1 = (
    quality &
    (v("mu2_HLT_Mu7p5_Track3p5_Jpsi_tag")   > 0.5) &   # tag requirement
    (v("mu1_HLT_Mu7p5_Track3p5_Jpsi_probe") > 0.5) &   # probe requirement
    (np.abs(v("mu1_eta")) < 1.5) &
    v("mu1_id_medium").astype(bool) &
    v("mu2_id_medium").astype(bool)
)

# mu2 as probe: mu1 is the tag, mu2 is the probe
mask2 = (
    quality &
    (v("mu1_HLT_Mu7p5_Track3p5_Jpsi_tag")   > 0.5) &   # tag requirement
    (v("mu2_HLT_Mu7p5_Track3p5_Jpsi_probe") > 0.5) &   # probe requirement
    (np.abs(v("mu2_eta")) < 1.5) &
    v("mu1_id_medium").astype(bool) &
    v("mu2_id_medium").astype(bool)
)

# ──────────────────────────────────────────────────────────────────────────────
# BUILD OUTPUT ARRAYS
# ──────────────────────────────────────────────────────────────────────────────

run_out      = np.concatenate([
    v("run")[mask1],
    v("run")[mask2],
]).astype(np.float32)

lumi_out      = np.concatenate([
    v("lumi")[mask1],
    v("lumi")[mask2],
]).astype(np.float32)

event_out      = np.concatenate([
    v("event")[mask1],
    v("event")[mask2],
]).astype(np.float32)

mass_out      = np.concatenate([
    v("mass")[mask1],
    v("mass")[mask2],
]).astype(np.float32)

pt_out      = np.concatenate([
    v("mu1_pt")[mask1],
    v("mu2_pt")[mask2],
]).astype(np.float32)

eta_out     = np.concatenate([
    v("mu1_eta")[mask1],               # signed eta, |eta| < 1.5 by construction
    v("mu2_eta")[mask2],
]).astype(np.float32)

dxy_out = np.concatenate([
    np.abs(v("mu1_bs_dxy")[mask1]),
    np.abs(v("mu2_bs_dxy")[mask2]),
]).astype(np.float32)

dxy_e_out = np.concatenate([
    np.abs(v("mu1_bs_dxy_e")[mask1]),
    np.abs(v("mu2_bs_dxy_e")[mask2]),
]).astype(np.float32)

dxy_sig_out = np.concatenate([
    np.abs(v("mu1_bs_dxy_sig")[mask1]),
    np.abs(v("mu2_bs_dxy_sig")[mask2]),
]).astype(np.float32)

# pass = 1 if the probe muon itself fired the HLT
pass_out    = np.concatenate([
    mu1_fired[mask1].astype(np.int32),
    mu2_fired[mask2].astype(np.int32),
])

n_total = len(pt_out)
n_pass  = int(pass_out.sum())
print(f"\nSelected probe candidates : {n_total:,}")
print(f"  pass (HLT fired)         : {n_pass:,}  ({100*n_pass/n_total:.1f}%)")
print(f"  fail                     : {n_total - n_pass:,}  ({100*(1-n_pass/n_total):.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUT ROOT FILE
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nWriting {OUTPUT_FILE} …")

with uproot.recreate(OUTPUT_FILE) as out:
    out["tree"] = {
        "run"    : run_out    ,
        "lumi"   : lumi_out   ,
        "event"  : event_out  ,
        "mass"   : mass_out   ,
        "pt"     : pt_out     ,
        "eta"    : eta_out    ,
        "dxy"    : dxy_out    ,
        "dxy_e"  : dxy_e_out  ,
        "dxy_sig": dxy_sig_out,
        "pass"   : pass_out   ,
    }

print(f"Done. Output written to {OUTPUT_FILE}")
print(f"  tree branches: pt (float32), eta (float32), dxy_sig (float32), pass (int32)")

