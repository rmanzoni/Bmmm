# Fit-scheme comparison on the signal MC

The ntuplizer fits each candidate with up to four sets of mass constraints and
computes every Bs-dependent observable with several Bs momentum hypotheses.
This suite compares them on the signal MC, to decide which scheme and which
hypothesis to use for q², m²miss, E*μ and the other observables.

| scheme | constraints in the vertex fits |
|---|---|
| `cons` | m(KK) = m(φ) and m(KKπ) = m(Ds) |
| `phic` | m(KK) = m(φ) only (the AN choice) |
| `dsc`  | m(KKπ) = m(Ds) only (ntuples produced after commit 3755ffb) |
| `free` | none |

| hypothesis | Bs momentum |
|---|---|
| `coll` | p(Dsμ) × M(Bs)/m(Dsμ) |
| `xyz_bs` | \|p(Dsμ)\| × M(Bs)/m(Dsμ), along PV → V_Bs |
| `math1_bs`, `math2_bs` | the two neutrino solutions along PV → V_Bs (math1 is the lower \|p(Bs)\|, almost always) |
| `math_avg` | built here: the average of the two solutions' observables (χ: circular average) |
| `math_oracle` | built here: the solution closer to the true \|p(Bs)\|. **Uses the truth**: a reference for how much a perfect choice would gain, never ranked |
| `gamma_*` | the same with Ds + soft photon in place of the Ds |

Schemes and hypotheses are read from the branch names, so older ntuples with
three schemes work unchanged.

## Running

```bash
# at PSI, any python3 with numpy, scipy, matplotlib, uproot (scikit-learn for the BDT)
python3 compare_fits.py /pnfs/psi.ch/cms/trivcat/store/user/manzoni/rds_prod.root -o cmp_v1
```

Then open `cmp_v1/index.html`. The run prints a timestamped log and writes
`log.txt` and `status.json` (current stage, elapsed time) as it goes, and
`results.json` after every stage, so a long run can be followed and a crash
leaves the finished stages on disk. The selected arrays are cached in
`cmp_v1/cache/`; a second run with other options on the same input skips the
reading.

Reading. The file is read in blocks of 100k entries taken in random order,
by 4 processes in parallel, and the reading stops once 10⁶ truth-matched
signal candidates are kept (about 250k per channel, far more than any metric
here needs). Random blocks make the part read an unbiased sample of the whole
file; the report states the fraction read. Floats are kept as float32
(covariances as float64). Each progress line gives the memory in use.

Useful options:

| option | effect |
|---|---|
| `--max-candidates N` | stop reading at N signal candidates (default 10⁶; 0 = whole file) |
| `-j N` | reader processes (default 4) |
| `--max-entries N` | consider only the first N rows of the file |
| `--no-bdt` | skip the BDTs, the slowest stage |
| `--bdt-sets coll,all_hyps` | fewer BDTs |
| `--bdt-max N` | BDT candidates per channel (default 50000) |
| `--r-target 0.299,0.244` | R(Ds), R(Ds*) the τ channels are weighted to (approximate SM values) |
| `--r-mc 1,1` | R(Ds), R(Ds*) the MC was generated with |
| `--no-trigger`, `--no-species`, `--all-candidates`, `--no-common` | loosen the selection (below) |
| `--pdf` | PDF of every plot too |

Without the ntuple at hand, `make_toy_ntuple.py -o toy.root -n 4000` writes a
toy with the same branch names (phase space, crude vertex smearing; the
schemes differ only in a vertex-resolution scale factor, cons < dsc < phic <
free). It only checks that the code runs; its physics numbers mean nothing.

The BDT stage needs scikit-learn built against the same numpy as the one
imported. In a CMSSW environment with numpy 2 installed by `pip install
--user`, the CMSSW scikit-learn (built against numpy 1) does not import; the
suite checks this at the start and skips the BDTs with a warning. To have
them, run with the CMSSW numpy (`PYTHONNOUSERSITE=1 python3 compare_fits.py ...`,
if uproot is available there) or in a separate environment
(`python3 -m venv cf; . cf/bin/activate; pip install numpy scipy matplotlib uproot scikit-learn`).

Tests: `python3 -m unittest test_compare_fits -v`.

## Selection

1. Truth-matched signal: `gen_cand` 0 (Ds μ ν), 1 (Ds* μ ν), 2 (Ds τ ν), 3 (Ds* τ ν).
2. Correct species: the two kaons matched to K±, the pion to π±, the muon to μ±.
3. Trigger: one of the parking paths fired and the muon is matched to its trigger object.
4. One candidate per event: the first one (lowest `cand_idx`).
5. Common sample: every scheme has a usable Bs vertex. All schemes are then
   compared on the same events. Efficiencies are computed before this step.

The cutflow per channel is in the report.

## Metrics

**τ/μ separation (the main criterion).**

- *σ τ/μ*: expected relative precision on N(τ)/N(μ) from a binned fit of two
  yields (τ and μ; Ds and Ds* merged at their MC proportions). This measures
  the τ/μ separation alone. It is used to sort the tables.
- *σR(Ds), σR(Ds\*)*: the same from a fit of the four yields. These also need
  Ds and Ds* told apart, and without the photon that usually dominates them.

How the precision is computed. The covariance of the yields is the inverse of
the Fisher information of a binned Poisson fit, evaluated at the true yields:

  F_ab = Σ_i t_ai t_bi / ν_i

where t are the normalised MC templates and ν_i is the expected content of
bin i. Bins are equal-population bins; candidates where the observable is
undefined (no real neutrino solution, failed fit) go to one extra bin. The τ
channels are weighted to `--r-target`, and the result is scaled to 10⁵ Ds μ ν
candidates (`--n-ref`).

What the number does and does not include. It has no background, no
systematics and perfect templates. It is a figure of merit for ranking
reconstructions, not the sensitivity of the analysis. "× counting" is its
ratio to the perfect-separation limit √(1/N_τ + 1/N_μ).

Limitation. The templates come from the same finite MC as the pseudo-data, so
sparse bins make the separation look better than it is. `min_mc` and
`frac_low` in `results.json` give the smallest MC bin content and the
fraction of bins with fewer than 10 events. Compare entries with similar values.

The precision is computed for:

- every single observable (20 bins);
- (q², m²miss) and (q², E*μ), 12 × 12 bins;
- (q², m²miss, E*μ), 6 × 6 × 6 bins;
- the generator-level values of the same sets, as the reference for a perfect reconstruction;
- the outputs (p_τ, p_Ds*) of a multiclass BDT, 10 × 10 bins.

There is one BDT per scheme and per set of hypotheses (`coll`, `xyz`, `math`,
`all_hyps`), plus one with every scheme together. Each uses the same
topology and common inputs, with fixed hyperparameters and out-of-fold
predictions (2 folds).

Single-observable AUC. For τ vs μ, Ds τ vs Ds μ, and Ds* τ vs Ds* μ, it shows
how well each observable separates the channels. For Ds* μ vs Ds μ, it shows
how well it tells Ds* from Ds. The AUC is oriented (≥ 0.5), so a high value
means good separation in either direction. The heat maps put the
generator-level AUC in the first column as the reachable maximum.

*Degeneracy.* `max |ρ|` is the largest rank correlation among q², m²miss and
E*μ of one hypothesis. In the collinear approximation all three are
functions of m(Dsμ) alone: every Lorentz invariant built from
p(Bs) = k·p(Dsμ) depends only on m(Dsμ). So for `coll` |ρ| = 1, and the 2D
and 3D fits contain one variable. The run log flags every hypothesis with
|ρ| > 0.99.

**Resolution.** reco − gen per channel. q² is compared to (Bs − Ds)², which
is what is reconstructed, not to the W* mass in the Ds* channels; the photon
hypotheses are compared to the W* mass. The report gives:

- the median (bias);
- σ68, half the central 68% interval;
- the fraction beyond 3 σ68;
- the fraction where the observable is defined;
- the rank correlation of reco with gen.

pT(Bs) is a relative residual; χ is wrapped to (−π, π].

**Topology, per scheme.**

- Fit success and the vertex-probability cut efficiency.
- Bs vertex status.
- The flatness of P(V): the KS distance from a uniform distribution and the fraction below 1%.
- The Bs vertex residual and pull, along and across the true flight direction, separately for μ and τ channels. In the τ channels the fitted vertex is pulled towards the τ decay, which shows up as a bias along the flight.
- The angle between PV → V_Bs and the true Bs direction, which sets the quality of `xyz` and `math`.
- m(Ds) and pT(Ds) resolution.
- How often math1/math2 is the right solution.

## Output

| file | content |
|---|---|
| `index.html` | tables and every plot, with notes on how to read them |
| `summary.md` | the tables |
| `results.json` | every number; `results_long.csv` the same flattened, one per row |
| `plots/` | PNG (and PDF with `--pdf`) |

The main tables:

- per (scheme, hypothesis): q², m²miss and E*μ resolutions, AUCs, and σ τ/μ, σR(Ds), σR(Ds*) from the 2D/3D fits;
- per scheme: topology;
- the BDTs;
- the 40 best single observables.

## Files

| file | content |
|---|---|
| `compare_fits.py` | the driver |
| `cfio.py` | reading (uproot, cached), selection, derived quantities (`math_avg`, `math_oracle`, generator-level q² and m²miss), branch layout |
| `cfcore.py` | statistics, numpy only: robust residual statistics, weighted AUC, ROC, equal-population binning, Fisher precision |
| `cfanalysis.py` | the metrics |
| `cfplots.py` | the plots |
| `cfreport.py` | tables, `index.html`, `summary.md`, `results.json` |
| `make_toy_ntuple.py` | the toy ntuple |
| `test_compare_fits.py` | unit tests of `cfcore` |
