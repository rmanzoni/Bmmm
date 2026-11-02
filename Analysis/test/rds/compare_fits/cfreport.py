'''
Report of the fit-scheme comparison: summary tables (also as markdown) and
an index.html with the tables and every plot, plus results.json (all the
numbers) and results_long.csv (one row per number).
'''

import os
import csv
import json
import html

import numpy as np

from cfio import CLASSES, CLASS_NAME, SCHEME_LABEL, ORACLE_HYPS


def _fmt(x, f='%.4f'):
    if x is None:
        return '–'
    try:
        if not np.isfinite(x):
            return '–'
    except TypeError:
        return str(x)
    return f % x


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


##########################################################################################
#####      TABLES
##########################################################################################

KEY_SORT = 12      # key_table column it is sorted by: sigma N(tau)/N(mu), 3D


def key_table(res, layout):
    '''One row per (scheme, Bs hypothesis): resolutions, tau/mu separation and
    the expected precisions. Sorted by the 3D precision on N(tau)/N(mu) (the
    truth-picked hypothesis is shown, not ranked).'''
    header = ['scheme', 'hypothesis',
              'q² σ68 Dsμ', 'q² σ68 Dsτ', 'q² bias Dsμ', 'm²miss σ68 Dsμ', 'E*μ σ68 Dsμ',
              'AUC τ/μ q²', 'AUC τ/μ m²miss', 'AUC τ/μ E*μ', 'max |ρ| q²,m²,E*',
              'σ τ/μ q²,m²', 'σ τ/μ q²,m²,E*', 'σR(Ds) q²,m²,E*', 'σR(Ds*) q²,m²,E*']
    rows = []
    for s in layout.schemes:
        for h in layout.hyps[s]:
            r = _get(res, 'resolution', s, h) or {}
            sep = res.get('separation', {})
            md = _get(res, 'multidim', s, h) or {}
            b = lambda q: '%s_%s_%s' % (s, h, q)
            row = [s, h + (' (truth)' if h in ORACLE_HYPS else ''),
                   _get(r, 'q2', 'Ds mu', 's68'), _get(r, 'q2', 'Ds tau', 's68'),
                   _get(r, 'q2', 'Ds mu', 'median'), _get(r, 'm_miss2', 'Ds mu', 's68'),
                   _get(r, 'e_star_mu', 'Ds mu', 's68'),
                   _get(sep, b('q2'), 'tau_vs_mu', 'auc'), _get(sep, b('m_miss2'), 'tau_vs_mu', 'auc'),
                   _get(sep, b('e_star_mu'), 'tau_vs_mu', 'auc'),
                   _get(res, 'degeneracy', s, h, 'max_abs'),
                   _get(md, 'q2_mmiss2', 'r_taumu'), _get(md, 'q2_mmiss2_estar', 'r_taumu'),
                   _get(md, 'q2_mmiss2_estar', 'r_ds'), _get(md, 'q2_mmiss2_estar', 'r_dsst')]
            # the neutrino solutions impose m(nu)^2 = 0: m_miss2 is 0 by construction
            if row[5] is not None and abs(row[5]) < 1e-6:
                row[5] = '≡ 0'
            rows.append(row)
    key = lambda r: (r[1].endswith('(truth)'), r[KEY_SORT] if r[KEY_SORT] is not None else 9.)
    rows.sort(key=key)
    return header, rows


def scheme_table(res, layout):
    '''One row per scheme: fit success and cut efficiency (signal classes
    averaged), fit-probability tails, Bs vertex and direction resolution,
    Ds mass and pT resolution.'''
    header = ['scheme', 'Bs fit ok', 'vertex cuts', 'real ν sol.',
              'P(V_Bs)<1% μ', 'P(V_Bs)<1% τ', 'KS P(V_Ds) μ',
              'V_Bs long σ68 μ [µm]', 'V_Bs long bias τ [µm]', 'V_Bs trans σ68 μ [µm]',
              'pull long σ68 μ', 'angle σ68 μ [mrad]', 'm(Ds) σ68 [MeV]', 'pT(Ds) σ68 [%]']
    rows = []
    for s in layout.schemes:
        e = _get(res, 'efficiency', s) or {}
        mean = lambda k: np.nanmean([_get(e, str(c), k) if _get(e, str(c), k) is not None else np.nan
                                     for c in CLASSES])
        fq = _get(res, 'fit_quality', s) or {}
        vx = _get(res, 'vertex', s) or {}
        ms = _get(res, 'masses', 'mass', '%s_ds' % s)
        dpt = _get(res, 'masses', 'ds_pt', s)
        rows.append([s, mean('good_bs'), mean('vtx_cut'), mean('math_real'),
                     _get(fq, 'bs', 'mu', 'p_lt_001'), _get(fq, 'bs', 'tau', 'p_lt_001'),
                     _get(fq, 'ds', 'mu', 'ks_uniform'),
                     _get(vx, 'mu', 'long', 's68'), _get(vx, 'tau', 'long', 'median'),
                     _get(vx, 'mu', 't1', 's68'), _get(vx, 'mu', 'long_pull', 's68'),
                     _get(vx, 'mu', 'angle', 's68'),
                     1e3 * ms['s68'] if ms and ms.get('s68') is not None else None,
                     100 * dpt['s68'] if dpt and dpt.get('s68') is not None else None])
    return header, rows


def bdt_table(res):
    header = ['BDT', 'features', 'AUC τ/μ', 'AUC Dsτ/Dsμ', 'AUC Ds*τ/Ds*μ', 'AUC Ds*/Ds (μ)',
              'σ τ/μ', 'σR(Ds)', 'σR(Ds*)', '× counting τ/μ', '× counting R(Ds)', '× counting R(Ds*)']
    rows = []
    for name, r in sorted((res.get('bdt') or {}).items(), key=lambda kv: kv[1]['fisher'].get('r_taumu') or 9):
        f = r['fisher']
        rows.append([name, r['n_features'], r['auc_tau_vs_mu'], r['auc_ds_tau_vs_mu'],
                     r['auc_dsst_tau_vs_mu'], r['auc_dsst_vs_ds_mu'],
                     f.get('r_taumu'), f['r_ds'], f['r_dsst'], f.get('dil_taumu'), f['dil_ds'], f['dil_dsst']])
    return header, rows


def gamma_table(res, layout):
    """The photon hypotheses in the multidimensional fits, next to the xyz_bs
    references. Sorted by sigma R(Ds)."""
    header = ['scheme', 'hypothesis', 'observables', 'σ τ/μ', 'σR(Ds)', 'σR(Ds*)',
              '× counting R(Ds)', 'bins', 'min MC bin', 'bins < 10 MC']
    rows = []
    for s in layout.schemes:
        for h, sets in sorted((_get(res, 'multidim', s) or {}).items()):
            if 'gamma' not in h and h != 'xyz_bs':
                continue
            for name, f in sets.items():
                if not f:
                    continue
                rows.append([s, h, name, f.get('r_taumu'), f.get('r_ds'), f.get('r_dsst'), f.get('dil_ds'),
                             f.get('nbins'), f.get('min_mc'), f.get('frac_low')])
    rows.sort(key=lambda r: r[4] if r[4] is not None else 9.)
    return header, rows


def top_observables(res, n=40):
    header = ['observable', 'scheme', 'hypothesis', 'AUC τ/μ', 'AUC Dsτ/Dsμ', 'AUC Ds*τ/Ds*μ',
              'AUC Ds*/Ds (μ)', 'σ τ/μ 1D', 'σR(Ds) 1D', 'σR(Ds*) 1D']
    rows = []
    for key, r in (res.get('separation') or {}).items():
        if r.get('hyp') in ORACLE_HYPS:
            continue
        rows.append([key, r.get('scheme') or '–', r.get('hyp') or '–',
                     _get(r, 'tau_vs_mu', 'auc'), _get(r, 'ds_tau_vs_mu', 'auc'),
                     _get(r, 'dsst_tau_vs_mu', 'auc'), _get(r, 'dsst_vs_ds_mu', 'auc'),
                     _get(r, 'fisher', 'r_taumu'), _get(r, 'fisher', 'r_ds'), _get(r, 'fisher', 'r_dsst')])
    rows.sort(key=lambda x: -(x[3] or 0))
    return header, rows[:n]


def cutflow_table(res):
    header = ['step'] + [CLASS_NAME[c] for c in CLASSES]
    rows = [[name] + counts for name, counts in res['cutflow']]
    return header, rows


##########################################################################################
#####      OUTPUT
##########################################################################################

def _md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for r in rows:
        out.append('| ' + ' | '.join(c if isinstance(c, str) else (str(c) if isinstance(c, int) else _fmt(c)) for c in r) + ' |')
    return '\n'.join(out)


def _html_table(header, rows, best_cols=None):
    '''best_cols: {column index: 'min' | 'max'} -> the best value in bold.'''
    best = {}
    for j, how in (best_cols or {}).items():
        vals = [r[j] for r in rows if isinstance(r[j], (int, float)) and r[j] is not None
                and np.isfinite(r[j]) and not str(r[1]).endswith('(truth)')]
        if vals:
            best[j] = min(vals) if how == 'min' else max(vals)
    out = ['<table><thead><tr>' + ''.join('<th>%s</th>' % html.escape(h) for h in header) + '</tr></thead><tbody>']
    for r in rows:
        cells = []
        for j, c in enumerate(r):
            if isinstance(c, str):
                txt = html.escape(c)
            elif isinstance(c, int):
                txt = str(c)
            else:
                txt = _fmt(c)
            if j in best and c is not None and not isinstance(c, str) and np.isfinite(c) and c == best[j]:
                txt = '<b>%s</b>' % txt
            cells.append('<td>%s</td>' % txt)
        out.append('<tr>' + ''.join(cells) + '</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


CSS = '''
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; color: #222; max-width: 1500px; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 32px; border-bottom: 1px solid #ddd; }
h3 { font-size: 13px; margin-top: 18px; }
table { border-collapse: collapse; font-size: 11px; margin: 8px 0 16px 0; }
th, td { border: 1px solid #ddd; padding: 3px 6px; text-align: right; }
th { background: #f4f4f4; } td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; border: 1px solid #eee; margin: 6px 0; }
p.note { font-size: 12px; color: #444; max-width: 1100px; }
code { background: #f4f4f4; padding: 0 3px; }
'''


def write(outdir, res, layout, figures, meta):
    '''results.json, results_long.csv, summary.md, index.html.'''
    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(res, f, indent=1, default=lambda x: None)
    _long_csv(os.path.join(outdir, 'results_long.csv'), res)

    tabs = dict(key=key_table(res, layout), scheme=scheme_table(res, layout),
                bdt=bdt_table(res), gamma=gamma_table(res, layout), top=top_observables(res), cutflow=cutflow_table(res))

    md = ['# Fit-scheme comparison, R(Ds) signal MC', '',
          'Input: `%s` (%s)' % (meta['input'], meta['when']), '',
          'Schemes: %s. Selection: %s. τ weights to R(Ds) = %.3f, R(Ds*) = %.3f; precision scaled to %d Ds μ ν.'
          % (', '.join(layout.schemes), meta['selection'], meta['r_target'][0], meta['r_target'][1], meta['n_ref']), '',
          '## Per scheme and Bs hypothesis', '', _md_table(*tabs['key']), '',
          '## Per scheme', '', _md_table(*tabs['scheme']), '']
    if tabs['bdt'][1]:
        md += ['## BDT', '', _md_table(*tabs['bdt']), '']
    if tabs['gamma'][1]:
        md += ['## Photon hypotheses in the 2D / 3D fits', '', _md_table(*tabs['gamma']), '']
    md += ['## Cutflow', '', _md_table(*tabs['cutflow']), '']
    with open(os.path.join(outdir, 'summary.md'), 'w') as f:
        f.write('\n'.join(md))

    H = ['<!doctype html><html><head><meta charset="utf-8"><title>Fit-scheme comparison</title>',
         '<style>%s</style></head><body>' % CSS,
         '<h1>Fit-scheme comparison on the R(D<sub>s</sub>) signal MC</h1>',
         '<p class="note">Input <code>%s</code>, %s. Schemes: %s. Selection: %s.</p>'
         % (html.escape(meta['input']), meta['when'], ', '.join(SCHEME_LABEL.get(s, s) for s in layout.schemes),
            html.escape(meta['selection'])),
         '<p class="note"><b>How to read the precision numbers.</b> σR is the expected relative uncertainty on '
         'R(D<sub>s</sub>) or R(D<sub>s</sub><sup>*</sup>) from a binned fit of the four signal yields to the '
         'distribution of the observable(s), with perfect templates, no background and no systematics, the τ '
         'channels weighted to R = %.3f / %.3f and the sample scaled to %d D<sub>s</sub>μν candidates. It is a '
         'figure of merit for comparing reconstructions, not the sensitivity of the analysis. "× counting" is the '
         'ratio to perfect separation (√(1/N<sub>τ</sub> + 1/N<sub>μ</sub>)). Resolutions: σ<sub>68</sub> is half '
         'the central 68%% interval of reco − gen; q² is compared to (B<sub>s</sub> − D<sub>s</sub>)², the quantity '
         'reconstructed (D<sub>s</sub>, not D<sub>s</sub><sup>*</sup>). Bold: best of the column (truth-picked '
         'hypothesis excluded).</p>' % (meta['r_target'][0], meta['r_target'][1], meta['n_ref'])]

    def sec(title, note=None):
        H.append('<h2>%s</h2>' % title)
        if note:
            H.append('<p class="note">%s</p>' % note)

    def fig(key):
        if figures.get(key):
            H.append('<img src="%s">' % figures[key])

    sec('Summary per scheme and Bs hypothesis', 'Sorted by σ τ/μ from the 3D fit (q², m²<sub>miss</sub>, E*<sub>μ</sub>). '
        '"σ τ/μ": precision on N(τ)/N(μ) from a two-yield fit, D<sub>s</sub> and D<sub>s</sub><sup>*</sup> merged: '
        'the τ/μ separation alone. σR(D<sub>s</sub>), σR(D<sub>s</sub><sup>*</sup>): four-yield fit, which also needs '
        'D<sub>s</sub> and D<sub>s</sub><sup>*</sup> told apart and is usually limited by that. "max |ρ|": the largest '
        'rank correlation among q², m²<sub>miss</sub>, E*<sub>μ</sub> of that hypothesis; near 1 the 2D/3D fit has '
        'effectively one variable (always the case for the collinear approximation, where all three are functions '
        'of m(D<sub>s</sub>μ)). For the neutrino solutions (math) m²<sub>miss</sub> is 0 by construction.')
    H.append(_html_table(*tabs['key'], best_cols={2: 'min', 3: 'min', 5: 'min', 6: 'min', 7: 'max', 8: 'max', 9: 'max',
                                                 11: 'min', 12: 'min', 13: 'min', 14: 'min'}))
    sec('Summary per scheme', 'Efficiencies averaged over the four signal channels; vertex quantities for the μ '
        'channels, where V<sub>Bs</sub> is the B<sub>s</sub> decay vertex (in the τ channels it is pulled towards '
        'the τ decay: the long bias).')
    H.append(_html_table(*tabs['scheme'], best_cols={1: 'max', 2: 'max', 7: 'min', 9: 'min', 11: 'min', 12: 'min', 13: 'min'}))
    if tabs['bdt'][1]:
        sec('Multivariate', 'A multiclass BDT of the four channels per scheme and set of inputs (names in README: e.g. xyz = '
            'xyz_bs observables + topology + photon + common; a+b-c adds and removes feature groups); out-of-fold predictions. σR from a fit to the '
            '(p(τ), p(D<sub>s</sub><sup>*</sup>)) plane.')
        H.append(_html_table(*tabs['bdt'], best_cols={2: 'max', 6: 'min', 7: 'min', 8: 'min'}))
        fig('fisher_bdt')
        fig('roc_bdt')
        fig('bdt_scores')
    sec('Expected precision per observable set')
    fig('fisher_multidim')
    if tabs['gamma'][1]:
        sec('The photon in the 2D / 3D fits', 'gamma_xyz_bs: D<sub>s</sub>γ in place of the D<sub>s</sub> (q², '
            'm²miss only; E*μ from xyz_bs). xyz_bs+gamma: photon and xyz_bs observables together, q2g = q² with '
            'the photon. Events without a photon candidate have their own bin on the photon axes.')
        H.append(_html_table(*tabs['gamma'], best_cols={3: 'min', 4: 'min', 5: 'min'}))
        fig('fisher_gamma')
    fig('fisher_1d_taumu')
    fig('fisher_1d_ds')
    fig('fisher_1d_dsst')
    sec('τ/μ separation of single observables', 'Heat maps: AUC (oriented, 0.5 = none), colour relative to the '
        'best scheme/hypothesis of the row.')
    fig('auc_tau_mu')
    fig('auc_ds_tau_mu')
    fig('auc_dsst_tau_mu')
    fig('auc_dsst_ds_mu')
    fig('roc_q2')
    fig('roc_mmiss2')
    fig('roc_estar')
    H.append('<h3>Best single observables</h3>')
    H.append(_html_table(*tabs['top']))
    sec('Resolution summary', 'σ<sub>68</sub> of reco − gen, colour relative to the best of the row.')
    for g in ('Ds mu', 'Ds* mu', 'Ds tau', 'Ds* tau'):
        fig('res_summary_%s' % g.replace(' ', '_').replace('*', 'st'))
    fig('bias_summary_Ds_mu')
    sec('Resolution per observable', 'Rows: channels; columns: Bs hypotheses; colours: schemes. Legend: median / σ<sub>68</sub>.')
    for k in sorted(k for k in figures if k.startswith('res_') and not k.startswith('res_summary')):
        fig(k)
    fig('q2_vs_gen')
    sec('Shapes of the four channels', 'Rows: schemes; columns: Bs hypotheses.')
    for k in sorted(k for k in figures if k.startswith('shape_')):
        fig(k)
    fig('topology_shapes')
    sec('Fit quality, masses, vertices')
    fig('efficiency')
    fig('probabilities')
    fig('masses')
    fig('vertex')
    fig('math_choice')
    sec('Selection')
    H.append(_html_table(*tabs['cutflow']))
    fig('cutflow')
    H.append('<p class="note">All numbers: <code>results.json</code>, <code>results_long.csv</code>. Log: <code>log.txt</code>.</p>')
    H.append('</body></html>')
    with open(os.path.join(outdir, 'index.html'), 'w') as f:
        f.write('\n'.join(H))
    return tabs


def _long_csv(path, res):
    rows = []

    def walk(prefix, x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(prefix + [str(k)], v)
        elif isinstance(x, (int, float)) or x is None:
            rows.append(('/'.join(prefix), x))
    for k in ('efficiency', 'fit_quality', 'masses', 'vertex', 'resolution', 'separation', 'multidim'):
        walk([k], res.get(k, {}))
    for name, r in (res.get('bdt') or {}).items():
        walk(['bdt', name], dict((k, v) for k, v in r.items() if k != 'features'))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerows(rows)
