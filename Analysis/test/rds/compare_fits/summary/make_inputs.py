#!/usr/bin/env python3
'''Figure and LaTeX tables of the summary note, generated from
hypinfo_full/hypinfo.json (so that no number is copied by hand).

    python3 make_inputs.py ../hypinfo_full/hypinfo.json
'''
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '../hypinfo_full/hypinfo.json'))
B, BDT = r['bootstrap'], r['bdt']

FAM = [  # (family, colour (default categorical slots 1-4), marker, sets)
    ('kinematics only', '#2a78d6', 'o', ['q3', 'coll_obs', 'math_obs', 'xyz+coll', 'xyz+math', 'xyz+coll+math']),
    ('raw measurements', '#eb6834', 's', ['raw', 'inv', 'raw+inv', 'raw+inv+xyz', 'raw+inv+xyz+coll+math']),
    ('positive controls', '#1baf7a', 'D', ['CONTROL xyz+oracle', 'CONTROL xyz+gen_q2_smeared']),
    ('with context', '#eda100', '^', ['ctx xyz', 'ctx xyz+coll+math', 'ctx all']),
]
LABEL = {'CONTROL xyz+oracle': 'xyz + oracle', 'CONTROL xyz+gen_q2_smeared': r'xyz + gen $q^2$ (1 GeV$^2$)',
         'ctx xyz': 'xyz + ctx', 'ctx xyz+coll+math': 'xyz+coll+math + ctx',
         'ctx all': r'xyz+coll+math+$\gamma$hyp + ctx'}
TEXLABEL = {'CONTROL xyz+oracle': r'xyz + oracle', 'CONTROL xyz+gen_q2_smeared': r'xyz + gen.\ $q^2$ smeared',
            'ctx xyz': r'xyz + ctx', 'ctx xyz+coll+math': r'xyz+coll+math + ctx',
            'ctx all': r'xyz+coll+math+$\gamma$-hyp + ctx'}
Q = ((0, r'$\sigma_{\tau/\mu}$'), (1, r'$\sigma R(D_s)$'), (2, r'$\sigma R(D_s^*)$'))
FOM = 1  # b20


def rel(name, ref='xyz_obs'):
    if ref == 'xyz_obs':
        x = B[name]['b20']['rel_to_xyz_obs']
    else:
        x = B[name]['b20']['rel_to_ctx xyz']
    return 100 * np.array(x['mean']), 100 * np.array(x['std'])


# ---- figure -----------------------------------------------------------------
rows = []
for fam, col, mk, sets in FAM:
    for s in sets:
        rows.append((fam, col, mk, s))
n = len(rows)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.36 * n + 1.6), sharey=True)
XL = (-55, 32)
for (iq, title), ax in zip(Q, axes):
    for k, (fam, col, mk, s) in enumerate(rows):
        y = n - 1 - k
        mu, sd = rel(s)
        v, e = mu[iq], sd[iq]
        if v > XL[1]:
            ax.annotate('%+.0f%% \u2192' % v, (XL[1], y), xytext=(-2, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=8, color='#555555')
            continue
        ax.errorbar(v, y, xerr=e, fmt=mk, color=col, ms=6, capsize=0, lw=1.5, mec='white', mew=0.8)
    ax.axvline(0, color='#333333', lw=1)
    ax.set_xlim(*XL)
    ax.set_title(title + '  vs xyz_obs', fontsize=11)
    ax.set_xlabel('relative change [%]   (negative = better)', fontsize=9)
    ax.grid(axis='x', color='#e5e5e5', lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    y0 = n
    for fam, col, mk, sets in FAM[:-1]:
        y0 -= len(sets)
        ax.axhline(y0 - 0.5, color='#cccccc', lw=0.8, ls='--')
axes[0].set_yticks(range(n))
axes[0].set_yticklabels([LABEL.get(s, s) for _, _, _, s in rows][::-1], fontsize=9)
handles = [plt.Line2D([], [], marker=mk, color=col, ls='', ms=7, label=fam) for fam, col, mk, _ in FAM]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig('bdt_gain.pdf')
fig.savefig('bdt_gain.png', dpi=130)

# ---- tables ---------------------------------------------------------------------
def nfeat(s):
    return len(BDT[s + '|long']['features'])


def tex(s):
    return TEXLABEL.get(s, s).replace('_', r'\_')


with open('tab_bdt_full.tex', 'w') as f:
    f.write('\\begin{tabular}{lr|ccc|c|ccc}\n\\toprule\n')
    f.write(' & & \\multicolumn{3}{c|}{expected precision (20$\\times$20 bins)} & & '
            '\\multicolumn{3}{c}{change vs \\code{xyz\\_obs} (paired bootstrap)} \\\\\n')
    f.write('inputs & $n$ & $\\sigma_{\\tau/\\mu}$ & $\\sigma R(\\Ds)$ & $\\sigma R(\\Dss)$ & log-loss & '
            '$\\tau/\\mu$ & $R(\\Ds)$ & $R(\\Dss)$ \\\\\n\\midrule\n')
    for fam, _, _, sets in [('reference', None, None, ['xyz_obs'])] + FAM:
        f.write('\\multicolumn{9}{l}{\\emph{%s}} \\\\\n' % fam)
        for s in sets:
            v = B[s]['b20']['mean']
            mu, sd = rel(s)
            ll = BDT[s + '|long']['runs'][0]['logloss']
            cells = ['--'] * 3 if s == 'xyz_obs' else ['$%+.1f\\pm%.1f$' % (a, b) for a, b in zip(mu, sd)]
            f.write('\\quad %s & %d & %.4f & %.3f & %.3f & %.4f & %s \\\\\n'
                    % (tex(s), nfeat(s), v[0], v[1], v[2], ll, ' & '.join(cells)))
    f.write('\\bottomrule\n\\end{tabular}\n')

with open('tab_bdt_ctx.tex', 'w') as f:
    f.write('\\begin{tabular}{lr|ccc}\n\\toprule\n')
    f.write('added to xyz + ctx & $n$ & $\\tau/\\mu$ & $R(\\Ds)$ & $R(\\Dss)$ \\\\\n\\midrule\n')
    for s, lab in (('ctx xyz+coll+math', 'coll + math'), ('ctx all', r'coll + math + photon hypotheses')):
        mu, sd = rel(s, 'ctx')
        f.write('%s & %d & %s \\\\\n' % (lab, nfeat(s), ' & '.join('$%+.1f\\pm%.1f\\%%$' % (a, b)
                                                                  for a, b in zip(mu, sd))))
    f.write('\\bottomrule\n\\end{tabular}\n')

def varname(v):
    v = v[len('free_'):]
    for h, ht in (('xyz_bs_', 'xyz'), ('math_avg_', 'math\\_avg'), ('math1_bs_', 'math1'), ('math2_bs_', 'math2'),
                  ('coll_', 'coll')):
        if v.startswith(h):
            q = v[len(h):]
            return '%s:%s' % (ht, {'q2': '$q^2$', 'm_miss2': '$m^2_\\text{miss}$', 'e_star_mu': '$E^*_\\mu$'}[q])
    return v.replace('_', '\\_')


# best binned combinations
with open('tab_binned.tex', 'w') as f:
    f.write('\\begin{tabular}{llccc c}\n\\toprule\n')
    f.write('dim & observables & $\\sigma_{\\tau/\\mu}$ & $\\sigma R(\\Ds)$ & $\\sigma R(\\Dss)$ & min MC/bin \\\\\n'
            '\\midrule\n')
    for dim in ('2d', '3d'):
        L = [x for x in r['binned'][dim] if x['r_ds'] is not None]
        pure = [x for x in L if all(v.startswith('free_xyz_bs_') for v in x['vars'])]
        picks = [min(pure, key=lambda x: x['r_ds'])] + sorted(L, key=lambda x: x['r_ds'])[:2]
        seen = set()
        for x in picks:
            key = tuple(x['vars'])
            if key in seen:
                continue
            seen.add(key)
            obs = ', '.join(varname(v) for v in x['vars'])
            f.write('%s & %s & %.4f & %.3f & %.3f & %d \\\\\n'
                    % (dim.upper(), obs, x['r_taumu'], x['r_ds'], x['r_dsst'], int(x['min_mc'])))
        if dim == '2d':
            f.write('\\midrule\n')
    f.write('\\bottomrule\n\\end{tabular}\n')
print('ok')
