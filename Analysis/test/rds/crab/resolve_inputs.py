#!/usr/bin/env python3
'''
xrootd URLs to try for each input LFN, in order, the way cmsRun would:

  1. the site's own storage: the <data-access> catalogs of the site-local-config
     ($SITECONFIG_PATH/JobConfig/site-local-config.xml), each one turned into
     a root:// PFN with the XRootD protocol of that volume in the storage.json
     of its site (the local one, or the site= attribute: usually the regional
     federation, e.g. T1_IT_CNAF Eurasian_Federation, T1_US_FNAL
     American_Federation);
  2. the global redirector, last.

Catalogs whose protocol does not give a root:// URL (dcap, posix, mixed
without an XRootD rule) are skipped: FWLite opens files with TFile::Open.

    python3 resolve_inputs.py inputfiles.txt > sources.txt

prints, per input, one line with the candidate URLs separated by spaces.
Inputs that are already URLs (root://, file:) are printed as they are.
'''

import os
import re
import sys
import json
import xml.etree.ElementTree as ET

GLOBAL = 'root://cms-xrd-global.cern.ch/'
SITECONF = '/cvmfs/cms.cern.ch/SITECONF'


def site_dir():
    return os.environ.get('SITECONFIG_PATH') or os.path.join(SITECONF, 'local')


def data_access(sdir):
    '''[(site or None, volume, protocol)] from the site-local-config.'''
    path = os.path.join(sdir, 'JobConfig', 'site-local-config.xml')
    root = ET.parse(path).getroot()
    out = []
    for cat in root.iter('catalog'):
        if cat.get('volume') and cat.get('protocol'):
            out.append((cat.get('site'), cat.get('volume'), cat.get('protocol')))
    return out


def storage(sdir_or_site):
    path = sdir_or_site if os.path.isdir(sdir_or_site) else os.path.join(SITECONF, sdir_or_site)
    return json.load(open(os.path.join(path, 'storage.json')))


def protocol_entry(stor, volume, protocol):
    for v in stor:
        if v.get('volume') != volume:
            continue
        for p in v.get('protocols', []):
            if p.get('protocol') == protocol:
                return v, p
    return None, None


def apply(vol, p, lfn, depth=0):
    '''LFN -> PFN with one storage.json protocol entry (prefix, or rules with
    an optional chain to another protocol of the same volume).'''
    if depth > 5:
        return None
    if 'prefix' in p:
        return p['prefix'] + lfn
    for r in p.get('rules', []):
        m = re.match(r['lfn'] + '$', lfn)
        if not m:
            continue
        x = lfn
        if r.get('chain'):
            ch = next((q for q in vol.get('protocols', []) if q.get('protocol') == r['chain']), None)
            if ch is None:
                return None
            x = apply(vol, ch, lfn, depth + 1)
            if x is None:
                return None
            m = re.match(r['lfn'] + '$', x)
            if not m:
                return None
        return re.sub(r'\$(\d+)', lambda g: m.group(int(g.group(1))) or '', r['pfn'])
    return None


def sources(lfn, sdir=None):
    sdir = sdir or site_dir()
    out = []
    try:
        cats = data_access(sdir)
    except Exception as e:
        print('resolve_inputs: no site-local-config in %s (%s)' % (sdir, e), file=sys.stderr)
        cats = []
    for site, volume, protocol in cats:
        try:
            stor = storage(site if site else sdir)
        except Exception as e:
            print('resolve_inputs: no storage.json for %s (%s)' % (site or sdir, e), file=sys.stderr)
            continue
        protos = [protocol] if protocol == 'XRootD' else [protocol, 'XRootD']
        for proto in protos:
            vol, p = protocol_entry(stor, volume, proto)
            if p is None:
                continue
            pfn = apply(vol, p, lfn)
            if pfn and pfn.startswith('root://'):
                out.append(pfn)
                break
    out.append(GLOBAL + lfn)
    seen = set()
    return [u for u in out if not (u in seen or seen.add(u))]


def main():
    for line in open(sys.argv[1]):
        f = line.strip()
        if not f:
            continue
        if f.startswith('/store/'):
            print(' '.join(sources(f)))
        else:
            print(f)


if __name__ == '__main__':
    main()
