'''
Write a FrameworkJobReport.xml for a scriptExe job.

WHY THIS EXISTS
---------------
CRAB's post-job always parses FrameworkJobReport.xml, whatever the job ran. A
cmsRun job gets one for free -- the framework writes it. A scriptExe job does
not, and CRAB then fails the job with

    exit code 50115 : BadFWJRXML

which says nothing about the analysis and everything about the missing file.
That is the failure mode this fixes; it is the same thing NanoAOD-tools does
from python at the end of its own scriptExe.

WHAT IT REPORTS
---------------
The output ntuple (name, size, entries) and the input LFNs, plus the runs and
lumisections actually read -- taken from the JSON the ntuplizer writes with
--lumi-json, which counts EVERY event the loop touched, not just the ones that
survived selection. Getting that right is what makes `crab report` and the
processed-lumi accounting truthful; without the JSON the report is still valid
XML and the job passes, but CRAB will believe no lumis were processed.

    python3 make_fjr.py --output dimuon_ntuple.root \
                        --inputs pfns.txt          \
                        --lumi-json processed_lumis.json
'''

from __future__ import print_function

import argparse
import json
import os
import sys


def esc(text):
    '''XML-escape, so a PFN with an & in the query string cannot break the report.'''
    return (str(text).replace('&', '&amp;').replace('<', '&lt;')
                     .replace('>', '&gt;').replace('"', '&quot;'))


def runs_block(processed, indent):
    '''<Runs> ... </Runs> from {run: {lumi: nevents}}, or '' if unknown.'''
    if not processed:
        return ''
    pad   = ' ' * indent
    lines = ['%s<Runs>' % pad]
    for run in sorted(processed, key=int):
        lines.append('%s  <Run ID="%s">' % (pad, esc(run)))
        for lumi in sorted(processed[run], key=int):
            lines.append('%s    <LumiSection ID="%s" NEvents="%s"/>'
                         % (pad, esc(lumi), esc(processed[run][lumi])))
        lines.append('%s  </Run>' % pad)
    lines.append('%s</Runs>' % pad)
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output'   , required=True,
                        help='the ntuple this job produced')
    parser.add_argument('--inputs'   , default='',
                        help='file with one input LFN/PFN per line')
    parser.add_argument('--lumi-json', dest='lumi_json', default='',
                        help='the ntuplizer --lumi-json output')
    parser.add_argument('--fjr'      , default='FrameworkJobReport.xml')
    args = parser.parse_args()

    # ---- what the job read ---------------------------------------------------
    processed, events_read = {}, 0
    if args.lumi_json and os.path.isfile(args.lumi_json):
        try:
            with open(args.lumi_json) as fin:
                payload = json.load(fin)
            processed   = payload.get('processed_lumis', {})
            events_read = int(payload.get('events_read', 0))
        except (ValueError, KeyError, TypeError) as err:
            print('WARNING: could not read %s (%s); the report will claim no '
                  'lumis were processed' % (args.lumi_json, err), file=sys.stderr)
    else:
        print('WARNING: no lumi JSON at %r; the report will be valid but will '
              'claim no lumis were processed' % args.lumi_json, file=sys.stderr)

    inputs = []
    if args.inputs and os.path.isfile(args.inputs):
        with open(args.inputs) as fin:
            inputs = [ln.strip() for ln in fin if ln.strip()]

    # ---- what the job wrote --------------------------------------------------
    out_name  = os.path.basename(args.output)
    out_bytes = os.path.getsize(args.output) if os.path.isfile(args.output) else 0
    out_events = 0
    try:
        import uproot
        with uproot.open(args.output) as fout:
            out_events = int(fout['tree'].num_entries)
    except Exception as err:
        # a missing or unreadable output is a real problem, but it is not this
        # script's job to hide it: report zero and let CRAB see an empty output
        print('WARNING: could not count entries in %s (%s)' % (args.output, err),
              file=sys.stderr)

    # ---- the report ----------------------------------------------------------
    # Structure and the StorageStatistics block follow what CRAB's post-job
    # expects to find; the values it actually uses are the file blocks below.
    parts = ['<FrameworkJobReport>\n',
             '<ReadBranches>\n</ReadBranches>\n',
             '<PerformanceReport>\n'
             '  <PerformanceSummary Metric="StorageStatistics">\n'
             '    <Metric Name="Parameter-untracked-bool-enabled" Value="true"/>\n'
             '    <Metric Name="Parameter-untracked-bool-stats" Value="true"/>\n'
             '    <Metric Name="Parameter-untracked-string-cacheHint" Value="application-only"/>\n'
             '    <Metric Name="Parameter-untracked-string-readHint" Value="auto-detect"/>\n'
             '    <Metric Name="ROOT-tfile-read-totalMegabytes" Value="0"/>\n'
             '    <Metric Name="ROOT-tfile-write-totalMegabytes" Value="%d"/>\n'
             '  </PerformanceSummary>\n'
             '</PerformanceReport>\n' % (out_bytes // (1024 * 1024)),
             '<GeneratorInfo>\n</GeneratorInfo>\n']

    parts.append(
        '<File>\n'
        '  <LFN></LFN>\n'
        '  <PFN>%s</PFN>\n'
        '  <Catalog></Catalog>\n'
        '  <ModuleLabel>dimuon</ModuleLabel>\n'
        '  <OutputModuleClass>PoolOutputModule</OutputModuleClass>\n'
        '  <GUID></GUID>\n'
        '  <DataType></DataType>\n'
        '  <BranchHash></BranchHash>\n'
        '  <TotalEvents>%d</TotalEvents>\n'
        '  <Size>%d</Size>\n'
        '%s'
        '</File>\n' % (esc(out_name), out_events, out_bytes, runs_block(processed, 2)))

    for lfn in inputs:
        parts.append(
            '<InputFile>\n'
            '  <LFN></LFN>\n'
            '  <PFN>%s</PFN>\n'
            '  <Catalog></Catalog>\n'
            '  <InputType>primaryFiles</InputType>\n'
            '  <ModuleLabel>source</ModuleLabel>\n'
            '  <GUID></GUID>\n'
            '  <InputSourceClass>PoolSource</InputSourceClass>\n'
            '  <EventsRead>%d</EventsRead>\n'
            '</InputFile>\n' % (esc(lfn), events_read if len(inputs) == 1 else 0))

    parts.append('</FrameworkJobReport>\n')

    with open(args.fjr, 'w') as fout:
        fout.write(''.join(parts))

    # parse it back: a report that does not parse is exactly the failure this
    # script exists to prevent, so never leave one behind unchecked
    try:
        import xml.etree.ElementTree as ET
        ET.parse(args.fjr)
    except Exception as err:
        print('ERROR: wrote an unparsable %s (%s)' % (args.fjr, err), file=sys.stderr)
        return 1

    nlumi = sum(len(v) for v in processed.values())
    print('wrote %s: %d entries out, %d events read, %d run(s) / %d lumi(s), %d input file(s)'
          % (args.fjr, out_events, events_read, len(processed), nlumi, len(inputs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
