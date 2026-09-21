#!/usr/bin/env python3
'''Assert that `import hammer` gives the v2 Cython (snake_case) API.

Sourced by hammer_env.sh / hammer_env.csh right after the paths are set, so a
cppyy build -- which imports fine but exposes the CamelCase C++ names instead --
fails here rather than three hours into a reweighting job.
'''
import sys

try:
    import hammer
except ImportError as exc:
    sys.exit('[FATAL] cannot import hammer: %s\n'
             '        check HAMMER_PREFIX / PYTHONPATH, or run install_hammer.sh' % exc)

# every method Bmmm.Analysis.HammerFF calls, and the four module-level types
# HammerSession constructs. Verified present in v2.0.0's Cython bindings.
REQUIRED = ('include_decay', 'add_ff_scheme', 'set_ff_input_scheme',
            'set_ff_eigenvectors', 'init_run', 'init_event', 'add_process',
            'process_event', 'get_weight', 'set_options', 'set_units')
REQUIRED_TYPES = ('Hammer', 'Process', 'Particle', 'FourMomentum')

missing_types = [name for name in REQUIRED_TYPES if not hasattr(hammer, name)]
if missing_types:
    sys.exit('[FATAL] hammer imported from %s but is missing %s at module level'
             % (hammer.__file__, missing_types))

ham = hammer.Hammer()
missing = [name for name in REQUIRED if not hasattr(ham, name)]
if missing:
    sys.exit('[FATAL] hammer imported from %s but the Cython (snake_case) API is '
             'missing %s -- rebuild with -DPYTHON_USE_CPPYY=OFF'
             % (hammer.__file__, missing))

# the attribute is `version`, not `__version__`
print('#### hammer %s from %s' % (getattr(hammer, 'version', '?'), hammer.__file__))
print('####   %d/%d methods and %d/%d types present'
      % (len(REQUIRED), len(REQUIRED), len(REQUIRED_TYPES), len(REQUIRED_TYPES)))
