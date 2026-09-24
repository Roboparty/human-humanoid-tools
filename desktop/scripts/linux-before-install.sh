#!/bin/sh

# Early releases registered /usr/bin/hhtools as an alternative for the
# Electron binary. Remove only that exact legacy target; the thin GUI package
# deliberately leaves the separately installed Python CLI untouched.
legacy_gui='/opt/Human-Humanoid Tools/hhtools'

if command -v update-alternatives >/dev/null 2>&1 && \
   update-alternatives --query hhtools 2>/dev/null | \
       grep -Fqx "Alternative: $legacy_gui"; then
    update-alternatives --remove hhtools "$legacy_gui" || true
fi

if [ -L /usr/bin/hhtools ] && \
   [ "$(readlink -f /usr/bin/hhtools 2>/dev/null)" = "$legacy_gui" ]; then
    rm -f /usr/bin/hhtools
fi

printf '%s\n' \
    'Human-Humanoid Tools declares its Linux GUI libraries in this Debian package.' \
    'When using dpkg directly, if it reports missing dependencies, run:' \
    '  sudo apt-get -f install'
