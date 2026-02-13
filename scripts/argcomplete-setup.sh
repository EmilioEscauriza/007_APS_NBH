# Source this from ~/.zshrc to enable tab completion for aps_analysis.
# Use: aps_analysis <TAB>  (with this dir on PATH).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WRAPPER="$SCRIPT_DIR/aps_analysis"

# Static list so completion works without running the (slow) Python script.
_aps_analysis() {
  _arguments \
    '1:command:(h5-inspector g2 ttc intensity-vs-time static-vs-dynamic-bins combined-plot q-spacing integrated-intensities-inspector integrated-intensities-plot bragg-peak-center bragg-peak-brightest bragg-peak-skewnorm bragg-peak-metrics make-qphi-maps check-qphi-npz oauth-test image-upload figure-upload)' \
    '--file-id[file ID (e.g. A073)]:' \
    '--filename[path to results HDF]:_files' \
    '--base-dir[base directory]:_files -/' \
    '(-h --help)'{-h,--help}'[show help]'
}

if [[ -n "$ZSH_VERSION" ]]; then
  autoload -Uz compinit && compinit -u 2>/dev/null
  compdef _aps_analysis aps_analysis "$WRAPPER"
fi
