#!/usr/bin/env bash
# Install or remove a user LaunchAgent for MLXServe (macOS).
# Run ON the Mac where MLXServe lives (e.g. ollamac), after venv + pip install -e ".[mlx]".
#
# Default port is 8088 (not 8080) so MLXServe can coexist with stacks that bind host :8080
# (e.g. SafePerform deploy-docker-web). Override: MLXSERVE_PORT=8080 ./scripts/install_launchagent_macos.sh install
#
# Install (defaults):  ./scripts/install_launchagent_macos.sh install
# Uninstall:           ./scripts/install_launchagent_macos.sh uninstall
#
# Override repo root:  SOLOMLX_ROOT=/path/to/SoloMLX-server ./scripts/install_launchagent_macos.sh install

set -euo pipefail

LABEL="${MLXSERVE_LAUNCHD_LABEL:-com.mlxserve.server}"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
ROOT="${SOLOMLX_ROOT:-${HOME}/Documents/MLXServe/SoloMLX-server}"
MLXBIN="${ROOT}/.venv/bin/mlxserve"
OUT_LOG="${HOME}/Library/Logs/mlxserve.out.log"
ERR_LOG="${HOME}/Library/Logs/mlxserve.err.log"

# Defaults (override via env before calling this script)
MLXSERVE_HOST="${MLXSERVE_HOST:-127.0.0.1}"
MLXSERVE_PORT="${MLXSERVE_PORT:-8088}"
MLXSERVE_RUNTIME_BACKEND="${MLXSERVE_RUNTIME_BACKEND:-mlx}"
MLXSERVE_DEFAULT_MODEL="${MLXSERVE_DEFAULT_MODEL:-mlx-community/Mistral-Small-24B-Instruct-2501-4bit}"
MLXSERVE_PREFILL_STEP_SIZE="${MLXSERVE_PREFILL_STEP_SIZE:-512}"
MLXSERVE_KV_BITS="${MLXSERVE_KV_BITS:-4}"
MLXSERVE_QUANTIZED_KV_START="${MLXSERVE_QUANTIZED_KV_START:-32}"
MLXSERVE_MAX_MEMORY_GB="${MLXSERVE_MAX_MEMORY_GB:-14}"
MLXSERVE_HARD_MEMORY_GB="${MLXSERVE_HARD_MEMORY_GB:-15}"
MLXSERVE_IDLE_UNLOAD_ENABLED="${MLXSERVE_IDLE_UNLOAD_ENABLED:-true}"
# Warm default model at process start (first HTTP chat is fast; boot takes longer).
MLXSERVE_PRELOAD_DEFAULT_MODEL="${MLXSERVE_PRELOAD_DEFAULT_MODEL:-true}"

usage() {
  echo "Usage: $0 {install|uninstall}" >&2
  exit 1
}

cmd="${1:-}"
[[ "$cmd" == "install" || "$cmd" == "uninstall" ]] || usage

uninstall() {
  if [[ -f "$PLIST" ]]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed $PLIST (service stopped if it was loaded)."
  else
    echo "No plist at $PLIST"
  fi
}

install() {
  if [[ ! -x "$MLXBIN" ]]; then
    echo "ERROR: MLXServe binary not found or not executable: $MLXBIN" >&2
    echo "Set SOLOMLX_ROOT to your clone path, or create the venv first." >&2
    exit 1
  fi

  mkdir -p "${HOME}/Library/LaunchAgents" "${HOME}/Library/Logs"

  cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>WorkingDirectory</key>
  <string>${ROOT}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>MLXSERVE_HOST</key><string>${MLXSERVE_HOST}</string>
    <key>MLXSERVE_PORT</key><string>${MLXSERVE_PORT}</string>
    <key>MLXSERVE_RUNTIME_BACKEND</key><string>${MLXSERVE_RUNTIME_BACKEND}</string>
    <key>MLXSERVE_DEFAULT_MODEL</key><string>${MLXSERVE_DEFAULT_MODEL}</string>
    <key>MLXSERVE_PREFILL_STEP_SIZE</key><string>${MLXSERVE_PREFILL_STEP_SIZE}</string>
    <key>MLXSERVE_KV_BITS</key><string>${MLXSERVE_KV_BITS}</string>
    <key>MLXSERVE_QUANTIZED_KV_START</key><string>${MLXSERVE_QUANTIZED_KV_START}</string>
    <key>MLXSERVE_MAX_MEMORY_GB</key><string>${MLXSERVE_MAX_MEMORY_GB}</string>
    <key>MLXSERVE_HARD_MEMORY_GB</key><string>${MLXSERVE_HARD_MEMORY_GB}</string>
    <key>MLXSERVE_IDLE_UNLOAD_ENABLED</key><string>${MLXSERVE_IDLE_UNLOAD_ENABLED}</string>
    <key>MLXSERVE_PRELOAD_DEFAULT_MODEL</key><string>${MLXSERVE_PRELOAD_DEFAULT_MODEL}</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>${MLXBIN}</string>
    <string>serve</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${OUT_LOG}</string>
  <key>StandardErrorPath</key>
  <string>${ERR_LOG}</string>
</dict>
</plist>
EOF

  launchctl unload "$PLIST" 2>/dev/null || true
  if launchctl load "$PLIST" 2>/dev/null; then
    echo "Loaded LaunchAgent: $PLIST"
  elif launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null; then
    echo "Bootstrapped LaunchAgent (gui/$(id -u)): $PLIST"
  else
    echo "ERROR: launchctl load/bootstrap failed. Try: launchctl load $PLIST" >&2
    exit 1
  fi

  echo "Logs: $OUT_LOG / $ERR_LOG"
  echo "Check: launchctl list | grep ${LABEL}"
  echo "Health: curl -s http://${MLXSERVE_HOST}:${MLXSERVE_PORT}/health"
}

case "$cmd" in
  install) install ;;
  uninstall) uninstall ;;
esac
