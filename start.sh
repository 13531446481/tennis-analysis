#!/usr/bin/env bash
set -euo pipefail

need_cmd() { command -v "$1" >/dev/null 2>&1; }

echo "[1/5] Checking prerequisites..."

# ---- auto install curl if missing ----
if ! need_cmd curl; then
  echo "[1/5] curl not found, installing..."
  if need_cmd apt; then
    apt update
    apt install -y curl
  elif need_cmd yum; then
    yum install -y curl
  elif need_cmd apk; then
    apk add --no-cache curl
  else
    echo "ERROR: package manager not found, install curl manually."
    exit 1
  fi
fi

# ---- ensure PATH for current shell (even if .bashrc not loaded) ----
export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"
hash -r || true

# ---- persist PATH to ~/.bashrc and ~/.profile (idempotent) ----
OPENCODE_PATH_LINE='export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"'

for rc in "$HOME/.bashrc" "$HOME/.profile"; do
  if [ -f "$rc" ]; then
    if ! grep -Fq "$OPENCODE_PATH_LINE" "$rc"; then
      echo "$OPENCODE_PATH_LINE" >> "$rc"
      echo "[PATH] Added opencode paths to $rc"
    fi
  else
    echo "$OPENCODE_PATH_LINE" > "$rc"
    echo "[PATH] Created $rc with opencode paths"
  fi
done

# =========================================================
# 🔒 HARDCODED DEFAULTS (press Enter to use)
# =========================================================
DEFAULT_RELAY_URL="https://berlfpdvarqc.ap-northeast-1.clawcloudrun.com"
DEFAULT_RELAY_KEY="sk-gI6wLbmQgS9x749sfuPtAiynM3j7eLsPWy2d238RecoHvCN1"

read -r -p "Relay base URL [default: ${DEFAULT_RELAY_URL}]: " RELAY_URL
read -r -s -p "API key [default preset, press Enter to use]: " RELAY_KEY
echo

RELAY_URL="${RELAY_URL:-$DEFAULT_RELAY_URL}"
RELAY_KEY="${RELAY_KEY:-$DEFAULT_RELAY_KEY}"

# ---- normalize baseURL to end with /v1 ----
RELAY_URL="${RELAY_URL%/}"
if [[ "$RELAY_URL" != *"/v1" ]]; then
  BASE_URL="${RELAY_URL}/v1"
else
  BASE_URL="$RELAY_URL"
fi

PROVIDER_ID="relay"
MODEL_ID="gpt-5.2-codex"

echo "[2/5] Ensuring opencode is available..."
if ! need_cmd opencode; then
  echo "[2/5] opencode not found, installing via jsDelivr..."
  # Force IPv4 to avoid IPv6 issues in some containers
  curl -4 -fsSL https://fastly.jsdelivr.net/gh/anomalyco/opencode@dev/install | bash
  export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"
  hash -r || true
fi

if ! need_cmd opencode; then
  echo "ERROR: opencode still not found after install."
  echo "Try: source ~/.bashrc && opencode"
  exit 1
fi

echo "[2/5] opencode found: $(command -v opencode)"

echo "[3/5] Cleaning invalid global config (if any)..."
rm -f "$HOME/.config/opencode/config.json" || true

echo "[4/5] Writing OpenCode provider config..."
mkdir -p "$HOME/.config/opencode"

cat > "$HOME/.config/opencode/opencode.json" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "${PROVIDER_ID}": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "NewAPI (aggregated)",
      "options": {
        "baseURL": "${BASE_URL}",
        "apiKey": "${RELAY_KEY}"
      },
      "models": {
        "${MODEL_ID}": {
          "name": "GPT-5.2-Codex (NewAPI)"
        }
      }
    }
  },
  "model": "${PROVIDER_ID}/${MODEL_ID}",
  "small_model": "${PROVIDER_ID}/${MODEL_ID}"
}
EOF

echo "[5/5] Done."
echo " - Config written to: $HOME/.config/opencode/opencode.json"
echo " - Base URL: ${BASE_URL}"
echo " - Default model: ${PROVIDER_ID}/${MODEL_ID}"

echo
echo "Launching OpenCode TUI..."
opencode