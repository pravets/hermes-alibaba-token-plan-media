#!/usr/bin/env bash
# Remove the symlinks created by install.sh (the Hermes config entries stay; disable them too).
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

for kind in image_gen video_gen; do
  target="$HERMES_HOME/plugins/$kind/alibaba-token-plan"
  if [ -L "$target" ]; then
    rm -f "$target"
    echo "removed $target"
  fi
done

echo "Also run: hermes plugins disable image_gen/alibaba-token-plan video_gen/alibaba-token-plan"
