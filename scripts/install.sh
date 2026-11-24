#!/usr/bin/env bash
# Link both plugins into the Hermes plugin directories (symlink — edits in the repo apply live).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

for kind in image_gen video_gen; do
  mkdir -p "$HERMES_HOME/plugins/$kind"
  ln -sfn "$ROOT/$kind" "$HERMES_HOME/plugins/$kind/alibaba-token-plan"
  echo "linked $HERMES_HOME/plugins/$kind/alibaba-token-plan -> $ROOT/$kind"
done

cat <<'EOF'

Now enable the plugins and pick the backend:

  hermes plugins enable image_gen/alibaba-token-plan
  hermes plugins enable video_gen/alibaba-token-plan
  hermes tools            # Image Generation / Video Generation -> Alibaba Token Plan

The media API key is independent of any chat-provider key. Store it either in
$HERMES_HOME/.env as ALIBABA_TOKEN_PLAN_MEDIA_API_KEY=sk-sp-...
or in ~/.alibaba/token-plan-media.key (chmod 600).
EOF
