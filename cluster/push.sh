#!/usr/bin/env bash
# Copies this repo (code only, no data or checkpoints) to
# /scratch/$USER/sar/code on the cluster.
#   cluster/push.sh            # host alias "explorer"
#   SAR_HOST=myhpc cluster/push.sh
#
# It needs an already-open ssh master connection and will not open one: every
# new login costs a Duo prompt, and repeated failed logins lock the account
# out of the login nodes. Open it yourself first with `ssh explorer true`.
set -euo pipefail
HOST=${SAR_HOST:-explorer}
if ! ssh -O check "$HOST" 2>/dev/null; then
  echo "no ssh master connection to $HOST; run: ssh $HOST true" >&2
  exit 1
fi
# BatchMode: if the master drops mid-run, fail instead of trying a password.
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
cd "$(dirname "$0")/.."

remote_user=$("${SSH[@]}" "$HOST" 'echo $USER')
dest=/scratch/$remote_user/sar/code
# tar over ssh: scp to this cluster drops connections intermittently. On macOS,
# COPYFILE_DISABLE and the two flags keep AppleDouble ._ files and extended
# attributes out of the archive.
tar_flags=()
if [ "$(uname)" = Darwin ]; then
  export COPYFILE_DISABLE=1
  tar_flags=(--no-mac-metadata --no-xattrs)
fi
git ls-files -co --exclude-standard -z |
  tar "${tar_flags[@]}" --null -T - -cf - |
  "${SSH[@]}" "$HOST" "mkdir -p $dest /scratch/$remote_user/sar/logs && tar -C $dest -xf -"
# The cluster copy has no .git: record which commit (and whether the tree
# was dirty) it came from, so every run can log its code version.
{ git rev-parse HEAD 2>/dev/null || echo "no-commit"; git status --porcelain | head -20; } |
  "${SSH[@]}" "$HOST" "cat > $dest/VERSION"
echo "pushed to $HOST:$dest"
