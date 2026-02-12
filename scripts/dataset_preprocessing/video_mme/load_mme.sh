#!/usr/bin/env bash
set -euo pipefail

hf download lmms-lab/Video-MME  --repo-type dataset

# 1. Go to your HF snapshot directory
cd "${HF_HOME}/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b"
# 2. Extract all chunks into a local "videos" directory
unzip subtitle.zip
for f in videos_chunked_*.zip; do unzip "$f"; done