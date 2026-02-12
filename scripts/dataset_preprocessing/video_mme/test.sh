#!/usr/bin/env bash
# 1. Check if the subtitles filter is even compiled in
ffmpeg -filters | grep subtitles

# 2. Check for missing shared libraries
# This often reveals if libfreetype or libfontconfig are 'not found'
ldd $(which ffmpeg) | grep -E "freetype|fontconfig|fribidi"

# 3. Test a dummy subtitle burn with a simple generated video
# This isolates the issue from your specific .srt or .mp4 files
echo -e "1\n00:00:00,000 --> 00:00:10,000\nTest Subtitle" > test.srt
ffmpeg -y -f lavfi -i testsrc=duration=5:size=640x360:rate=30 -vf "subtitles=test.srt" debug_out.mp4