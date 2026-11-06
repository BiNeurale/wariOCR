#!/bin/sh
# Rebuild the demo photos in examples/ from scratch: sheet image -> Blender render -> phone JPEG.
# Usage: tools/demo/make_photos.sh [WORKDIR]   (needs Blender on PATH and numpy+Pillow in python3; ~10 min on CPU)
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
work=${1:-$root/build/demo}
py=${PYTHON:-python3}
mkdir -p "$work" "$root/examples"
"$py" "$here/page.py" "$work/page.png"
for v in 1 2; do
    blender --background --python-exit-code 1 --python "$here/scene.py" -- \
        "$work/page.png" "$work/render-$v.png" "$v" 75 > "$work/blender-$v.log" 2>&1
    "$py" "$here/finish.py" "$work/render-$v.png" "$root/examples/notice-$v.jpg" "$v"
done
