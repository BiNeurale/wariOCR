# Where the demo images come from

All images in this folder and in `examples/` were made for this repository and are covered by
its licence. No third-party photos or textures are used.

- **`examples/notice-1.jpg`, `examples/notice-2.jpg`** — synthetic "phone photos". The sheet
  (a notice of a fictional village library; every name, address and date is invented) is
  drawn by `tools/demo/page.py` with the Liberation fonts, rendered on a procedural table with
  Blender/Cycles by `tools/demo/scene.py` (camera tilt, gentle curl, a folded-in-three page,
  a soft shadow of the phone, warm/cool light), then given sensor noise, vignetting and JPEG
  compression by `tools/demo/finish.py`. Rebuild them with `tools/demo/make_photos.sh`.
- **`hero.jpg`, `pipeline.gif`, `terminal.gif`, `social-preview.png`** — assembled by
  `tools/demo/media.py` from a real run of the `wariocr` command on those photos: the
  outlines, rectified sheets and final pages are the program's own output files, and the
  terminal recording replays the command's actual output with its actual timing.
