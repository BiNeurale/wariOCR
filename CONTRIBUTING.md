# Contributing to wariOCR

Bug reports and sample photos that break the detection are the most useful contributions.
Please open an issue with:

- the command you ran and its full output;
- if you can share it, the photo (only images you own and are happy to publish — never
  documents with personal data);
- what you expected the page to look like.

## Code contributions

wariOCR is copyright BiNeurale (Emanuele Lovato) and released under the
AGPL-3.0-or-later. To keep the option of offering it under other licences as well,
**pull requests can only be merged with a copyright assignment** to BiNeurale. If you want
to send code, open an issue first: I'll send you the assignment text, and we can agree on the
change before you spend time on it.

Before opening a pull request:

```sh
pip install -e '.[dev]'
pytest -q
ruff check wariocr tests tools
```

Keep changes small and focused, with a test when behaviour changes. Comments explain *why*,
not what the code already says.
