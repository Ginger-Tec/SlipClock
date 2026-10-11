# Korean figure-font source record

The Korean-only glyph subsets in this directory come from the Korean face
(TTC face index 1) of Ubuntu package `fonts-noto-cjk`
`1:20230817+repack1-3`, whose upstream is
<https://github.com/notofonts/noto-cjk>.

Pinned source files:

- `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`:
  `b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a`
- `/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc`:
  `faa5f3656a78b2e2d450d27fe8382c778bc2b6bb5ea29c986664a6a435056ceb`

Each face was subset with FontTools 4.63.0 to the Hangul characters in
`subset_text.txt`, converted from CFF to quadratic outlines with `otf2ttf`
0.2.3, and renamed to `SlipClock Noto Sans KR`.  Latin letters, numbers, and
punctuation in the Korean chart remain Linux Biolinum O.  The small derived
fonts avoid the Matplotlib Type-42 overflow caused by the full 65k-glyph TTC.

Derived-file SHA-256 values:

- `SlipClockNotoSansKR-Regular.ttf`:
  `c8a710faa3e9d5a02541ce81fa2233b35dbb77986391bba23139747168eadf78`
- `SlipClockNotoSansKR-Bold.ttf`:
  `4b5ff327edd4068342ab599f8b42e4841cf290d2f6ce2081a7ef6255a42a0722`

The accompanying `DEBIAN_COPYRIGHT` preserves the Debian package record. A
standalone `OFL.txt` reproduces the applicable SIL Open Font License 1.1 text
and upstream copyright notice for redistribution with the subset fonts.
