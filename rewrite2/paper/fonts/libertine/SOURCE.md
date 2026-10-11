# Linux Biolinum source record

The three OpenType files in this directory are extracted unchanged from the
CTAN `libertine` package, version 5.3.0 (2024-12-04):

- Package page: <https://ctan.org/pkg/libertine>
- TDS archive: <https://mirrors.ctan.org/install/fonts/libertine.tds.zip>
- Downloaded archive SHA-256: `4f06db8de016fe5e4109d7b6d4e6078b14958aca0e8528cd1d6620e9853ddba7`

Files and SHA-256 values:

- `LinBiolinum_R.otf`: `a3440f6d67878de395c8b34074bfb611e6e0721d60eeb0608acba43522ccbfd1`
- `LinBiolinum_RB.otf`: `196fa146480ffee1c9abcb68862087599bfc6ccac68633f0fcf04bdd70527b00`
- `LinBiolinum_RI.otf`: `8b94c5d8ffd7fb726b0c3705b85728e6648e9530714e477c9943d87d7793ab30`

The corresponding `.ttf` files are deterministic outline conversions of
those exact OTF sources, produced with `otf2ttf` 0.2.3 using its default
quadratic-curve tolerance.  Matplotlib uses the TTF copies because its native
PDF backend embeds them as searchable, subsetted CID TrueType fonts; forcing
the CFF-flavoured OTF files through that backend produces invalid Type 42
encoding.  The converted-file SHA-256 values are:

- `LinBiolinum_R.ttf`: `24c8d2306da9d9b14b6f747bf03bb0c66ebb63149f0ee2b90610a3637e27c985`
- `LinBiolinum_RB.ttf`: `4660f7da6e5b7f63f9d9894bfd800c7bac0e54dd89ed689087c7448492ce7e4e`
- `LinBiolinum_RI.ttf`: `e8e61b95d6a748cde30718fdbf102f8c5a2c168103ec3c07aaf053edf0124e86`

The accompanying `README`, `LICENCE.txt`, and `GPL.txt` are copied from the
same archive.  CTAN lists the package under the LPPL, GPL with font exception,
and SIL Open Font License.  These files are used only to render and embed
subset fonts in the paper's vector figure PDFs.  The OTF originals remain the
canonical source; the TTF copies are rendering derivatives, not substituted
typefaces.
