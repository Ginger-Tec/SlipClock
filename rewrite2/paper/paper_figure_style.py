"""Shared publication style for SlipClock paper figures.

Figures are drawn at their final ACM physical size.  Linux Biolinum O is the
sans-serif companion selected by ``acmart`` and is registered from the pinned
CTAN files in ``fonts/libertine`` so rendering does not depend on system fonts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


COLUMN_WIDTH_IN = 3.32
TEXT_WIDTH_IN = 6.90
FONT_ROOT = Path(__file__).resolve().parent / "fonts"
FONT_DIR = FONT_ROOT / "libertine"
# Matplotlib's PDF backend cannot embed the CFF outlines in the upstream OTF
# files as CID Type 42 fonts.  These TTF files are deterministic quadratic-
# outline conversions of the pinned OTF originals; see fonts/libertine/SOURCE.md.
FONT_FILES = (
    FONT_DIR / "LinBiolinum_R.ttf",
    FONT_DIR / "LinBiolinum_RB.ttf",
    FONT_DIR / "LinBiolinum_RI.ttf",
)
FONT_FAMILY = "Linux Biolinum O"
KOREAN_FONT_FILES = (
    FONT_ROOT / "noto_kr" / "SlipClockNotoSansKR-Regular.ttf",
    FONT_ROOT / "noto_kr" / "SlipClockNotoSansKR-Bold.ttf",
)
KOREAN_FONT_FAMILY = "SlipClock Noto Sans KR"

INK = "#202124"
MUTED = "#5F6368"
GRID = "#DADCE0"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GOLD = "#C58A00"
LIGHT_GREY = "#B8BDC3"

UNIVERSE_COLORS = {"top25": BLUE, "top100": VERMILLION, "top200": GREEN}
UNIVERSE_LABELS = {"top25": "Top-25", "top100": "Top-100", "top200": "Top-200"}
STRATEGY_COLORS = {
    "ts_momentum": BLUE,
    "cs_momentum": VERMILLION,
    "mean_reversion": GREEN,
    "vol_scaled": PURPLE,
    "mlp": GOLD,
}

PDF_METADATA = {
    "Creator": "SlipClock",
    "CreationDate": datetime(2026, 8, 6, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 8, 6, tzinfo=timezone.utc),
}


def apply_paper_style() -> None:
    """Apply one typography and line system to every paper-facing figure."""
    registered_fonts = FONT_FILES + KOREAN_FONT_FILES
    missing = [path for path in registered_fonts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pinned Linux Biolinum fonts: {missing}")
    for path in registered_fonts:
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, KOREAN_FONT_FAMILY],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.9,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.2,
            "grid.color": GRID,
            "grid.linewidth": 0.45,
            "grid.alpha": 1.0,
            "text.color": INK,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
