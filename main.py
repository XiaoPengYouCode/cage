from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
HELIX_VORONOI_SRC = PROJECT_ROOT / "packages" / "helix-voronoi" / "src"
if str(HELIX_VORONOI_SRC) not in sys.path:
    sys.path.insert(0, str(HELIX_VORONOI_SRC))

from helix_voronoi.cli import main


if __name__ == "__main__":
    main()
