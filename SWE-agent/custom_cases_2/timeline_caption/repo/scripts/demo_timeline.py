from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import render_timeline_caption


def main() -> None:
    print(render_timeline_caption("operations/o'neil-ward", 3))


if __name__ == "__main__":
    main()
