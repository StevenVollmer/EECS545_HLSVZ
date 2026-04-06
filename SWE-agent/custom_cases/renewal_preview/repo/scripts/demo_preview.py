from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.presenters.renewal_presenter import render_renewal_preview


def main() -> None:
    print(render_renewal_preview("mcintyre ross", "zx-14"))


if __name__ == "__main__":
    main()
