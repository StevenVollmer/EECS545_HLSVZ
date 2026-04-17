from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import preview_brief
from app.models.partner import Partner


def main() -> None:
    partners = [Partner("north", paused=False), Partner("west", paused=True)]
    print(preview_brief(partners))


if __name__ == "__main__":
    main()

