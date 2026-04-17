from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import preview_banner


def main() -> None:
    print(preview_banner(1280.50))


if __name__ == "__main__":
    main()

