from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import export_owner


def main() -> None:
    print(export_owner("mcallister-smith", 4))


if __name__ == "__main__":
    main()
