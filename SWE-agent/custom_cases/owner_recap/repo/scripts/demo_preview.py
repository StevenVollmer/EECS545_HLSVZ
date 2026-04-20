from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import preview_owner_recap


def main() -> None:
    print(preview_owner_recap("mcallister-smith", 4))


if __name__ == "__main__":
    main()
