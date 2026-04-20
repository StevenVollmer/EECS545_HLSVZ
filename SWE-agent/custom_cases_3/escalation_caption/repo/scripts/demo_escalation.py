from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import escalation_caption


def main() -> None:
    print(escalation_caption("o'neil-ward", "ops", 3))


if __name__ == '__main__':
    main()
