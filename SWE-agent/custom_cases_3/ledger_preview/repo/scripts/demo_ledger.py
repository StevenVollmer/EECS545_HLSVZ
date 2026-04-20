from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import ledger_preview


def main() -> None:
    print(ledger_preview(1280.5))


if __name__ == '__main__':
    main()
