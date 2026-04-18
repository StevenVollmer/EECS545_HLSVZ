from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import transfer_digest_preview


def main() -> None:
    print(transfer_digest_preview("o'neil-ward", 0.008))


if __name__ == '__main__':
    main()
