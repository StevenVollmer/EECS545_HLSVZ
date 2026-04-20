from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import dispatch_preview


def main() -> None:
    print(dispatch_preview("d'arcy-lee"))


if __name__ == "__main__":
    main()

