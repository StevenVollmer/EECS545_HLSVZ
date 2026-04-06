from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.exports.renewal_export import render_renewal_export


def main() -> None:
    print(render_renewal_export("mcintyre ross", "zx-14"))


if __name__ == "__main__":
    main()
