# DDT Extended

Destiny Damage Tracker Extended is a damage scanner utility for Destiny 2. DDT-ext tracks the health of a boss via the health bar, allowing for more accurate health readings.

Inspired by [A2TC-YT/Destiny-Damage-Tracker](https://github.com/A2TC-YT/Destiny-Damage-Tracker).

## Platforms Supported
* Windows 11

## Dependencies
* Python 3.12

## Usage

1. Clone the repository.

```powershell
git clone https://github.com/ColonialDagger/ddt-extended.git
```

2. Create a virtual environment.

```powershell
py -3.12 -m venv .venv
```

3. Activate the virtual environment.

```powershell
.\.venv\Scripts\activate
```

4. Build the program.

```powershell
python -m nuitka .\scanner.py --standalone --mingw64 --include-data-dir="negatives=negatives" --include-data-file="utils.py=utils.py" --include-data-file="README.md=README.md" --include-data-file="LICENSE=LICENSE"
```
