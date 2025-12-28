# DDT Extended

Destiny Damage Tracker Extended is a damage scanner utility for Destiny 2. DDT-ext tracks the health of a boss via the health bar, allowing for more accurate health readings.

Inspired by [A2TC-YT/Destiny-Damage-Tracker](https://github.com/A2TC-YT/Destiny-Damage-Tracker).

## Platforms Supported
* Windows 11

## Build Dependencies
* Python 3.12

## Building

1. Clone the repository.

```powershell
git clone https://github.com/ColonialDagger/ddt-extended.git
cd ddt-extended
```

2. Create a virtual environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

4. Install dependencies.

```powershell
pip install -r requirements.txt
```

4. Build the program.
   
```powershell
python -m nuitka .\ddt.py `
  --onefile `
  --mingw64 `
  --enable-plugin=numpy `
  --enable-plugin=pyside6 `
  --include-data-dir="negatives=negatives" `
  --include-data-file="README.md=README.md" `
  --include-data-file="LICENSE=LICENSE" `
  --include-onefile-external-data=negatives `
  --include-onefile-external-data=README.md `
  --include-onefile-external-data=LICENSE `
  --output-dir="builds/w11-onefile" `
  --windows-console-mode=disable
```

## Known Issues
* When opening the scanner, the derivative graph gridlines are not actualized until the first update.