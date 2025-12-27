# DDT Extended

Destiny Damage Tracker Extended is a damage scanner utility for Destiny 2. DDT-ext tracks the health of a boss via the health bar, allowing for more accurate health readings.

Inspired by [A2TC-YT/Destiny-Damage-Tracker](https://github.com/A2TC-YT/Destiny-Damage-Tracker).

## Platforms Supported
* Windows 11

## Dependencies
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
python -m nuitka .\gui.py `
  --onefile `
  --mingw64 `
  --enable-plugin=numpy `
  --enable-plugin=tk-inter `
  --include-data-dir="negatives=negatives" `
  --include-data-file="utils.py=utils.py" `
  --include-data-file="README.md=README.md" `
  --include-data-file="LICENSE=LICENSE" `
  --output-dir="builds/onefile" `
  --windows-console-mode=disable
```

## Known Issues
* Only brightness 4 works.
* Delta_T sometimes reports 0.0ms.
