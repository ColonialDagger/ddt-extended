# ! WARNING !

This is only a proof of concept of GPU accelerated health estimation. Significant portions of the code are AI generated, and need to be heavily reviewed.

Problems include:

* Refusal to run after compiling with Nuitka

---

# DDT Extended

Destiny Damage Tracker Extended is a damage scanner utility for Destiny 2. DDT-ext tracks the health of a boss via the health bar, allowing for more accurate health readings.

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

## License
This project is license under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file and [tl;drLegal](https://www.tldrlegal.com/license/gnu-general-public-license-v3-gpl-3) for details.

## Credits
* Inspired by [A2TC-YT/Destiny-Damage-Tracker](https://github.com/A2TC-YT/Destiny-Damage-Tracker).
* <a href="https://www.flaticon.com/free-icons/heart" title="heart icons">Heart icons created by Good Ware - Flaticon</a>
* `windows-capture` library by [NiiightmareXD/windows-capture](https://github.com/NiiightmareXD/windows-capture/tree/main/windows-capture-python)
* `nuitka` compiler by [Nuitka/Nuitka](https://github.com/Nuitka/Nuitka).
* `pyside6` library by [Qt for Python](https://www.qt.io/qt-for-python).
