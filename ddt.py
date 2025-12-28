# ddt.py
import sys
from PySide6.QtWidgets import QApplication
from gui.gui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()   # ❗ No scanner passed in
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
