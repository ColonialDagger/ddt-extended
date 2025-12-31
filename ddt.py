import sys
print("DDT STARTED", file=sys.stderr)

from PySide6.QtWidgets import QApplication

from gui.gui import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("error.log", "w") as f:
            f.write(str(e))
        print(f"An error occurred: {e}")
