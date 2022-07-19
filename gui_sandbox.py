# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:34:00 2021

@author: kunal
"""

#%% How to exit properly

try:
    from PyQt5.QtCore import QTimer, pyqtSlot
    from PyQt5.QtGui import QKeySequence, QIcon
    from PyQt5.QtWidgets import QMainWindow, QMessageBox, qApp, QMenu, QSystemTrayIcon
except ImportError:
    from PySide2.QtCore import QTimer, Slot as pyqtSlot
    from PySide2.QtGui import QKeySequence, QIcon
    from PySide2.QtWidgets import QMainWindow, QMessageBox, qApp, QMenu, QSystemTrayIcon


class MainWindow(QMainWindow):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._menu = QMenu()
        self._menu.addAction("&Quit", qApp.quit, QKeySequence.Quit)

        self._trayIcon = QSystemTrayIcon(QIcon("./icon.png"), self)
        self._trayIcon.setContextMenu(self._menu)
        self._trayIcon.show()

        # This defers the call to open the dialog after the main event loop has started
        QTimer.singleShot(0, self.setProfile)

    @pyqtSlot()
    def setProfile(self):
        if QMessageBox.question(self, "Quit?", "Quit?") != QMessageBox.No:
            qApp.quit()
        self.hide()


if __name__ == "__main__":
    from sys import exit, argv

    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        from PySide2.QtWidgets import QApplication

    a = QApplication(argv)
    m = MainWindow()
    m.show()
    exit(a.exec_())