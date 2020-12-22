import sys
import numpy as np
import random
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
from functools import partial
from time import sleep
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import subprocess

halbach_exe = r"HalbachArea.exe"


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class App(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'HalbachArea'
        self.layout = self.controls = self.plot = None
        self.initUI()

    def appendControl(self, name, default, switch, units=None, checked=None, digits=2, step=1):
        """Add a control.
        Set checked to True or False for a checkbox, otherwise will be a label.
        Set step to a list to make a dropdown."""
        hbox = QtWidgets.QHBoxLayout()
        if step is None:
            spinbox = None
        elif isinstance(step, list):
            spinbox = QtWidgets.QComboBox()
            [spinbox.addItem(i) for i in step]
            spinbox.setCurrentIndex(1)
        else:
            spinbox = QtWidgets.QDoubleSpinBox(decimals=digits) if digits > 0 else QtWidgets.QSpinBox()
            spinbox.setMaximum(2147483647)
            spinbox.setMinimum(0)
            spinbox.setSingleStep(step)
            spinbox.setSuffix(f' {units}' if units else '')
            spinbox.setValue(default)
        if checked is None:
            checkbox = QtWidgets.QLabel(name)
        else:
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.setChecked(checked)
            if spinbox is not None:
                spinbox.valueChanged.connect(partial(self.spinbox_changed, checkbox))
                checkbox.clicked.connect(self.checkbox_changed)
        hbox.addWidget(checkbox)
        hbox.addWidget(spinbox)
        self.layout.addLayout(hbox)
        self.controls.append((switch, spinbox, checkbox if checked in (True, False) else None))
        return spinbox

    def initUI(self):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        vert_layout = QtWidgets.QVBoxLayout()
        horiz_layout = QtWidgets.QHBoxLayout()
        vert_layout.addLayout(horiz_layout)
        self.layout = QtWidgets.QVBoxLayout()
        horiz_layout.addLayout(self.layout)
        self.controls = []

        self.appendControl('Radius', 5, 'R', 'mm', digits=1)
        self.appendControl('Good field region', 3, 'gfr', 'mm', False, digits=1)
        self.appendControl('Dipole', 0.2, 'dipole', 'T', False, step=0.1)
        self.appendControl('Quadrupole', 300, 'quad', 'T/m', True, step=10)
        self.appendControl('Sextupole', 66, 'sext', 'T/m²', False)
        self.appendControl('Octupole', 888, 'oct', 'T/m³', False, step=10)
        self.appendControl('Remanent field', 1.07, 'Br', 'T', step=0.1)
        self.appendControl('Wedges', 16, 'wedges', digits=0, step=4)
        self.appendControl('Symmetry', '2', 'symmetry', step=['None', 'Top/bottom', 'Quad'])
        self.appendControl('Offset by half-width', False, 'halfoff', checked=False, step=None)
        self.appendControl('Midplane half-height', 0, 'ymidplane', 'mm', digits=2)
        self.appendControl('Remove midplanes', '2', 'midplanes', step=['Left only', 'Horizontal', 'Cross'])
        self.appendControl('Remove wedges by midplane', False, 'removeadjacent', checked=False, step=None)

        hbox = QtWidgets.QHBoxLayout()
        self.checkbox = QtWidgets.QCheckBox('Auto')
        # self.checkbox.clicked.connect(self.run_simulation)
        hbox.addWidget(self.checkbox)
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.clicked.connect(self.run_simulation)
        hbox.addWidget(self.run_button)
        self.layout.addLayout(hbox)
        self.status_bar = QtWidgets.QLabel()
        vert_layout.addWidget(self.status_bar)

        self.plot = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.plot, self)  # TODO: make it appear above the plot
        horiz_layout.addWidget(self.plot)

        self.setLayout(vert_layout)
        self.show()

    def spinbox_changed(self, checkbox):
        checkbox.setChecked(True)

    def checkbox_changed(self):
        n_checked = 0
        for switch, spinbox, checkbox in self.controls:
            if checkbox is not None and checkbox.text().endswith('pole') and checkbox.isChecked():
                n_checked += 1
        if n_checked == 0:
            self.status_bar.setText('Select at least one multipole.')
            self.run_button.setEnabled(False)
        else:
            self.status_bar.setText('Ready.')
            self.run_button.setEnabled(True)

    @QtCore.pyqtSlot()
    def run_simulation(self):
        cmd = [halbach_exe, ]
        for switch, spinbox, checkbox in self.controls:
            if checkbox is None or checkbox.isChecked():
                if spinbox is None:
                    value = 1
                elif isinstance(spinbox, QtWidgets.QComboBox):
                    value = (1, 2, 4)[spinbox.currentIndex()]
                else:
                    value = spinbox.value() * (0.001 if spinbox.suffix() == ' mm' else 1)
                cmd.append(f'{switch}={value}')

        # if self.checkbox.isChecked():
        self.status_bar.setText('Running...')
        self.run_button.setEnabled(False)
        self.status_bar.update()
        print(cmd)
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        while popen.poll() is None:
            sleep(0.5)
        stdout, stderr = popen.communicate()
        self.run_button.setEnabled(True)
        print(stdout)
        if popen.returncode:
            self.status_bar.setText(stdout.splitlines()[-1])
            return
        self.status_bar.setText('Done.')

        polygons = np.loadtxt(r"magnet.csv", skiprows=1, usecols=range(1, 11), delimiter=',')
        min_width = min(np.ptp(polygons[:, 2::2], axis=1))
        min_height = min(np.ptp(polygons[:, 3::2], axis=1))
        max_b = max(polygons[:, :2].flat)
        scale = min(min_width, min_height) / max_b
        print(scale)
        self.plot.axes.cla()
        shapes = [Polygon(polygon[2:10].reshape(4, 2)) for polygon in polygons]
        self.plot.axes.add_collection(PatchCollection(shapes))
        for polygon in polygons:
            bx, by = polygon[:2] * scale
            self.plot.axes.arrow(np.mean(polygon[2::2]) - bx / 2, np.mean(polygon[3::2]) - by / 2,
                                 bx, by, width=0.1 * np.sqrt(bx ** 2 + by ** 2), length_includes_head=True)
        self.plot.axes.set_xlim([min(polygons[:, 2::2].flat), max(polygons[:, 2::2].flat)])
        self.plot.axes.set_ylim([min(polygons[:, 3::2].flat), max(polygons[:, 3::2].flat)])
        self.plot.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
