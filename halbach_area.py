import sys
import numpy as np
import os
import re
import subprocess
import pickle
import bz2
from PyQt5 import QtWidgets, QtCore, QtGui
from functools import partial
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

app_name = 'HalbachArea'
halbach_exe = f"{app_name}.exe"
state_file = f'{app_name}.db'
# For some reason the output from halbach_exe is buffered so we can't read it until the program completes.
# I found a workaround but it requires the use of winpty: https://github.com/rprichard/winpty/releases
# If winpty is not available it will just use halbach_exe and wait until the program exits.
# https://stackoverflow.com/questions/11516258/what-is-the-equivalent-of-unbuffer-program-on-windows
winpty = 'winpty.exe'
cmd = [winpty, '-Xallow-non-tty', '-Xplain', halbach_exe] if os.path.exists(winpty) else [halbach_exe, ]


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class App(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.title = app_name
        self.auto_checkbox = self.mid_wedges = self.mid_symmetry = self.layout = self.controls = self.plot = None
        self.listview = None
        try:
            self.state = pickle.load(bz2.open(state_file, 'rb'))
        except:
            self.state = {'results': {}, 'icons': {}}
        self.status_bar = self.run_button = self.progress_bar = None
        self.init_ui()

    def appendControl(self, name, default, switch, units=None, checked=None, digits=2, step=1):
        """Add a control.
        Set checked to True or False for a checkbox, otherwise will be a label.
        Set step to a list to make a dropdown."""
        hbox = QtWidgets.QHBoxLayout()
        if step is None:  # checkbox only
            control = None
        elif isinstance(step, list):  # dropdown box
            control = QtWidgets.QComboBox()
            [control.addItem(i) for i in step]
            control.setCurrentIndex(1)
            change_event = control.currentIndexChanged
        else:
            control = QtWidgets.QDoubleSpinBox(decimals=digits) if digits > 0 else QtWidgets.QSpinBox()
            control.setMaximum(2**31-1)
            control.setMinimum(0)
            control.setSingleStep(step)
            control.setSuffix(f' {units}' if units else '')
            control.setValue(default)
            change_event = control.valueChanged
        if checked is None:  # non-optional control
            checkbox = QtWidgets.QLabel(name)
            change_event.connect(self.autorun)
        else:  # optional so include a checkbox
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.setChecked(checked)
            if control is not None:
                control.valueChanged.connect(partial(self.spinbox_changed, checkbox))
                checkbox.clicked.connect(self.checkbox_changed)
            else:
                checkbox.clicked.connect(self.autorun)
        hbox.addWidget(checkbox)
        hbox.addWidget(control)
        self.layout.addLayout(hbox)
        self.controls[switch] = (control, checkbox if checked in (True, False) else None)
        return checkbox if control is None else control

    def init_ui(self):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        vert_layout = QtWidgets.QVBoxLayout()
        horiz_layout = QtWidgets.QHBoxLayout()
        vert_layout.addLayout(horiz_layout)
        self.layout = QtWidgets.QVBoxLayout()
        self.listview = QtWidgets.QListWidget()
        self.listview.addItems([', '.join(params) for params in self.state['results'].keys()])
        self.listview.itemClicked.connect(self.listview_item_clicked)
        horiz_layout.addWidget(self.listview)
        horiz_layout.addLayout(self.layout)
        self.controls = {}

        self.appendControl('Radius', 5, 'R', 'mm', digits=1)
        self.appendControl('Good field region', 3, 'gfr', 'mm', False, digits=1)
        self.appendControl('Dipole', 0.2, 'dipole', 'T', False, step=0.1)
        self.appendControl('Quadrupole', 300, 'quad', 'T/m', True, step=10)
        self.appendControl('Sextupole', 66, 'sext', 'T/m²', False, step=10)
        self.appendControl('Octupole', 888, 'oct', 'T/m³', False, step=10)
        self.appendControl('Remanent field', 1.07, 'Br', 'T', step=0.1)
        self.appendControl('Wedges', 16, 'wedges', digits=0, step=4)
        self.appendControl('Symmetry', '2', 'symmetry', step=['None', 'Top/bottom', 'Quad'])
        self.appendControl('Offset by half-width', False, 'halfoff', checked=False, step=None)
        mhh_dropdown = self.appendControl('Midplane half-height', 0, 'ymidplane', 'mm', digits=2)
        mhh_dropdown.valueChanged.connect(self.check_midplane_controls)
        self.mid_symmetry = self.appendControl('Remove midplanes', '2', 'midplanes',
                                               step=['Left only', 'Horizontal', 'Cross'])
        self.mid_wedges = self.appendControl('Remove wedges by midplane', False, 'removeadjacent', checked=False,
                                             step=None)
        self.check_midplane_controls(0)

        hbox = QtWidgets.QHBoxLayout()
        self.auto_checkbox = QtWidgets.QCheckBox('Auto')
        self.auto_checkbox.setChecked(True)
        hbox.addWidget(self.auto_checkbox)
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.clicked.connect(self.run_simulation)
        hbox.addWidget(self.run_button)
        self.layout.addLayout(hbox)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximum(25)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)
        self.status_bar = QtWidgets.QLabel()
        vert_layout.addWidget(self.status_bar)

        self.plot = MplCanvas(self, width=5, height=4, dpi=100)
        NavigationToolbar(self.plot, self.plot)
        horiz_layout.addWidget(self.plot)

        self.setLayout(vert_layout)
        self.show()
        self.autorun()

    def listview_item_clicked(self, item):
        """Set controls for previously-selected parameters from the list box."""
        was_checked = self.auto_checkbox.isChecked()
        self.auto_checkbox.setChecked(False)  # prevent automatically running models while controls are changed
        params = tuple(item.text().split(', '))
        params_dict = dict([param.split('=') for param in params])
        for switch, (control, checkbox) in self.controls.items():
            if isinstance(control, QtWidgets.QComboBox):  # dropdown, options are always 1, 2, 4
                control.setCurrentIndex((1, 2, 4).index(int(params_dict[switch])))
            elif control is None:  # checkbox only
                checkbox.setChecked(switch in params_dict.keys())
            else:  # spinbox
                try:
                    convert = float if isinstance(control, QtWidgets.QDoubleSpinBox) else int
                    checkbox.setChecked(switch in params_dict.keys())
                    value = convert(params_dict[switch])
                    control.setValue(value * (1000 if control.suffix() == ' mm' else 1))
                except (KeyError, AttributeError):  # no checkbox or param not specified
                    if switch == 'ymidplane':
                        break  # don't bother with the last two
        self.auto_checkbox.setChecked(was_checked)
        self.run_simulation()

    def check_midplane_controls(self, value):
        """Grey out midplane controls when midplane gap is set to zero."""
        self.mid_wedges.setEnabled(value > 0)
        self.mid_symmetry.setEnabled(value > 0)

    def spinbox_changed(self, checkbox):
        """Check the relevant box when one of the spinboxes is changed."""
        checkbox.setChecked(True)
        self.autorun()

    def autorun(self):
        """Run a new simulation when something changes."""
        if self.auto_checkbox.isChecked() and self.run_button.isEnabled():
            self.run_simulation()

    def checkbox_changed(self):
        """Check that at least one multipole is checked."""
        n_checked = 0
        for switch, (control, checkbox) in self.controls.items():
            if checkbox is not None and checkbox.text().endswith('pole') and checkbox.isChecked():
                n_checked += 1
        if n_checked == 0:
            self.status_bar.setText('Select at least one multipole.')
            self.run_button.setEnabled(False)
        else:
            self.status_bar.setText('Ready.')
            self.run_button.setEnabled(True)
            self.autorun()

    def run_simulation(self):
        """Run the simulation using specified parameters."""
        args = []
        colour = [0, 0, 0]
        # add command-line arguments
        for switch, (control, checkbox) in self.controls.items():
            if checkbox is None or checkbox.isChecked():
                if control is None:  # just a checkbox
                    value = 1
                elif isinstance(control, QtWidgets.QComboBox):  # dropdown, options are always 1, 2, 4
                    value = (1, 2, 4)[control.currentIndex()]
                else:  # spinbox
                    value = control.value() * (0.001 if control.suffix() == ' mm' else 1)
                    try:
                        colour[('dipole', 'sext', 'quad').index(switch)] += 1  # dipoles are red, quadrupoles are blue
                    except ValueError:
                        pass
                if switch == 'ymidplane' and value == 0:
                    break  # don't bother with the last two
                args.append(f'{switch}={value:g}')  # g here removes trailing zeros
        args = tuple(args)  # make it hashable so it can be a dict key
        if args not in self.state['results'].keys():
            self.status_bar.setText('Running...')
            self.progress_bar.setValue(0)
            self.run_button.setEnabled(False)
            QtCore.QCoreApplication.processEvents()
            process = subprocess.Popen(cmd + list(args), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, universal_newlines=True)
            status_line = re.compile(r'Iteration (\d+)')
            failed = False
            while process.poll() is None:
                stdout = process.stdout.readline()
                try:
                    self.progress_bar.setValue(int(status_line.match(stdout).group(1)))
                except AttributeError:  # not a line with "Iteration X" in it
                    pass
                self.status_bar.setText(stdout)
                # print(stdout, end='')
                failed |= 'ERROR' in stdout
                QtCore.QCoreApplication.processEvents()  # update the GUI
            # print('')
            self.run_button.setEnabled(True)
            if failed:
                print(process.returncode)
                self.status_bar.setText('Failed.')
                self.progress_bar.setValue(0)
                self.store_results(args, None)
                return
            self.status_bar.setText('Done.')
            self.progress_bar.setValue(self.progress_bar.maximum())

            # get the magnet shapes from the output CSV file
            # format is Bx, By, x0, y0, x1, y1, x2, y2, x3, y3
            # Turn into dimensions in mm (and remanent fields in kT, but that doesn't matter)
            polygons = np.loadtxt("magnet.csv", skiprows=1, usecols=range(1, 11), delimiter=',') * 1000
            item = self.store_results(args, polygons)
        else:  # use saved result
            args_str = ', '.join(args)
            for i in range(self.listview.count()):
                item = self.listview.item(i)
                if item.text() == args_str:
                    item.setSelected(True)
                    break
            result = self.state['results'][args]
            if result is None:
                self.status_bar.setText('Failed.')
                return
            polygons = result

        min_width = min(np.ptp(polygons[:, 2::2], axis=1))
        min_height = min(np.ptp(polygons[:, 3::2], axis=1))
        max_b = max(polygons[:, :2].flat)
        scale = min(min_width, min_height) / max_b  # how to represent Bx/By vectors on graph?
        self.plot.axes.cla()  # clear axes
        shapes = [Polygon(polygon[2:10].reshape(4, 2), facecolor=colour, edgecolor='black') for polygon in polygons]
        self.plot.axes.add_collection(PatchCollection(shapes, match_original=True))
        # add remanent field vectors
        for polygon in polygons:
            bx, by = polygon[:2] * scale
            self.plot.axes.arrow(np.mean(polygon[2::2]) - bx / 2, np.mean(polygon[3::2]) - by / 2,
                                 bx, by, width=0.1 * np.sqrt(bx ** 2 + by ** 2), length_includes_head=True)
        self.plot.axes.set_xlim([min(polygons[:, 2::2].flat), max(polygons[:, 2::2].flat)])
        self.plot.axes.set_ylim([min(polygons[:, 3::2].flat), max(polygons[:, 3::2].flat)])
        self.plot.draw()

        try:
            img_data = self.state['icons'][args]
        except KeyError:
            # capture plot image and resize to icon size (around 32 pixels)
            width, height = self.plot.get_width_height()
            step = min(height, width) // 32  # how many rows/cols to skip along
            img_data = np.frombuffer(self.plot.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
            img_data = img_data[::step, ::step, :]
            img_data = img_data.copy()
            self.state['icons'][args] = img_data
            pickle.dump(self.state, bz2.open(state_file, 'wb'))
        height, width, bpp = img_data.shape
        image = QtGui.QImage(img_data, width, height, width * bpp, QtGui.QImage.Format_RGB888)
        icon = QtGui.QIcon(QtGui.QPixmap(image))
        item.setIcon(icon)

    def store_results(self, args, result):
        self.state['results'][args] = result
        self.listview.addItem(', '.join(args))
        item = self.listview.item(self.listview.count() - 1)
        item.setSelected(True)
        pickle.dump(self.state, bz2.open(state_file, 'wb'))
        return item


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
