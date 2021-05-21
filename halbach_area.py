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


class Parameter:
    """Base class for a parameter defining how to build our magnet."""
    def __init__(self, switch, label, essential=False):
        self.switch = switch
        self.label = label
        self.essential = essential
        self.type = self.control = None
        self.hbox = QtWidgets.QHBoxLayout()

    def build(self, **kwargs):
        self.hbox.addWidget(QtWidgets.QLabel(self.label))


class NumericParameter(Parameter):
    """Parameter with a numerical value."""
    def __init__(self, switch, label, default, units, step=1, essential=False, **kwargs):
        super().__init__(switch, label, essential)
        self.default = default
        self.units = units
        self.step = step

    def build(self, change_function, **kwargs):
        """Add control to GUI."""
        super().build(change_function=change_function, **kwargs)
        self.add_spin_box(change_function)
        return self.hbox

    def add_spin_box(self, change_function):
        digits = 1 if self.units == 'mm' else 2
        self.control = QtWidgets.QDoubleSpinBox(decimals=digits) if self.units else QtWidgets.QSpinBox()
        self.control.setRange(0, 10000)
        self.control.setSingleStep(self.step)
        self.control.setSuffix(f' {self.units}' if self.units else '')
        self.control.setValue(self.default)
        self.control.valueChanged.connect(change_function)
        self.hbox.addWidget(self.control)

    def get_arg(self):
        value = self.control.value()
        return_value = value * (0.001 if self.units == 'mm' else 1)
        return f'{self.switch}={return_value:g}' if (self.essential or value != self.default) else ''

    def set_from_args(self, args):
        convert = float if self.units else int
        value = convert(args[self.switch])
        self.control.setValue(value * (1000 if self.units == 'mm' else 1))


class OnOffParameter(Parameter):
    """Checkbox on/off parameter."""

    def __init__(self, switch, label, checked=False, **kwargs):
        super().__init__(switch, label, **kwargs)
        self.checkbox = QtWidgets.QCheckBox(self.label)
        self.checkbox.setChecked(checked)

    def build(self, change_function):
        """Add control to GUI."""
        self.hbox.addWidget(self.checkbox)
        self.checkbox.clicked.connect(change_function)
        return self.hbox

    def get_arg(self):
        return f'{self.switch}=1' if self.checkbox.isChecked() else ''

    def set_from_args(self, args):
        self.checkbox.setChecked(self.switch in args)


class OptionalNumericParameter(NumericParameter, OnOffParameter):
    """Optional parameter with a numerical value."""
    def __init__(self, switch, label, default, units, step=1, checked=False, **kwargs):
        super().__init__(switch, label, default, units, checked=checked, step=step, essential=False, **kwargs)
        self.checkbox.setChecked(checked)  # OOP doesn't get passed the 'checked' parameter...

    def build(self, change_function, **kwargs):
        """Add control to GUI."""
        super().build(change_function=change_function, **kwargs)  # checkbox, then spinbox
        self.control.valueChanged.connect(partial(self.checkbox.setChecked, True))
        return self.hbox

    def get_arg(self):
        value = self.control.value()
        return_value = value * (0.001 if self.units == 'mm' else 1)
        return f'{self.switch}={return_value:g}' if self.checkbox.isChecked() else ''

    def set_from_args(self, args):
        try:
            super().set_from_args()
        except KeyError:
            pass


class ChoiceParameter(Parameter):
    """Parameter with fixed choices."""
    VALUES = (1, 2, 4)

    def __init__(self, switch, label, choices, essential=False):
        super().__init__(switch, label, essential)
        self.choices = choices

    def build(self, change_function):
        super().build()
        self.control = QtWidgets.QComboBox()
        [self.control.addItem(i) for i in self.choices]
        self.control.setCurrentIndex(1)  # default is always middle option
        self.control.currentIndexChanged.connect(change_function)
        self.hbox.addWidget(self.control)
        return self.hbox

    def get_arg(self):
        value = self.VALUES[self.control.currentIndex()]
        return f'{self.switch}={value}' if (self.essential or value != 2) else ''

    def set_from_args(self, args):
        self.control.setCurrentIndex(self.VALUES.index(int(args[self.switch])))


class App(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.title = app_name
        self.auto_checkbox = self.layout = self.controls = self.plot = None
        self.mid_remove = self.remove_adjacent = self.midplane_height = None
        self.listview = None
        try:
            self.state = pickle.load(bz2.open(state_file, 'rb'))
        except:
            self.state = {'results': {}, 'icons': {}}
        self.status_bar = self.run_button = self.delete_button = self.progress_bar = None
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
                checkbox.clicked.connect(self.multipoles_checked)
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

        self.midplane_height = NumericParameter('ymidplane', 'Midplane half-height', 0, 'mm')
        self.mid_remove = ChoiceParameter('midplanes', 'Remove midplanes', ['Left only', 'Horizontal', 'Cross'])
        self.remove_adjacent = OnOffParameter('removeadjacent', 'Remove wedges by midplane')
        self.controls = [
            NumericParameter('R', 'Radius', 5, 'mm', essential=True),
            OptionalNumericParameter('gfr', 'Good field region', 3, 'mm'),
            OptionalNumericParameter('dipole', 'Dipole', 0.2, 'T', step=0.1, checked=True),
            OptionalNumericParameter('quad', 'Quadrupole', 300, 'T/m', step=10),
            OptionalNumericParameter('sext', 'Sextupole', 60, 'T/m²', step=10),
            OptionalNumericParameter('oct', 'Octupole', 900, 'T/m³', step=10),
            NumericParameter('Br', 'Remanent field', 1.07, 'T', step=0.1),
            NumericParameter('wedges', 'Wedges', 16, '', step=4),
            ChoiceParameter('symmetry', 'Symmetry', ['None', 'Top/bottom', 'Quad']),
            OnOffParameter('halfoff', 'Offset by half-width'),
            self.midplane_height, self.mid_remove, self.remove_adjacent]

        [self.layout.addLayout(control.build(self.autorun)) for control in self.controls]

        hbox = QtWidgets.QHBoxLayout()
        self.auto_checkbox = QtWidgets.QCheckBox('Auto')
        self.auto_checkbox.setChecked(True)
        hbox.addWidget(self.auto_checkbox)
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.clicked.connect(self.run_simulation)
        self.delete_button = QtWidgets.QPushButton('Delete')
        self.delete_button.clicked.connect(self.delete_case)
        hbox.addWidget(self.run_button)
        hbox.addWidget(self.delete_button)
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
            convert = float if isinstance(control, QtWidgets.QDoubleSpinBox) else int
            if isinstance(control, QtWidgets.QComboBox):  # dropdown, options are always 1, 2, 4
                control.setCurrentIndex((1, 2, 4).index(int(params_dict[switch])))
            elif isinstance(checkbox, QtWidgets.QCheckBox):
                if switch in params_dict.keys():
                    checkbox.setChecked(True)
                    if control is not None:
                        value = convert(params_dict[switch])
                        control.setValue(value * (1000 if control.suffix() == ' mm' else 1))
                else:
                    checkbox.setChecked(False)
            else:
                try:
                    value = convert(params_dict[switch])
                except KeyError:
                    value = 0
                control.setValue(value * (1000 if control.suffix() == ' mm' else 1))
                if switch == 'ymidplane' and value == 0:
                    break  # don't bother with the last two

        self.auto_checkbox.setChecked(was_checked)
        self.run_simulation()

    def check_midplane_controls(self):
        """Grey out midplane controls when midplane gap is set to zero."""
        value = self.midplane_height.control.value()
        self.mid_remove.control.setEnabled(value > 0)
        self.remove_adjacent.checkbox.setEnabled(value > 0)

    def autorun(self):
        """Run a new simulation when something changes."""
        self.check_midplane_controls()
        if self.multipoles_checked() and self.auto_checkbox.isChecked() and self.run_button.isEnabled():
            self.run_simulation()

    def multipoles_checked(self):
        """Check that at least one multipole is checked."""
        try:
            next(ctrl for ctrl in self.controls if ctrl.label.endswith('pole') and ctrl.get_arg())
            self.status_bar.setText('Ready.')
            self.run_button.setEnabled(True)
            return True
        except StopIteration:  # none checked
            self.status_bar.setText('Select at least one multipole.')
            self.run_button.setEnabled(False)
            return False

    def run_simulation(self):
        """Run the simulation using specified parameters."""
        args = []
        colour = [0, 0, 0]
        # add command-line arguments
        args = list(filter(None, [control.get_arg() for control in self.controls]))
        print(args)
        print(', '.join(args))
        colour = [(0.8 if control.get_arg() else 0) for control in self.controls if control.label.endswith('pole')]
        colour[2] += (0.2 if colour[3] else 0)  # fudge to combine octupole + sextupole
        colour.pop(3)  # remove last

        args = tuple(args)  # make it hashable so it can be a dict key
        print(args)
        if args not in self.state['results'].keys():
            print('Running')
            self.status_bar.setText('Running...')
            self.progress_bar.setValue(0)
            self.run_button.setEnabled(False)
            QtCore.QCoreApplication.processEvents()
            print(cmd + list(args))
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
                print(stdout, end='')
                failed |= 'ERROR' in stdout
                QtCore.QCoreApplication.processEvents()  # update the GUI
            print('')
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
            print('using saved result')
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

    def delete_case(self):
        for item in self.listview.selectedItems():
            params = tuple(item.text().split(', '))
            try:
                self.state['results'].pop(params)
                self.state['icons'].pop(params)
            except KeyError:
                pass
            pickle.dump(self.state, bz2.open(state_file, 'wb'))
            self.listview.takeItem(self.listview.row(item))

    def store_results(self, args, result):
        self.state['results'][args] = result
        print(self.state['results'].keys())
        self.listview.addItem(', '.join(args))
        item = self.listview.item(self.listview.count() - 1)
        item.setSelected(True)
        pickle.dump(self.state, bz2.open(state_file, 'wb'))
        return item


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
