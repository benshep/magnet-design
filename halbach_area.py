import sys
import numpy as np
import os
import re
import subprocess
import pickle
import bz2
import io
import requests
from math import factorial
from zipfile import ZipFile  # for decompression
from PyQt5 import QtWidgets, QtCore, QtGui, Qt
from functools import partial
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from operapy import canvas, opera2d

    model = opera2d.get_model_interface()
    graphing = opera2d.get_graphing_interface(model)
    post = opera2d.get_post_processing_interface(model)
except ModuleNotFoundError:
    canvas = opera2d = model = graphing = post = None
try:
    import radia as rad
except ModuleNotFoundError:
    rad = None

app_name = 'HalbachArea'
halbach_exe = f"{app_name}.exe"
if not os.path.exists(halbach_exe):
    compressed_file = io.BytesIO(requests.get('https://stephenbrooks.org/ap/halbacharea/halbacharea.zip').content)
    ZipFile(compressed_file).extractall('.')

state_file = f'{app_name}.db'
# For some reason the output from halbach_exe is buffered so we can't read it until the program completes.
# I found a workaround but it requires the use of winpty: https://github.com/rprichard/winpty/releases
# If winpty is not available it will just use halbach_exe and wait until the program exits.
# https://stackoverflow.com/questions/11516258/what-is-the-equivalent-of-unbuffer-program-on-windows
winpty = 'winpty.exe'
cmd = [winpty, '-Xallow-non-tty', '-Xplain', halbach_exe] if os.path.exists(winpty) else [halbach_exe, ]


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class MplCanvas3d(QtWidgets.QWidget):
    def __init__(self, width=5, height=4, dpi=100):
        super().__init__()
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas = FigureCanvasQTAgg(fig)
        self.axes = fig.add_subplot(111, projection='3d')
        self.axes.set_position([0, 0, 1, 1])

        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(self.canvas, 0, 0)


class Parameter:
    """Base class for a parameter defining how to build our magnet."""
    def __init__(self, switch, label, essential=False):
        self.switch = switch
        self.label = label
        self.essential = essential
        self.type = self.control = None
        self.hbox = QtWidgets.QHBoxLayout()

    def build(self, **kwargs):
        """Add the control(s) to the GUI."""
        self.hbox.addWidget(QtWidgets.QLabel(self.label))

    def set_from_args(self, args):
        """Set the control value based on a list of arguments to the HalbachArea program."""
        pass


class NumericParameter(Parameter):
    """Parameter with a numerical value, represented by a spin box."""
    def __init__(self, switch, label, default, units, step=1, essential=False, **kwargs):
        super().__init__(switch, label, essential)
        self.default = default
        self.units = units
        self.step = step

    def build(self, change_function, **kwargs):
        super().build(change_function=change_function, **kwargs)
        self.add_spin_box(change_function)
        return self.hbox

    def add_spin_box(self, change_function):
        """Add a spin box to the GUI."""
        digits = 1 if self.units == 'mm' else 2
        self.control = QtWidgets.QDoubleSpinBox(decimals=digits) if self.units else QtWidgets.QSpinBox()
        self.control.setRange(0, 10000)
        self.control.setSingleStep(self.step)
        self.control.setSuffix(f' {self.units}' if self.units else '')
        self.control.setValue(self.default)
        self.control.valueChanged.connect(change_function)
        self.hbox.addWidget(self.control)

    def get_arg(self):
        """Convert the current value shown in the control to an argument to pass to HalbachArea."""
        value = self.control.value()
        return_value = value * (0.001 if self.units == 'mm' else 1)
        return f'{self.switch}={return_value:g}' if (self.essential or value != self.default) else ''

    def set_from_args(self, args):
        convert = float if self.units else int
        value = convert(args[self.switch]) if self.switch in args else self.default
        self.control.setValue(value * (1000 if self.units == 'mm' else 1))
        super().set_from_args(args)


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
        super().set_from_args(args)


class OptionalNumericParameter(NumericParameter, OnOffParameter):
    """Optional parameter with a numerical value, represented by a checkbox and a spin box."""
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
        super().set_from_args(args)


class ChoiceParameter(Parameter):
    """Parameter with fixed choices, represented by a dropdown box."""
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
        self.control.setCurrentIndex(self.VALUES.index(int(args[self.switch])) if self.switch in args else 1)
        super().set_from_args(args)


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
            self.state = {'results': {}, 'icons': {}, 'harmonics': {}}
        self.status_bar = self.run_button = self.delete_button = self.progress_bar = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        vert_layout = QtWidgets.QVBoxLayout()
        horiz_layout = QtWidgets.QHBoxLayout()
        vert_layout.addLayout(horiz_layout)
        self.layout = QtWidgets.QVBoxLayout()
        self.listview = QtWidgets.QListWidget()
        self.listview.addItems([', '.join(params) for params in self.state['results'].keys()])
        for i in range(self.listview.count()):
            item = self.listview.item(i)
            try:
                img_data = self.state['icons'][tuple(item.text().split(', '))]
                height, width, bpp = img_data.shape
                image = QtGui.QImage(img_data, width, height, width * bpp, QtGui.QImage.Format_RGB888)
                icon = QtGui.QIcon(QtGui.QPixmap(image))
                item.setIcon(icon)
            except KeyError:  # no icon
                pass

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
        self.run_button = QtWidgets.QToolButton()
        self.run_button.setText('▶ Run')
        self.run_button.clicked.connect(self.run_simulation)
        self.delete_button = QtWidgets.QToolButton()
        self.delete_button.setText('🗑️ Delete')
        self.delete_button.clicked.connect(self.delete_case)
        hbox.addWidget(self.run_button)
        hbox.addWidget(self.delete_button)
        self.layout.addLayout(hbox)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Build model with length'))
        self.length_spinbox = QtWidgets.QDoubleSpinBox()
        self.length_spinbox.setSuffix(' mm')
        self.length_spinbox.setValue(50)
        self.length_spinbox.setMinimum(1)
        self.length_spinbox.setMaximum(1000)
        self.length_spinbox.setSingleStep(5)
        hbox.addWidget(self.length_spinbox)
        menu = QtWidgets.QMenu('menu')
        radia = menu.addAction('Radia', partial(self.menu_clicked, 'Radia'))
        opera = menu.addAction('Opera 2D', partial(self.menu_clicked, 'Opera 2D'))
        radia.setEnabled(rad is not None)
        opera.setEnabled(opera2d is not None)
        self.build_button = QtWidgets.QToolButton()
        self.build_button.setPopupMode(1)  # add a dropdown arrow
        self.build_button.setText('Radia' if rad is not None else 'Opera 2D')
        self.build_button.setEnabled(rad is not None or opera2d is not None)
        self.build_button.clicked.connect(self.build_model)
        self.build_button.setMenu(menu)
        hbox.addWidget(self.build_button)
        self.layout.addLayout(hbox)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximum(25)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)
        self.status_bar = QtWidgets.QLabel()
        vert_layout.addWidget(self.status_bar)

        self.tab_control = QtWidgets.QTabWidget()
        self.plot = MplCanvas(width=5, height=4, dpi=100)
        self.tab_control.addTab(self.plot, '2D')
        pane = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout()
        pane.setLayout(hbox)
        self.harmonics = [MplCanvas(width=2, height=4, dpi=100), MplCanvas(width=2, height=4, dpi=100)]
        hbox.addWidget(self.harmonics[0])
        hbox.addWidget(self.harmonics[1])
        self.tab_control.addTab(pane, 'Harmonics')

        self.plot3d = MplCanvas3d()
        self.tab_control.addTab(self.plot3d, '3D')

        self.plot.setMinimumWidth(300)
        NavigationToolbar(self.plot, self.plot)
        horiz_layout.addWidget(self.tab_control)  # self.plot)
        horiz_layout.setStretch(0, 1)  # listbox stretches a bit...
        horiz_layout.setStretch(1, 0)  # controls not at all...
        horiz_layout.setStretch(2, 3)  # plot stretches more as window expands

        self.setLayout(vert_layout)
        self.show()
        self.autorun()

    def menu_clicked(self, selected_item):
        self.build_button.setText(selected_item)

    def listview_item_clicked(self, item):
        """Set controls for previously-selected parameters from the list box."""
        was_checked = self.auto_checkbox.isChecked()
        self.auto_checkbox.setChecked(False)  # prevent automatically running models while controls are changed
        params = tuple(item.text().split(', '))
        params_dict = dict([param.split('=') for param in params])
        for control in self.controls:
            control.set_from_args(params_dict)

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

    def get_colour(self):
        """Return an RGB colour combination depending on the multipoles present."""
        colour = [(0.8 if control.get_arg() else 0) for control in self.controls if control.label.endswith('pole')]
        colour[2] += (0.2 if colour[3] else 0)  # fudge to combine octupole + sextupole
        colour.pop(3)  # remove last
        return colour

    def run_simulation(self):
        """Run the simulation using specified parameters."""
        # add command-line arguments
        args = tuple(filter(None, [control.get_arg() for control in self.controls]))
        colour = self.get_colour()

        failed = False
        if args not in self.state['results'].keys():
            self.status_bar.setText('Running...')
            self.progress_bar.setValue(0)
            self.run_button.setEnabled(False)
            QtCore.QCoreApplication.processEvents()
            process = subprocess.Popen(cmd + list(args), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, universal_newlines=True)
            status_line = re.compile(r'Iteration (\d+)')
            while process.poll() is None:
                stdout = process.stdout.readline()
                try:
                    self.progress_bar.setValue(int(status_line.match(stdout).group(1)))
                except AttributeError:  # not a line with "Iteration X" in it
                    pass
                self.status_bar.setText(stdout)
                failed |= 'ERROR' in stdout
                QtCore.QCoreApplication.processEvents()  # update the GUI
            self.run_button.setEnabled(True)
            if failed:
                self.progress_bar.setValue(0)
                self.store_results(args, None, None)
                return
            self.status_bar.setText('Done.')
            self.progress_bar.setValue(self.progress_bar.maximum())

            # get the magnet shapes from the output CSV file
            # format is Bx, By, x0, y0, x1, y1, x2, y2, x3, y3
            # Turn into dimensions in mm
            polygons = np.loadtxt("magnet.csv", skiprows=1, usecols=range(1, 11), delimiter=',') * ([1, 1] + [1000,] * 8)
            harmonics = np.loadtxt("magnet_harmonics.csv", delimiter=',')
            item = self.store_results(args, polygons, harmonics)
        else:  # use saved result
            args_str = ', '.join(args)
            for i in range(self.listview.count()):
                item = self.listview.item(i)
                if item.text() == args_str:
                    item.setSelected(True)
                    break
            result = self.state['results'][args]
            failed = result is None
            polygons = result
            harmonics = self.state['harmonics'][args]


        self.build_button.setEnabled(not failed)
        if failed:
            self.status_bar.setText('Failed.')
        else:
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
            if self.tab_control.currentIndex() > 1:
                self.tab_control.setCurrentIndex(0)  # show 2d view if 3d is selected

            n_harms = len(harmonics)
            show = [True, ] * n_harms
            for i in (1, 0):  # do skew first, and then figure out which not to show for normal
                self.harmonics[i].axes.cla()  # clear axis
                self.harmonics[i].axes.barh(range(n_harms), harmonics[:, i])
                self.harmonics[i].axes.set_xlim(left=min(harmonics[show, i]), right=max(harmonics[show, i]))
                # main harmonic will be 10^4, not interested in seeing that one for main harmonics
                for j, multipole in enumerate(('dipole=', 'quad=', 'sext=', 'oct=')):
                    show[j + 1] = not any(multipole in arg for arg in args)
                self.harmonics[i].axes.invert_yaxis()
                self.harmonics[i].axes.set_title('Normal' if i == 0 else 'Skew')
                self.harmonics[i].draw()

            try:
                img_data = self.state['icons'][args]
            except KeyError:
                # capture plot image and resize to icon size (around 32 pixels)
                width, height = self.plot.get_width_height()
                step = min(height, width) // 32  # how many rows/cols to skip along
                img_data = np.frombuffer(self.plot.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
                img_data = img_data[::step, 1::step, :]  # start from 1 otherwise we get a tiny vertical line
                img_data = img_data.copy()
                self.state['icons'][args] = img_data
                pickle.dump(self.state, bz2.open(state_file, 'wb'))
            height, width, bpp = img_data.shape
            image = QtGui.QImage(img_data, width, height, width * bpp, QtGui.QImage.Format_RGB888)
            icon = QtGui.QIcon(QtGui.QPixmap(image))
            item.setIcon(icon)

    def delete_case(self):
        """Delete button has been clicked - remove a set of arguments from the list."""
        for item in self.listview.selectedItems():
            params = tuple(item.text().split(', '))
            try:
                self.state['results'].pop(params)
                self.state['icons'].pop(params)
            except KeyError:
                pass
            pickle.dump(self.state, bz2.open(state_file, 'wb'))
            self.listview.takeItem(self.listview.row(item))

    def store_results(self, args, result, harmonics):
        """Save a new set of arguments to the list."""
        self.state['results'][args] = result
        self.state['harmonics'][args] = harmonics
        self.listview.addItem(', '.join(args))
        item = self.listview.item(self.listview.count() - 1)
        item.setSelected(True)
        pickle.dump(self.state, bz2.open(state_file, 'wb'))
        return item

    def build_model(self):
        """Build a Radia or Opera model with the current result set."""
        length = self.length_spinbox.value()
        if self.build_button.text() == 'Radia':
            rad.UtiDelAll()
            item = self.listview.selectedItems()[0]
            # build magnet geometry
            magnet = rad.ObjCnt([rad.ObjThckPgn(0, length, pg[2:].reshape((4, 2)).tolist(), "z", list(pg[:2]) + [0, ])
                                 for pg in self.state['results'][tuple(item.text().split(', '))]])
            rad.MatApl(magnet, rad.MatStd('NdFeB', next(c for c in self.controls if c.switch == 'Br').control.value()))

            # plot geometry in 3d
            ax = self.plot3d.axes
            ax.cla()
            ax.set_axis_off()
            polygons = rad.ObjDrwVTK(magnet)['polygons']
            vertices = np.array(polygons['vertices']).reshape((-1, 3))  # [x, y, z, x, y, z] -> [[x, y, z], [x, y, z]]
            [set_lim(vertices.min(), vertices.max()) for set_lim in (ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d)]
            vertices = np.split(vertices, np.cumsum(polygons['lengths'])[:-1])  # split to find each face
            ax.add_collection3d(Poly3DCollection(vertices, linewidths=0.1, edgecolors='black',
                                                 facecolors=self.get_colour(), alpha=0.2))

            # add arrows
            magn_data = np.array(rad.ObjM(magnet)).reshape((-1, 6)).T  # reshape to [x, y, z, mx, my, mz]
            for end in (-1, 1):  # one at each end of the block, not in the middle
                magn_data[2] = end * length / 2
                ax.quiver(*magn_data, color='black', lw=1, pivot='middle')

            self.tab_control.setCurrentIndex(2)  # switch to '3d' tab

            # solve the model
            try:
                rad.Solve(magnet, 0.00001, 10000)  # precision and number of iterations
            except RuntimeError:
                self.status_bar.setText('Radia solve error')

            # get results
            dx = 0.1
            multipoles = [i for i, _ in enumerate(c for c in self.controls if c.label.endswith('pole') and c.get_arg())]
            i = multipoles[-1]
            labels = ['Dipole', 'Quadrupole', 'Sextupole', 'Octupole']
            units = ['T.m', 'T', 'T/m', 'T/m²', 'T/m³']
            xs = np.linspace(-dx, dx, 4)
            fit_field = np.polyfit(xs / 1000, [rad.Fld(magnet, 'by', [x, 0, 0]) for x in xs], i)
            fit_int = np.polyfit(xs / 1000, [rad.FldInt(magnet, 'inf', 'iby', [x, 0, -1], [x, 0, 1]) * 0.001 for x in xs], i)
            text = ''
            for j, (l, c, ic, u, iu) in enumerate(zip(labels, fit_field[::-1], fit_int[::-1], units[1:], units[:-1])):
                if j in multipoles:
                    f = factorial(j)  # 1 for dip, quad; 2 for sext; 6 for oct
                    text += f'{l} field = {c * f:.3g} {u}, integral = {ic * f:.3g} {iu}, length = {ic / c:.3g} m\n'
            ax.text2D(1, 1, text, transform=ax.transAxes, va='top', ha='right', fontdict={'size': 8})
            self.plot3d.canvas.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
