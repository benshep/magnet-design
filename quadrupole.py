import os
import sys
import math
import numpy as np
import win32com.client as client
from enum import Enum
from time import sleep
import pywintypes

try:
    from scipy.optimize import minimize_scalar
except (ModuleNotFoundError, ImportError):
    minimize_scalar = None
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

sq2 = math.sqrt(2)
by_dx = "-HyDx*mu0*1000"


class GeometryError(Exception):
    pass


class SimulationCodeError(Exception):
    pass


def rotate45(points, positive=True):
    """Rotate a set of points through ±45°."""
    return [[((x - y) if positive else (x + y)) / sq2,
             ((x + y) if positive else (y - x)) / sq2] for x, y in points]


def to_tuples(points):
    """Convert a list of lists to a list of tuples."""
    return [tuple(pair) for pair in points]


script_name = rotate45.__code__.co_filename
script_folder = os.path.dirname(script_name)
state_filename = os.path.join(script_folder, 'opera-2d-3d-loop-state.txt')


class SolveState(Enum):
    NONE, SHAPES, BUILT, SOLVED = range(4)

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Quadrupole:
    """The base class with basic geometry-building functions."""

    def __init__(self, **kwargs):
        """Initialise a Quadrupole instance with some basic parameters."""
        self.r = self.hyp_width = self.tip_width = self.width = self.taper_height = None
        self.coil_height = self.coil_width = self.current_density = None
        # some have default values
        self.min_mesh = 0.25
        self.hyp_facets = 10
        self.coil_gap = 2
        self.solve_state = SolveState.NONE
        self.background_points = self.coil_x = self.coil_y = None
        self.tip_points = self.pole_points = self.yoke_points = self.coil_points = None
        self.set_param(**kwargs)

    def set_param(self, **params):
        """Set a parameter value and reset the solve state."""
        for parameter, value in params.items():
            setattr(self, parameter, value)
            self.solve_state = SolveState.NONE

    def define_shapes(self):
        """Define the 2D shapes comprising the outline of the quadrupole."""

        a = self.r ** 2 / 2
        if not self.width >= self.tip_width >= self.hyp_width:
            raise GeometryError(f'Width ({self.width}) must be >= tip width ({self.tip_width}), '
                                f'which must be >= hyperbolic face width ({self.hyp_width}).')

        # create background (triangular shape around magnet)
        magnet_height = self.r + self.taper_height + self.coil_height + self.coil_gap + self.width
        bg_extent = magnet_height * 2 / sq2
        self.background_points = [[magnet_height, 0], [self.r, 0], [0.8 * self.r, 0], [0, 0], [self.r, self.r],
                                  [bg_extent, bg_extent], [bg_extent, 0]]

        # pole tip
        tip_points = [[x, a / x] for x in np.linspace(self.r / sq2, (self.r + self.hyp_width) / sq2, self.hyp_facets)]
        x, y = tip_points[-1]
        # extend outwards from hyperbola
        gradient = -a / x ** 2
        tip_x = (self.r + self.tip_width) / sq2
        tip_points.append([tip_x, y + gradient * (tip_x - x)])
        # turn through 45° to make adding the rest of the points easier
        tip_points_r = rotate45(tip_points)
        x, y = tip_points_r[0]
        y += self.taper_height
        tip_points_r.append([self.width, y])
        tip_points_r.append([0, y])
        x = self.width + self.coil_gap
        self.coil_x, self.coil_y = x, y
        coil_points = [[x, y], [x + self.coil_width, y],
                       [x + self.coil_width, y + self.coil_height], [x, y + self.coil_height]]
        self.tip_points = rotate45(tip_points_r, False)

        # pole: start with top of pole tip
        pole_points = tip_points_r[-2:]
        x, y = pole_points[0]
        pole_points.append([0, y + self.coil_gap + self.coil_height])
        pole_points.append([x, y + self.coil_gap + self.coil_height])
        self.pole_points = rotate45(pole_points, False)

        # rest of yoke
        yoke_points = self.pole_points[-2:]
        x, y = yoke_points[-1]
        d = (self.coil_gap * 2 + self.coil_width) / sq2
        x, y = x + d, y - d
        yoke_points.extend([[x, y], [x, 0], [x + self.width, 0], [x + self.width, y + self.width / (2 * sq2)]])
        x, y = yoke_points[0]
        yoke_points.extend([[x + self.width * sq2, y], [x + self.width / sq2, y + self.width / sq2]])
        self.yoke_points = yoke_points

        self.coil_points = rotate45(coil_points, False)  # rotate back to the right place
        if not all(y > 0 for x, y in self.coil_points):
            raise GeometryError("Coil protrudes below x-axis.")
        self.solve_state = SolveState.SHAPES

    def scan(self, vary, scan_values, return_value):
        """
        Perform a parameter sweep over a given range.
        :param return_value: function to get output parameter
        :param vary: name of parameter to vary
        :param scan_values:  list of values to scan over (use numpy.linspace to generate)
        :return:
        """
        scan_result = []
        for value in scan_values:
            print(f'Scan: set {vary} to {value:.3g}')
            self.set_param(**{vary: value})
            ret_val = return_value()
            print(f'{return_value.__name__} = {ret_val:.3g}')
            scan_result.append(ret_val)
        return scan_result

    def find_optimum(self, vary, start, return_value, find_min=True, **parameters):
        """
        Find an optimum value for a given parameter.
        :param return_value: function to get output parameter
        :param find_min: look for a minimum? if False, looks for a maximum instead
        :param vary: name of parameter to vary
        :param start: tuple of 2 (or 3) numbers to start with, to pass to the optimizer
        :return:
        """
        assert minimize_scalar is not None
        input_values = []
        output_values = []

        def build_wrapper(value):
            """Wrapper around build function so that scipy's minimize function can call it."""
            print(f'Building with {vary} = {value}')
            input_values.append(value)
            self.set_param(**{vary: value})
            result = return_value()
            output_values.append(result)
            return result * (1 if find_min else -1)

        res = minimize_scalar(build_wrapper, start, options={'xtol': 20e-3})
        input_values, output_values = zip(*sorted(zip(input_values, output_values)))  # sort by input value
        return res, input_values, output_values

    def int_by_dz(self, x):
        return 0

    def integrated_gradient(self):
        """Return the integrated gradient (in T) close to the axis."""
        x = 0.1
        return self.int_by_dz(x) / x

    def field_quality(self):
        """Return the difference between the maximum and minimum relative gradient."""
        fq = self.relative_gradient()
        return max(fq) - min(fq)

    def gradient(self):
        return self.integrated_gradient()

    def central_gradient(self):
        return self.integrated_gradient()

    def relative_gradient(self):
        """Return the relative gradient quality over the aperture."""
        x_values = np.linspace(0, self.r * 0.8, 20)
        dx = x_values[1] - x_values[0]
        ints = np.vectorize(self.int_by_dz)(x_values)
        grads = (ints[1:] - ints[:-1]) / dx
        return grads / grads[0] - 1

    def magnetic_length(self):
        """Return the magnetic length."""
        return self.gradient() / self.central_gradient()


class Opera2DQuadrupole(Quadrupole):
    def __init__(self, filename='quadrupole', **kwargs):
        if opera2d is None:
            raise SimulationCodeError("No Opera-2D instance found")
        super().__init__(**kwargs)
        self.filename = os.path.join(script_folder, filename)  # used as base name with no extension
        self.fq_edge = self.pole_tip = self.pole = self.yoke = self.coil = self.results = None
        self.bh_file = os.path.abspath(os.path.join(os.path.dirname(sys.executable), os.pardir, 'bh', 'TenTen.bh'))

    def build(self):
        """Create a quadrupole with the given geometry."""
        if self.solve_state < SolveState.SHAPES:
            self.define_shapes()
        opera2d.clear()
        model.use_si_units()
        model.use_unit(opera2d.Unit.Length.Millimetre)

        settings = model.general_settings
        settings.symmetry_type = opera2d.SymmetryType.XY
        settings.element_type = opera2d.ElementType.Quadratic
        settings.integration_reflection_x = settings.integration_reflection_y = opera2d.FieldReflectionAxisType.NoReflection
        settings.periodic_type = opera2d.PeriodicSymmetryType.NoPeriodicity

        model.analysis_settings.physics_type = opera2d.PhysicsType.Magnetostatic
        model.analysis_settings.restart = False

        # mesh sizes
        gap_mesh = tip_mesh = self.min_mesh
        pole_mesh = self.min_mesh * 4
        outer_mesh = self.min_mesh * 40
        yoke_mesh = coil_mesh = self.min_mesh * 16
        settings.mesh_size = self.min_mesh * 120

        # boundary conditions
        norm_mag = model.create_boundary_condition('Normal B-field')
        norm_mag.set_normal_field_magnetic()
        tang_mag = model.create_boundary_condition('Tangential B-field')
        tang_mag.set_tangential_field_magnetic()
        zero_pot = model.create_boundary_condition('Zero potential')
        zero_pot.set_vector_potential(0)

        # create background (triangular shape around magnet)
        background = model.create_polyline(to_tuples(self.background_points), name='Background', close=True)

        # fix mesh and boundary conditions
        edges = background.edges
        self.fq_edge = edges[2]  # from (0, 0) to (0.8 * r, 0): evaluate field quality along here
        for i, edge in enumerate(edges):
            edge.mesh_size = yoke_mesh if i == 0 else gap_mesh if i < 4 else outer_mesh
            ((x0, y0), (x1, y1)) = (edge.vertices[0].pos, edge.vertices[1].pos)
            # tangential magnetic field along pole; normal magnetic field along x-axis; zero potential elsewhere
            edge.boundary_condition = tang_mag if (x0 == y0 and x1 == y1) else norm_mag if y0 == y1 == 0 else zero_pot

        # build pole tip
        self.pole_tip = model.create_polyline(to_tuples(self.tip_points), name='Pole tip', close=True)
        for edge in self.pole_tip.edges:
            edge.mesh_size = tip_mesh

        # build pole
        self.pole = model.create_polyline(to_tuples(self.pole_points), name='Pole', close=True)
        for edge in self.pole.edges:
            edge.mesh_size = pole_mesh

        # build rest of yoke
        self.yoke = model.create_polyline(to_tuples(self.yoke_points), name='Yoke', close=True)
        for edge in self.yoke.edges:
            edge.mesh_size = yoke_mesh

        self.coil = model.create_polyline(to_tuples(self.coil_points), name='Coil', close=True)
        for edge in self.coil.edges:
            edge.mesh_size = coil_mesh

        # set material properties
        steel_bh_curve = model.create_bh_curve('1010')
        steel_bh_curve.load(self.bh_file)
        steel = model.create_material('Steel')
        steel.permeability_type = opera2d.MaterialPermeabilityType.Nonlinear
        steel.directionality = opera2d.Directionality.Isotropic
        steel.bh_curve = steel_bh_curve

        for region in self.yoke.regions + self.pole.regions + self.pole_tip.regions:
            region.material = steel

        conductor = model.create_properties('Conductor')
        conductor.current_density = opera2d.ModelValue(self.current_density,
                                                       opera2d.Unit.CurrentDensity.AmperePerMillimetreSquared)
        self.coil.regions[0].properties = conductor
        self.solve_state = SolveState.BUILT

    def solve(self):
        """Solve the model."""
        if self.solve_state < SolveState.BUILT:
            self.build()
        model.generate_mesh()

        # solve the model
        model.solve(self.filename + ".op2_h5", overwrite=True)
        self.solve_state = SolveState.SOLVED

    def gradient(self):
        """Return the central gradient in the model."""
        if self.solve_state < SolveState.SOLVED:
            self.solve()
        grad_calc = post.calculate_field_at_point((0, 0), by_dx, name='Gradient')
        grad = grad_calc.field_expression_result
        model.create_variable('#grad', grad)
        return grad

    def max_bmod(self):
        """Return the maximum field in each model part."""
        if self.solve_state < SolveState.SOLVED:
            self.solve()
        max_bmod = {}
        for part in (self.pole_tip, self.pole, self.yoke):
            part_name = part.name.lower()
            bmod = post.calculate_contour_map(part.regions, 'bmod', 30, name=f'Bmod ({part_name})')
            bmax, max_pos = bmod.get_max()
            max_bmod[part_name] = bmax
        post.calculate_contour_map([], 'bmod', 30, name='Bmod (all)')  # display whole map over the top
        post.calculate_contour_map(field_expression='pot', contour_count=10,
                                   type=opera2d.ContourMapType.Lines, name='Flux lines')
        return max_bmod

    def relative_gradient(self):
        """Return the gradient relative to the central value over the aperture."""
        self.gradient()  # store in #grad
        label = 'Gradient vs x'
        rel_grad = f'{by_dx}/#grad-1'
        graphing.create_buffer_from_fields_on_edges(label, [self.fq_edge], 100, [rel_grad])
        graphing.create_line_from_buffer(label, label, 'x', rel_grad)
        graph_name = 'Field quality'
        graphing.create_graph(graph_name)
        graphing.plot_line(label, graph_name)
        graphing.show_graph(graph_name)
        return graphing.get_column_data(label, rel_grad)

    def refresh_view(self):
        view = canvas.get_view()
        magnet_height = self.r + self.taper_height + self.coil_height + self.coil_gap + self.width
        view.set_view(0, 0, magnet_height / sq2)  # zoom so top of magnet is in view
        # view.show_mesh = True

    def save_model(self):
        model.save_as(f'{self.filename}.op2_h5', overwrite=True)

    def scan(self, vary, scan_values, return_value):
        scan_result = super().scan(vary, scan_values, return_value)
        name = f'Sweep of {vary}'
        graphing.create_line(name, list(scan_values), scan_result, x_label=vary, y_label=return_value.__name__)
        graph_name = 'Parameter sweep'
        graphing.create_graph(graph_name)
        graphing.plot_line(name, graph_name)
        graphing.show_graph(graph_name)
        return scan_result

    def find_optimum(self, vary, start, return_value, find_min=True, **parameters):
        res, input_values, output_values = super().find_optimum()
        name = f'Optimisation of {vary}'
        graphing.create_line(name, list(input_values), list(output_values), x_label=vary, y_label=return_value)
        graph_name = 'Parameter sweep'
        graphing.create_graph(graph_name)
        graphing.plot_line(name, graph_name)
        graphing.show_graph(graph_name)
        return res, input_values, output_values


class Opera3DQuadrupole(Opera2DQuadrupole):

    def __init__(self, **kwargs):
        self.length = None
        self.chamfer = 0
        super().__init__(**kwargs)
        try:
            self.opera = client.Dispatch('operaFEA_manager.OperaApplication')
        except pywintypes.com_error:
            raise SimulationCodeError("Couldn't start Opera - is the Manager running?")

    def create_3d_comi(self):
        """Create a COMI script to define the variables to build and solve in 3D."""
        if self.solve_state < SolveState.SHAPES:
            self.define_shapes()
        # Need to export to legacy .op2 format first - Opera 3D can't import .op2_h5 yet
        opera2d.export_to_op2(f'{self.filename}.op2')
        variable_list = [('length', 'Quadrupole half-length'), ('r', 'Inscribed radius'),
                         ('min_mesh', 'Minimum mesh size'), ('current_density', 'Coil current density'),
                         ('coil_x', 'Coil X position'), ('coil_y', 'Coil Y position'),
                         ('coil_width', 'Coil width'), ('coil_height', 'Coil height'),
                         ('hyp_width', 'Width of hyperbolic pole section'), ('chamfer', 'Depth of end chamfer')]
        define_vars = '\n'.join([f"variable option=constant name=#{name} value={{{name}}} description='{description}'"
                                 for name, description in variable_list]).format(**vars(self))
        comi_name = os.path.join(script_folder, 'quadrupole.comi')
        text = f"""
        / Set up variables to build a quadrupole using a 2D model as a starting point
        / Automatically generated by {script_name}
        / Also see 2D file at {self.filename}.op2_h5
        $string YesOrNo 'yes'
        clear revert=no
        {define_vars}
        $string name=filename value='{self.filename}' 
        BHData Option=Load Label=steel File='{self.bh_file}'

        / Now build the quadrupole using these variables
        $comi  '{comi_name}'
        """
        open(f'{self.filename}-setup.comi', 'w').write('\n'.join(line.lstrip() for line in text.split('\n')))

    def build(self):
        """Invoke the Opera-3D Modeller and build the file in there."""
        super().build()  # need to do this to export .op2 file
        self.create_3d_comi()
        self.opera.stop(True)  # in case a Post-Processor instance has been running - would fail to start the Modeller
        if not self.opera.startModeller(True):
            raise SimulationCodeError("Couldn't start the 3D Modeller")
        # This will invoke quadrupole-setup.comi and then quadrupole.comi, saving the file as an .op3 but not solving
        self.opera.giveCommand(f"$comi '{self.filename}-setup.comi'")
        self.opera.stop(True)
        self.solve_state = SolveState.BUILT

    def solve(self):
        """Solve the model using the Opera-3D solver."""
        if self.solve_state < SolveState.BUILT:
            self.build()
        self.opera.startAnalysis(f'{self.filename}.op3', 'operaFEA-op3solve')  # always returns False?
        self.solve_state = SolveState.SOLVED

    def load_in_pp(self):
        """Open the model in the Post-Processor."""
        if self.solve_state < SolveState.SOLVED:
            self.solve()
        if not self.opera.startPost(True):  # returns False if already started - don't need to load OP3 file
            return
        self.opera.giveCommand(f"activate case=1 ModelSymmetry=database File='{self.filename}.op3' | load")
        self.opera.giveCommand('set field=integration')

    def int_by_dz(self, x):
        """Return the integral along z of the vertical B-field."""
        self.load_in_pp()
        self.opera.giveCommand(f"line buffer='Line' X1={x} X2=X1 Y1=0 Y2=Y1 Z1=0 Z2=#length*3 NP=#length*3")
        self.opera.giveCommand('plot component=by')  # to store in "Integral" variable
        return self.opera.getValue('integral') * 2  # double since we're only looking at half the model

    def gradient(self, integrated=True):
        """Gradient is ambiguous, so we've defined central and integrated separately."""
        return self.integrated_gradient() if integrated else self.central_gradient()

    def central_gradient(self):
        """Return the gradient (in T) at the centre of the magnet."""
        self.load_in_pp()
        x = 0.1
        self.opera.giveCommand(f'point xp={x} yp=0 zp=0 component=by')
        return self.opera.getValue('By') / x

    def relative_gradient(self):
        """Override the Opera-2D version of this with the default version."""
        return Quadrupole.relative_gradient(self)


class RadiaQuadrupole(Quadrupole):
    def __init__(self, **kwargs):
        if rad is None:
            raise SimulationCodeError("Couldn't import Radia")
        self.length = None
        self.radia_object = None
        super().__init__(**kwargs)
        self.min_mesh = 3  # can be much larger than Opera
        self.pole_mult = 2
        self.yoke_mult = 5

    def build(self):
        """Create a quadrupole with the given geometry."""
        if self.solve_state < SolveState.SHAPES:
            self.define_shapes()

        rad.UtiDelAll()
        origin = [0, 0, 0]
        nx = [1, 0, 0]
        ny = [0, 1, 0]
        nz = [0, 0, 1]

        tip_mesh = round(self.min_mesh)
        pole_mesh = round(self.min_mesh * self.pole_mult)
        yoke_mesh = round(self.min_mesh * self.yoke_mult)

        length = self.length

        # Subdivide the pole tip cylindrically. The axis is where the edge of the tapered pole meets the Y-axis.
        points = rotate45(self.tip_points)
        x2, y2 = points[-2]  # top right of pole
        x3, y3 = points[-3]  # bottom right of pole
        m = (y2 - y3) / (x2 - x3)
        c = y2 - m * x2
        pole_tip = rad.ObjThckPgn(length / 2, length, points, "z")
        # Slice off the chamfer (note the indexing at the end here - selects the pole not the cut-off piece)
        pole_tip = rad.ObjCutMag(pole_tip, [length - self.chamfer, 0, self.r], [1, 0, -1])[0]
        n_div = max(1, round(math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) / pole_mesh))
        # We have to specify the q values here (second element of each sublist in the subdivision argument)
        # otherwise weird things happen
        mesh = [[n_div, 4], [tip_mesh / 3, 1], [tip_mesh, 1]]
        div_opts = 'Frame->Lab;kxkykz->Size'
        # rad.ObjDivMag(pole_tip, [[tip_mesh, 1], [tip_mesh, 1], [tip_mesh, 3]], div_opts)
        rad.ObjDivMag(pole_tip, mesh, "cyl", [[[0, c, 0], nz], nx, 1], div_opts)
        rad.TrfOrnt(pole_tip, rad.TrfRot(origin, nz, -math.pi / 4))

        pole = rad.ObjThckPgn(length / 2, length, rotate45(self.pole_points), "z")
        rad.ObjDivMag(pole, [pole_mesh, ] * 3, div_opts)
        rad.TrfOrnt(pole, rad.TrfRot(origin, nz, -math.pi / 4))

        # Need to split yoke since Radia can't build concave blocks
        points = rotate45(self.yoke_points[:2] + self.yoke_points[-2:])
        # yoke1 is the part that joins the pole to the yoke
        # Subdivide this cylindrically since the flux goes around a corner here
        # The axis is the second point (x1, y1)
        x1, y1 = points[1]
        yoke1 = rad.ObjThckPgn(length / 2, length, points, "z")
        cyl_div = [[[x1, y1, 0], nz], [self.width, self.width, 0], 1]
        # The first (kr) argument, corresponding to radial subdivision,
        # in rad.ObjDivMag cuts by number not size even though kxkykz->Size is specified.
        # So we have to fudge this. It seems to require a larger number to give the right number of subdivisions.
        n_div = max(1, round(2 * self.width / yoke_mesh))
        rad.ObjDivMag(yoke1, [n_div, yoke_mesh, yoke_mesh], "cyl", cyl_div, div_opts)
        rad.TrfOrnt(yoke1, rad.TrfRot(origin, nz, -math.pi / 4))

        # For the second part of the yoke, we use cylindrical subdivision again. But the axis is not on the corner;
        # instead we calculate the point where the two lines converge (xc, yc).
        points = self.yoke_points[1:3] + self.yoke_points[-3:-1]
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        x3, y3 = points[3]
        m1 = (y3 - y0) / (x3 - x0)
        m2 = (y2 - y1) / (x2 - x1)
        c1 = y0 - m1 * x0
        c2 = y1 - m2 * x1
        xc = (c2 - c1) / (m1 - m2)
        yc = m1 * xc + c1
        yoke2 = rad.ObjThckPgn(length / 2, length, points, 'z')
        cyl_div = [[[xc, yc, 0], nz], [x3 - xc, y3 - yc, 0], 1]
        n_div = max(1, round(0.7 * n_div))  # this is a bit of a fudge
        rad.ObjDivMag(yoke2, [n_div, yoke_mesh, yoke_mesh], "cyl", cyl_div, div_opts)

        yoke3 = rad.ObjThckPgn(length / 2, length, self.yoke_points[2:6], "z")
        rad.ObjDivMag(yoke3, [yoke_mesh, ] * 3, div_opts)

        steel = rad.ObjCnt([pole_tip, pole, yoke1, yoke2, yoke3])
        rad.ObjDrwAtr(steel, [0, 0, 1], 0.001)  # blue steel
        rad.TrfOrnt(steel, rad.TrfRot(origin, ny, -math.pi / 2))
        rad.ObjDrwOpenGL(steel)
        rad.TrfOrnt(steel, rad.TrfRot(origin, ny, math.pi / 2))
        # rad.TrfMlt(steel, rad.TrfPlSym([0, 0, 0], [1, -1, 0]), 2)  # reflect along X=Y line to create a quadrant
        rad.TrfZerPerp(steel, origin, [1, -1, 0])
        rad.TrfZerPerp(steel, origin, nz)
        steel_material = rad.MatSatIsoFrm([2000, 2], [0.1, 2], [0.1, 2])
        steel_material = rad.MatStd('Steel42')
        steel_material = rad.MatSatIsoFrm([959.703184, 1.41019852], [33.9916543, 0.5389669], [1.39161186, 0.64144324])
        rad.MatApl(steel, steel_material)

        coil = rad.ObjRaceTrk(origin, [5, 5 + self.coil_width],
                              [self.coil_x * 2 - self.r, length * 2], self.coil_height, 4, self.current_density)
        rad.TrfOrnt(coil, rad.TrfRot(origin, nx, -math.pi / 2))
        rad.TrfOrnt(coil, rad.TrfTrsl([0, self.r + self.taper_height + self.coil_height / 2, 0]))
        rad.TrfOrnt(coil, rad.TrfRot(origin, nz, -math.pi / 4))
        rad.ObjDrwAtr(coil, [1, 0, 0], 0.001)  # red coil
        quad = rad.ObjCnt([steel, coil])

        rad.TrfZerPara(quad, origin, nx)
        rad.TrfZerPara(quad, origin, ny)

        # rad.ObjDrwOpenGL(quad)
        self.radia_object = quad
        self.solve_state = SolveState.BUILT

    def solve(self):
        """Solve the model."""
        if self.solve_state < SolveState.BUILT:
            self.build()
        # print('solving')
        res = rad.Solve(self.radia_object, 0.00001, 10000)  # precision and number of iterations
        self.solve_state = SolveState.SOLVED
        # print(res)
        # print(rad.Fld(self.radia_object, "B", [0, 0, 0]))
        # print(rad.Fld(self.radia_object, "B", [1, 0, 0]))
        # print(rad.FldLst(self.radia_object, "By", [1, 0, 0], [1, 0, self.length], 100, "noarg"))

    def int_by_dz(self, x):
        """Return the integral along z of the vertical B-field."""
        if self.solve_state < SolveState.SOLVED:
            self.solve()
        return rad.FldInt(self.radia_object, 'inf', 'iby', [x, 0, -1], [x, 0, 1])

    def central_gradient(self):
        """Return the gradient (in T) at the centre of the magnet."""
        if self.solve_state < SolveState.SOLVED:
            self.solve()
        x = 0.1
        return rad.Fld(self.radia_object, 'by', [x, 0, 0]) / x

    def check_segments(self, container):
        """Loop through all the objects in a container and evaluate the segmentation.
        Good shapes will have a magnetisation perpendicular to one of the faces.
        So find the normal of each face and evaluate the dot product with  the magnetisation, both normalised to 1.
        The best have a max dot product of 1. Theoretical min is 1/sqrt(3) though most will be above 1/sqrt(2)."""
        shapes = rad.ObjCntStuf(container)
        xmin, xmax, ymin, ymax, zmin, zmax = rad.ObjGeoLim(container)
        print(f'Checking {len(shapes)} shapes in {container}, extent: x {xmin:.1f} to {xmax:.1f}, y {ymin:.1f} to {ymax:.1f}, z {zmin:.1f} to {zmax:.1f}')
        dot_products = {}
        for shape in shapes:
            sub_shapes = rad.ObjCntStuf(shape)
            if len(sub_shapes) > 0:  # this shape is a container
                dot_products.update(self.check_segments(shape))  # recurse and update dict
            else:  # it's an atomic shape
                mag = rad.ObjM(shape)[0]  # returns [[mx, my, mz]], select the first element i.e. [mx, my, mz]
                norm = np.linalg.norm(mag)  # normalise so total is 1
                if norm == 0:
                    continue
                mag = mag / norm
                # Have to parse the information from rad.UtiDmp, no other way of getting polyhedron faces!
                info = rad.UtiDmp(shape).replace('{', '[').replace('}', ']')  # convert from Mathematica list format
                in_face_list = False
                # print(info)
                lines = info.split('\n')
                description = lines[0].split(': ')
                # print(description)
                object_type = description[-1]
                # print('Type is', object_type)
                if object_type == 'RecMag':  # cuboid with axes parallel to x, y, z
                    # simply find the largest component of normalised magnetisation - the closer to 1, the better
                    dot_products[shape] = max(abs(mag))
                elif object_type == 'Polyhedron':  # need to loop over the faces
                    product_list = []
                    for line in lines[1:]:
                        if in_face_list:
                            if '[' not in line:  # reached the end of the face list
                                break
                            points = np.array(eval(line.rstrip(',')))
                            normal = np.cross(points[1] - points[0], points[2] - points[0])
                            product_list.append(np.dot(normal / np.linalg.norm(normal), mag))  # normalise->unit vector
                        elif 'Face Vertices' in line:
                            in_face_list = True
                    dot_products[shape] = max(product_list)  # max seems to be a reasonable figure of merit
        return dot_products


if __name__ == '__main__':
    width = 36
    quad = RadiaQuadrupole(filename='quadrupole2', r=12.5, hyp_width=6.2, tip_width=11, width=width,
                           taper_height=width + 14, coil_height=48, coil_width=10, current_density=7,
                           length=60, chamfer=4)
    quad.define_shapes()
    quad.set_param(min_mesh=4)

    # quad.solve_state = SolveState.SOLVED

    # print(f'Integrated field along axis: {quad.int_by_dz(0):.3f} T')
    # print(f'Integrated gradient: {quad.gradient():.3f} T')
    print(f'Gradient: {quad.central_gradient() * 1000:.3f} T/m')
    dot_products = quad.check_segments(quad.radia_object)
    if len(dot_products) > 0:
        for q in (50, 80, 95):
            print(f'{100 - q}% above {np.percentile(list(dot_products.values()), q):.3f}')
    # print(f'Magnetic length: {quad.magnetic_length():.3f} mm')
    # for name, bmax in quad.max_bmod().items():
    #     print(f'Maximum field in {name}: {bmax:.3f} T')
    # scan_result = quad.scan('hyp_width', np.linspace(6, 7, 11), quad.field_quality)
    # scan_result = quad.scan('chamfer', np.linspace(0.5, 6, 12), quad.field_quality)
    # scan_result = quad.scan('min_mesh', np.linspace(9, 1, 9), quad.field_quality)
    # print(scan_result)
    # print(f'Field quality: {quad.field_quality():.3e}')

    # scan('hyp_width', np.linspace(start=6, stop=7, num=21), return_value='field_quality', r=12.5, tip_width=11,
    #      width=width, taper_height=width+14, coil_height=48, coil_width=10, current_density=7)
    # find_optimum('hyp_width', (6, 6.2), r=12.5, return_value='field_quality', tip_width=11, width=width,
    #              taper_height=width + 14, coil_height=48, coil_width=10, current_density=7)

