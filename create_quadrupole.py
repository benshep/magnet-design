from operapy import canvas
from operapy import opera2d
import math
import numpy as np

sqrt2 = math.sqrt(2)


def rotate45(points, positive=True):
    """Rotate a set of points through ±45°."""
    return [(((x - y) if positive else (x + y)) / sqrt2,
             ((x + y) if positive else (y - x)) / sqrt2) for x, y in points]


filename = r"C:\Users\bjs54\Documents\OPERA Models\quadrupole.op2_h5"
opera2d.clear()
model = opera2d.get_model_interface()
post = opera2d.get_post_processing_interface(model)
graphing = opera2d.get_graphing_interface(model)
model.use_si_units()
model.use_unit(opera2d.Unit.Length.Millimetre)

settings = model.general_settings
settings.symmetry_type = opera2d.SymmetryType.XY
settings.element_type = opera2d.ElementType.Quadratic
settings.integration_reflection_x = settings.integration_reflection_y = opera2d.FieldReflectionAxisType.NoReflection
settings.mesh_size = 30
settings.periodic_type = opera2d.PeriodicSymmetryType.NoPeriodicity

model.analysis_settings.physics_type = opera2d.PhysicsType.Magnetostatic
model.analysis_settings.restart = False

# dimensions
r = 12.5  # inscribed radius
a = r**2 / 2
hyp_width = 6.4
tip_width = 11
width = 36
taper_height = width + 14
assert width >= tip_width >= hyp_width
hyp_facets = 10
coil_height = 48
coil_width = 10
coil_gap = 2
current_density = 7  # A/mm²

# mesh sizes
gap_mesh = tip_mesh = 0.25
outer_mesh = 10
pole_mesh = 2
yoke_mesh = coil_mesh = 4

# boundary conditions
norm_mag = model.create_boundary_condition('Normal B-field')
norm_mag.set_normal_field_magnetic()
tang_mag = model.create_boundary_condition('Tangential B-field')
tang_mag.set_tangential_field_magnetic()
zero_pot = model.create_boundary_condition('Zero potential')
zero_pot.set_vector_potential(0)

# create background (triangular shape around magnet)
magnet_height = r + taper_height + coil_height + coil_gap + width
bg_extent = magnet_height * 2 / sqrt2
background = model.create_polyline([(magnet_height, 0), (r, 0), (0.8 * r, 0), (0, 0), (r, r),
                                    (bg_extent, bg_extent), (bg_extent, 0)], name='Background', close=True)
# fix mesh and boundary conditions
edges = background.edges
fq_edge = edges[2]  # from (0, 0) to (0.8 * r, 0): evaluate field quality along here
for i, edge in enumerate(edges):
    edge.mesh_size = yoke_mesh if i == 0 else gap_mesh if i < 4 else outer_mesh
    ((x0, y0), (x1, y1)) = (edge.vertices[0].pos, edge.vertices[1].pos)
    # tangential magnetic field along pole; normal magnetic field along x-axis; zero potential elsewhere
    edge.boundary_condition = tang_mag if (x0 == y0 and x1 == y1) else norm_mag if y0 == y1 == 0 else zero_pot

# build pole tip
tip_points = [(x, a / x) for x in np.linspace(r / sqrt2, (r + hyp_width) / sqrt2, hyp_facets)]
x, y = tip_points[-1]
# extend outwards from hyperbola
gradient = -a / x**2
tip_x = (r + tip_width) / sqrt2
tip_points.append((tip_x, y + gradient * (tip_x - x)))
# turn through 45° to make adding the rest of the points easier
tip_points_r = rotate45(tip_points)
x, y = tip_points_r[0]
y += taper_height
tip_points_r.append((width, y))
tip_points_r.append((0, y))
x = width + coil_gap
coil_points = [(x, y), (x + coil_width, y), (x + coil_width, y + coil_height), (x, y + coil_height)]
tip_points = rotate45(tip_points_r, False)
pole_tip = model.create_polyline(tip_points, name='Pole tip', close=True)
for edge in pole_tip.edges:
    edge.mesh_size = tip_mesh

# build pole: start with top of pole tip
pole_points = tip_points_r[-2:]
x, y = pole_points[0]
pole_points.append((0, y + coil_gap + coil_height))
pole_points.append((x, y + coil_gap + coil_height))
pole_points = rotate45(pole_points, False)
pole = model.create_polyline(pole_points, name='Pole', close=True)
for edge in pole.edges:
    edge.mesh_size = pole_mesh

# build rest of yoke
yoke_points = pole_points[-2:]
x, y = yoke_points[-1]
d = (coil_gap * 2 + coil_width) / sqrt2
x, y = x + d, y - d
yoke_points.append((x, y))
yoke_points.append((x, 0))
yoke_points.append((x + width, 0))
yoke_points.append((x + width, y + width / (2 * sqrt2)))
x, y = yoke_points[0]
yoke_points.append((x + width / sqrt2, y + width / sqrt2))
yoke = model.create_polyline(yoke_points, name='Yoke', close=True)
for edge in yoke.edges:
    edge.mesh_size = yoke_mesh

coil_points = rotate45(coil_points, False)  # rotate back to the right place
assert all(y > 0 for x, y in coil_points)  # check coil does not dip below axis
coil = model.create_polyline(coil_points, name='Coil', close=True)
for edge in coil.edges:
    edge.mesh_size = coil_mesh

# set material properties
steel_bh_curve = model.create_bh_curve('1010')
steel_bh_curve.load(r'C:\Program Files\OperaFEA\Opera 2020\code\bh\tenten.bh')
steel = model.create_material('Steel')
steel.permeability_type = opera2d.MaterialPermeabilityType.Nonlinear
steel.directionality = opera2d.Directionality.Isotropic
steel.bh_curve = steel_bh_curve

for region in yoke.regions + pole.regions + pole_tip.regions:
    region.material = steel

conductor = model.create_properties('Conductor')
conductor.current_density = opera2d.ModelValue(current_density, opera2d.Unit.CurrentDensity.AmperePerMillimetreSquared)
coil.regions[0].properties = conductor

model.generate_mesh()

# solve the model
model.solve(filename, overwrite=True)

# post-processing
bydx = "-hydx*mu0*1000"
gradient = post.calculate_field_at_point((0, 0), bydx, name='Gradient')
value = gradient.field_expression_result
model.create_variable('#grad', value)
print(f'Gradient: {value:.3f} T/m')

for part in (pole_tip, pole, yoke):
    name = part.name.lower()
    bmod = post.calculate_contour_map(part.regions, 'bmod', 30, name=f'Bmod ({name})')
    bmax, max_pos = bmod.get_max()
    print(f'Maximum field in {name}: {bmax:.3f} T')
bmod = post.calculate_contour_map([], 'bmod', 30, name='Bmod (all)')  # display whole map over the top
flux_lines = post.calculate_contour_map(field_expression='pot', contour_count=10, type=opera2d.ContourMapType.Lines,
                                        name='Flux lines')

name = 'Gradient vs x'
rel_grad = bydx + '/#grad-1'
graphing.create_buffer_from_fields_on_edges(name, [fq_edge], 100, [rel_grad])
graphing.create_line_from_buffer(name, name, 'x', rel_grad)
graph_name = 'Field quality'
graphing.create_graph(graph_name)
graphing.plot_line(name, graph_name)
graphing.show_graph(graph_name)
data = graphing.get_column_data(name, rel_grad)
field_quality = max(data) - min(data)
print(f'Field quality: {min(data):.3e}, {max(data):.3e}')

view = canvas.get_view()
view.set_view(0, 0, magnet_height / sqrt2)  # zoom so top of magnet is in view
# view.show_mesh = True
model.save_as(filename, overwrite=True)
