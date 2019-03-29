import tables
import numpy as np
import matplotlib as mpl
mpl.__path__.append('/usr/lib/python2.7/dist-packages/matplotlib/')
import matplotlib.pyplot as plt
import mpl_toolkits
mpl_toolkits.__path__.append('/usr/lib/python3/dist-packages/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap
import sys
import collections
import argparse
import re
import gc

# Start Using SHAPELY
import shapely.geometry as geometry
from shapely.geometry import Polygon, MultiPoint, Point
from shapely.ops import triangulate
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay

from descartes.patch import PolygonPatch

from simplekml import Kml, OverlayXY, ScreenXY, Units, RotationXY, AltitudeMode, Camera
import tempfile
import os

import configparser

if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")

def coords_span(values):
    if isinstance(values, str):
        return [(float(re.sub('[^0-9\.]', '', v)) * (-1 if v[-1].upper() in ('S', 'W') else 1) if v else None) for v in values.split('-')]
    elif isinstance(values, collections.Iterable):
        return [values[0], values[-1]]
    else:
        return [values]

class CoordsSpanArg(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(CoordsSpanArg, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        l = coords_span(values)
        setattr(namespace, self.dest, (l[0], l[-1]))

class DefaultListAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        self.choices = kwargs.get('choices', None)
        self.values = []
        super(DefaultListAction, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, value, option_string=None):
        if value not in self.choices:
            message = ("invalid choice: {0!r} (choose from {1})"
                        .format(value,
                                ', '.join([repr(action)
                                            for action in self.choices])))

            raise argparse.ArgumentError(self, message)
        elif value not in self.values:
            self.values.append(value)
        setattr(namespace, self.dest, self.values)

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('-c', '--config', help='configuration file; other options will override the settings in the file',
                default='{filename}.ini'.format(filename=os.path.splitext(__file__)[0]))
args, unknown_args = ap.parse_known_args()

config = configparser.ConfigParser()
config.read(args.config)

ap = argparse.ArgumentParser(description='Extracts and plots data from GMI and GPM HDF5 files.',
                             epilog='Finally, -c CONFIG, --config CONFIG loads all the missing parameters.')
ap.add_argument('-lat', '--latitude', help='latitude range to draw (default: 90S-90N)',
                action=CoordsSpanArg, default=coords_span(config.get('coordinates', 'latitude', fallback=[-90, 90])))
ap.add_argument('-lon', '--longitude', help='longitude range to draw (default: 180W-180E)',
                action=CoordsSpanArg, default=coords_span(config.get('coordinates', 'longitude', fallback=[-180, 180])))
ap.add_argument('-i', '--save-image', help='save image instead of displaying a plot', metavar='FILE',
                default=config.get('image', 'file name', fallback=None))
ap.add_argument('-if', '--image-format', help='format to save the image in (default: png)',
                choices=['png', 'svg', 'pdf'], default=config.get('image', 'format', fallback='png'))
ap.add_argument('-dpi', '--image-dpi', help='dpi of the image to be saved (default: 100)',
                type=int, default=config.getint('image', 'dpi', fallback=100))
ap.add_argument('-it', '--image-transparent', help='save image with a transparent background if supported',
                action='store_true', default=config.getboolean('image', 'transparent', fallback=False))
ap.add_argument('--tight-image', help='save image with less white space around',
                action='store_true', default=config.getboolean('image', 'tight', fallback=True))
ap.add_argument('-k', '--save-kml', help='save KML file instead of displaying a plot', metavar='FILE',
                default=config.get('KML', 'file name', fallback=None))
ap.add_argument('--no-colorbar', help='suppress creation of colorbar(s) for plot(s)',
                action='store_true', default=not config.getboolean('plot', 'colorbar', fallback=True))
ap.add_argument('-a', '--alpha', help='a value that defines the size of holes in the map',
                default=config.getfloat('plot', 'alpha', fallback=0))
ap.add_argument('--no-output', help='suppress creation of the result files or plots (for debugging purposes)',
                action='store_true', default=False)
ap.add_argument('-gmi', '--GMI', metavar='FILE', help='GMI file to process',
                default=config.get('GMI', 'file name', fallback=None))
ap.add_argument('-f', '--frequency', help='''probing frequency for the GMI data to plot:
    1) 10.65 GHz V-Pol 2) 10.65 GHz H-Pol
    3) 18.7 GHz V-Pol 4) 18.7 GHz H-Pol
    5) 23.8 GHz V-Pol
    6) 36.64 GHz V-Pol 7) 36.64 GHz H-Pol
    8) 89.0 GHz V-Pol 9) 89.0 GHz H-Pol
    10) 166.0 GHz V-Pol 11) 166.0 GHz H-Pol
    12) 183.31 ±3 GHz V-Pol and
    13) 183.31 ±7 GHz V-Pol
    (all if not set or 0)\
    ''', type=int, required=False, choices=range(0, 14), action=DefaultListAction,
                default=[int(value if value else 0) for value in config.get('GMI', 'frequency', fallback='').split(',')])
ap.add_argument('--gmi-range', help='range of GMI data values to color (auto if not set)',
                action=CoordsSpanArg, default=coords_span(config.get('GMI', 'range', fallback=[None, None])))
ap.add_argument('-gpm', '--GPM', metavar='FILE', help='GPM file to process',
                default=config.get('GPM', 'file name', fallback=None))
ap.add_argument('--gpm-range', help='range of GPM data values to color (auto if not set)',
                action=CoordsSpanArg, default=coords_span(config.get('GPM', 'range', fallback=[None, None])))
ap.add_argument('-p', '--precipitation', help='process precipitation data in GPM files',
                action='store_true', default=config.getboolean('GPM', 'draw precipitation', fallback=False))
#ap.add_argument('-b', '--batch', help='file with the list of files to process in pairs', metavar='FILE',
                #default=config.get('batch', 'file name', fallback=None))       # TODO
args = ap.parse_args()

lats = {'GMI': [np.array([])] * 13, 'GPM': {'MS': np.array([]), 'HS': np.array([]), 'NS': np.array([])}}
lons = {'GMI': [np.array([])] * 13, 'GPM': {'MS': np.array([]), 'HS': np.array([]), 'NS': np.array([])}}
data = {'GMI': [np.array([])] * 13, 'GPM': {'MS': np.array([]), 'HS': np.array([]), 'NS': np.array([])}}
triangulated_coordinates = {'MS': None, 'HS': None, 'NS': None}      # to re-use but not to re-calculate
triangulated_map_coordinates = {'MS': None, 'HS': None, 'NS': None}  # to re-use but not to re-calculate
default_attr = "CodeMissingValue"
description_attr = "LongName"
units_attr = "Units"
description = {'S1': "", 'S2': "", 'MS': "", 'HS': "", 'NS': ""}
units = {'S1': "", 'S2': "", 'MS': "", 'HS': "", 'NS': ""}

def get_attr(a, attr_name, attr_type, default=None):
        if attr_name in a.attrs._v_attrnames:
            attr = a.attrs[attr_name]
            if type(attr) is attr_type:
                return attr
            if isinstance(attr, bytes):
                for i in range(len(attr)-1):
                    try:
                        attr = attr.decode('utf-8')
                    except UnicodeDecodeError:
                        attr = attr[:-1]
                    else:
                        break
            if isinstance(attr, str):
                for i in range(len(attr)-1):
                    try:
                        return attr_type(attr)
                    except ValueError:
                        attr = attr[:-1]
        else:
            return default

def parse_list(s):
    if isinstance(s, bytes):
        try:
            s = s.decode('utf-8')
        except UnicodeDecodeError:
            s = s.decode('ascii')
    l = []
    p = re.compile(r'(?<=\d[\.\)])\s+(?P<item>[^\)]+?)(?:\s*\d+\)|\s*$|\s+and\s+)')
    for i in s.splitlines():
        l += p.findall(i)
    return l

def gearth_fig(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, pixels=1024):
    # https://ocefpaf.github.io/python4oceanographers/blog/2014/03/10/gearth/
    """Return a Matplotlib `fig` and `ax` handles for a Google-Earth Image"""
    aspect = np.cos(np.mean([llcrnrlat, urcrnrlat]) / 180.0 * np.pi)
    xsize = np.ptp([urcrnrlon, llcrnrlon]) * aspect
    ysize = np.ptp([urcrnrlat, llcrnrlat])
    aspect = ysize / xsize

    if aspect > 1.0:
        figsize = (10.0 / aspect, 10.0)
    else:
        figsize = (10.0, 10.0 * aspect)

    #plt.ioff()  # uncomment to prevent the KML components from poping-up
    if pixels < 10:
        pixels = 1024
    fig = plt.figure(figsize=figsize,
                     frameon=False,
                     dpi=pixels//10
                     )
    # KML friendly image
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(llcrnrlon, urcrnrlon)
    ax.set_ylim(llcrnrlat, urcrnrlat)
    return fig, ax

def make_kmlz(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, fig, colorbar=None, **kw):
    # https://ocefpaf.github.io/python4oceanographers/blog/2014/03/10/gearth/
    kml = Kml(name=kw.pop('name', ''))
    altitude = kw.pop('altitude', 1e7)
    roll = kw.pop('roll', 0)
    tilt = kw.pop('tilt', 0)
    altitudemode = kw.pop('altitudemode', AltitudeMode.relativetoground)
    camera = Camera(latitude=np.mean([urcrnrlat, llcrnrlat]),
                    longitude=np.mean([urcrnrlon, llcrnrlon]),
                    altitude=altitude, roll=roll, tilt=tilt,
                    altitudemode=altitudemode
                    )

    kml.document.camera = camera
    ground = kml.newgroundoverlay(name='GroundOverlay')
    ground.draworder = 1
    ground.visibility = kw.pop('visibility', 1)
    ground.name = kml.document.name
    ground.atomauthor = kw.pop('author', '')
    ground.latlonbox.rotation = kw.pop('rotation', 0)
    ground.description = kw.pop('description', '')
    ground.gxaltitudemode = kw.pop('gxaltitudemode', 'clampToSeaFloor')
    ground.icon.href = fig
    ground.latlonbox.east = llcrnrlon
    ground.latlonbox.south = llcrnrlat
    ground.latlonbox.north = urcrnrlat
    ground.latlonbox.west = urcrnrlon

    if colorbar:  # Options for colorbar are hard-coded (to avoid a big mess)
        screen = kml.newscreenoverlay(name='ScreenOverlay')
        screen.icon.href = colorbar
        screen.color = kw.pop('color', '9effffff')
        screen.overlayxy = OverlayXY(x=0, y=0,
                                     xunits=Units.fraction,
                                     yunits=Units.fraction
                                     )
        screen.screenxy = ScreenXY(x=0.015, y=0.085,
                                   xunits=Units.fraction,
                                   yunits=Units.fraction
                                   )
        screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                       xunits=Units.fraction,
                                       yunits=Units.fraction
                                       )
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = 1

    kmzfile = kw.pop('kmzfile', 'kmz.kmz')
    # HOTFIX
    if kmzfile.endswith('.kmz.kmz') or kmzfile.endswith('.kml.kmz'):
        kmzfile = kmzfile[:-4]
    try:
        kml.savekmz(kmzfile)
    except:
        print('Error: failed to save {filename}'.format(filename=kmzfile))

def alpha_shape(points, alpha=None):
    # http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    coords = np.array([point.coords[0] for point in points])
    edges = set()
    edge_points = []
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])
    def circum_radius(pa, pb, pc):
        """
        @param pa, pb, and pc are angles coordinates
        """
        # Lengths of sides of triangle
        a = np.hypot(pa[0]-pb[0], pa[1]-pb[1])
        b = np.hypot(pb[0]-pc[0], pb[1]-pc[1])
        c = np.hypot(pc[0]-pa[0], pc[1]-pa[1])
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        return a * b * c / (4.0 * area)

    tri = Delaunay(coords)
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    if alpha is None or alpha <= 0.:
        crs = np.array([circum_radius(coords[ia], coords[ib], coords[ic]) for ia, ib, ic in tri.vertices])
        alpha = 0.5 / crs.mean()
        del crs
        print('recommended alpha:', alpha)
    for ia, ib, ic in tri.vertices:
        # Here's the radius filter.
        if circum_radius(coords[ia], coords[ib], coords[ic]) < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    del edges
    return cascaded_union(list(polygonize(geometry.MultiLineString(edge_points))))

def triangles(x, y):
    print('calculating patches…')
    mask = np.logical_or(x<1.e20, y<1.e20)
    x = np.compress(mask, x)
    y = np.compress(mask, y)
    del mask
    # https://stackoverflow.com/a/48272157/8554611
    triang = mpl.tri.Triangulation(x, y)
    # create array of points from reduced exp data to convert to Polygon
    crds = np.array([x, y]).transpose()
    # Adjust the length of acceptable sides by adjusting the alpha parameter
    concave_hull = alpha_shape(MultiPoint(crds), alpha=args.alpha)
    del crds
    # Use the mean distance between the triangulated x & y poitns
    x2 = x[triang.triangles].mean(axis=1)
    y2 = y[triang.triangles].mean(axis=1)
    ##note the very obscure `mean` command, which, if not present causes an error.
    ##now we need some masking condition.
    #print('creating mask…')
    # iterate through points checking if the point lies within the polygon
    # apply masking
    triang.set_mask([1 - concave_hull.contains(Point(x, y)) for x, y in zip(x2, y2)])
    del concave_hull
    return triang

f = None
fn = args.GMI
if fn:
    try:
        if not tables.is_hdf5_file(fn):
            print("Error: the file {filename} is not an HDF5 file".format(filename=fn))
        else:
            f = tables.open_file(fn, 'r')
    except OSError:
        print("Error: file '{filename}' can't be read".format(filename=fn))
    else:
        if f and f.isopen:
            for dataset in ['S1', 'S2']:
                if args.frequency and dataset != ('S1' if any(freq < 10 for freq in args.frequency) else 'S2'):
                    continue
                try:
                    g = f.get_node('/' + dataset)
                except tables.exceptions.NoSuchNodeError:
                    print("Error: no such frequency in '{filename}'".format(filename=fn))
                    continue
                try:
                    lat = np.array(f.get_node('/' + dataset, 'Latitude')).ravel()
                    lon = np.array(f.get_node('/' + dataset, 'Longitude')).ravel()
                except tables.exceptions.NoSuchNodeError:
                    print("Error: coordinates data missing in {node} of '{filename}'".format(node=g._v_name, filename=fn))
                    continue
                try:
                    a = f.get_node('/' + dataset, 'Tc')
                except tables.exceptions.NoSuchNodeError:
                    print("Error: temperature data missing in {node} of '{filename}'".format(node=g._v_name, filename=fn))
                    continue
                default = get_attr(a, default_attr, float, -9999.9)
                units[dataset] = get_attr(a, units_attr, str, "")
                description[dataset] = get_attr(a, description_attr, str, "")
                tnpa = np.transpose(np.array(a))
                # removing unnecessary data
                discarded_items_indeces = []
                for i, l in enumerate(lat):
                    if   args.latitude[0] < args.latitude[1] and not (args.latitude[0] <= l <= args.latitude[1]):
                        discarded_items_indeces += [i]
                    elif args.latitude[0] > args.latitude[1] and      args.latitude[0] >= l >= args.latitude[1] :
                        discarded_items_indeces += [i]
                for i, l in enumerate(lon):
                    if   args.longitude[0] < args.longitude[1] and not (args.longitude[0] <= l <= args.longitude[1]):
                        discarded_items_indeces += [i]
                    elif args.longitude[0] > args.longitude[1] and      args.longitude[0] >= l >= args.longitude[1] :
                        discarded_items_indeces += [i]
                for frequency in ([freq - 1 for freq in args.frequency] if args.frequency else range(tnpa.shape[0])):
                    data_piece = np.transpose(tnpa[frequency % 9]).ravel()
                    # removing corrupted data
                    corrupted_items_indeces = []
                    for i, d in enumerate(data_piece):
                        if d == default:
                            corrupted_items_indeces += [i]
                    lats['GMI'][frequency] = np.append(lats['GMI'][frequency], np.delete(lat, corrupted_items_indeces + discarded_items_indeces))
                    lons['GMI'][frequency] = np.append(lons['GMI'][frequency], np.delete(lon, corrupted_items_indeces + discarded_items_indeces))
                    data['GMI'][frequency] = np.append(data['GMI'][frequency], np.delete(data_piece, corrupted_items_indeces + discarded_items_indeces))
                    del corrupted_items_indeces
                    print('data read for frequency {frequency} of file {filename}'.format(frequency=frequency+1, filename=fn))
                del discarded_items_indeces, tnpa
                gc.collect()
            f.close()

f = None
fn = args.GPM
if fn:
    try:
        if not tables.is_hdf5_file(fn):
            print("Error: the file {filename} is not an HDF5 file".format(filename=fn))
        else:
            f = tables.open_file(fn, 'r')
    except OSError:
        print("Error: file '{filename}' can't be read".format(filename=fn))
    else:
        if f and f.isopen:
            for dataset in ['HS', 'NS']: # and/or 'MS'
                try:
                    g = f.get_node('/' + dataset)
                except tables.exceptions.NoSuchNodeError:
                    continue
                try:
                    lat = np.array(f.get_node('/' + dataset, 'Latitude')).ravel()
                    lon = np.array(f.get_node('/' + dataset, 'Longitude')).ravel()
                except tables.exceptions.NoSuchNodeError:
                    print("Error: coordinates data missing in {dataset} of '{filename}'".format(dataset=dataset, filename=fn))
                    continue
                try:
                    if args.precipitation:
                        a = f.get_node('/' + dataset, 'SLV/precipRateNearSurface')
                    else:
                        a = f.get_node('/' + dataset, 'VER/sigmaZeroNPCorrected')
                except tables.exceptions.NoSuchNodeError:
                    print("Error: data missing in {dataset} of '{filename}'".format(dataset=dataset, filename=fn))
                    continue
                default = get_attr(a, default_attr, float)
                units[dataset] = get_attr(a, units_attr, str, "")
                description[dataset] = get_attr(a, description_attr, str, "")
                # removing unnecessary data
                discarded_items_indices = []
                for i, l in enumerate(lat):
                    if   args.latitude[0] < args.latitude[1] and not (args.latitude[0] <= l <= args.latitude[1]):
                        discarded_items_indices += [i]
                    elif args.latitude[0] > args.latitude[1] and      args.latitude[0] >= l >= args.latitude[1] :
                        discarded_items_indices += [i]
                for i, l in enumerate(lon):
                    if   args.longitude[0] < args.longitude[1] and not (args.longitude[0] <= l <= args.longitude[1]):
                        discarded_items_indices += [i]
                    elif args.longitude[0] > args.longitude[1] and      args.longitude[0] >= l >= args.longitude[1] :
                        discarded_items_indices += [i]
                data_piece = np.array(a).ravel()
                # removing corrupted data
                corrupted_items_indices = []
                for i, d in enumerate(data_piece):
                    if d == default:
                        corrupted_items_indeces += [i]
                lats['GPM'][dataset] = np.append(lats['GPM'][dataset], np.delete(lat, \
                        corrupted_items_indices + discarded_items_indices))
                lons['GPM'][dataset] = np.append(lons['GPM'][dataset], np.delete(lon, \
                        corrupted_items_indices + discarded_items_indices))
                data['GPM'][dataset] = np.append(data['GPM'][dataset], np.delete(data_piece, \
                        corrupted_items_indices + discarded_items_indices))
                del corrupted_items_indices, discarded_items_indices, lon, lat, data_piece
                print('data read for dataset {dataset} of file {filename}'.format(dataset=dataset, filename=fn))
                gc.collect()
            f.close()

if not args.save_kml:
    m = Basemap(projection='merc',
                llcrnrlat=args.latitude[0],  urcrnrlat=args.latitude[1],
                llcrnrlon=args.longitude[0], urcrnrlon=args.longitude[1],
                resolution='i' # None → 'c' → 'l' → 'i' → 'h' → 'f'
                )
print('map built')
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred'], N=256)
cmap.set_under('darkblue')
cmap.set_over('darkred')
parallels = np.arange(0., 90., 5.)
meridians = np.arange(0., 360., 10.)
if args.GMI:
    for frequency in ([freq - 1 for freq in args.frequency] if args.frequency else range(13)):
        if data['GMI'][frequency].size > 0:
            dataset = 'S1' if frequency < 9 else 'S2'
            print('processing data for frequency {frequency} ({value})'.\
                    format(frequency=frequency+1, value=parse_list(description[dataset])[frequency % 9]))
            if not args.save_kml or args.save_image:
                # draw coastlines, state and country boundaries, edge of map
                m.drawcoastlines()
                #m.drawstates()
                #m.drawcountries()
                m.drawparallels(parallels, labels=[1, 0, 0, 0])
                m.drawmeridians(meridians, labels=[0, 0, 0, 1])
                # compute map proj coordinates
                x, y = m(lons['GMI'][frequency], lats['GMI'][frequency])
                ## clear data necessary no further
                lons['GMI'][frequency] = np.array([])
                lats['GMI'][frequency] = np.array([])

                # draw filled contours
                print('drawing plot…')
                if args.gmi_range[0] is not None and args.gmi_range[1] is not None and args.gmi_range[0] != args.gmi_range[1]:
                    levels = np.linspace(min(args.gmi_range[0], args.gmi_range[1]), max(args.gmi_range[0], args.gmi_range[1]), 256)
                else:
                    levels = 256
                cs_gmi = plt.tricontourf(triangles(x, y),
                                         data['GMI'][frequency],
                                         levels,
                                         cmap=cmap
                                         )
                zorder = 1
                cs_gpm = None
                for ds in ['MS', 'HS', 'NS']:
                    if data['GPM'][ds].size > 0:
                        x, y = m(lons['GPM'][ds], lats['GPM'][ds])
                        lons['GPM'][ds] = np.array([])
                        lats['GPM'][ds] = np.array([])
                        if not triangulated_map_coordinates[ds]:
                            triangulated_map_coordinates[ds] = triangles(x, y)
                        if args.gpm_range[0] is not None and args.gpm_range[1] is not None and args.gpm_range[0] != args.gpm_range[1]:
                            levels = np.linspace(min(args.gpm_range[0], args.gpm_range[1]), max(args.gpm_range[0], args.gpm_range[1]), 256)
                        else:
                            levels = 256
                        cs_gpm = plt.tricontourf(triangulated_map_coordinates[ds],
                                                 data['GPM'][ds],
                                                 levels,
                                                 cmap=cmap,
                                                 zorder=zorder
                                                 )
                        if not args.no_colorbar:
                            # add colorbar
                            cbar = plt.colorbar(cs_gpm, format='%d')
                            label = ''
                            if args.precipitation:
                                label = 'Precipitation rate'
                            else:
                                label = 'Backscattering cross section'
                            if units[ds]:
                                label += ' [{unit}]'.format(unit=units[ds])
                            cbar.set_label(label)
                        zorder += 1

                if not args.no_colorbar:
                    # add colorbar
                    cbar = plt.colorbar(cs_gmi, format='%d')
                    label = 'Brightness temperature'
                    if units[dataset]:
                        label += ' [{unit}]'.format(unit=units[dataset])
                    cbar.set_label(label)
                del x, y
                # add title
                plt.title(parse_list(description[dataset])[frequency % 9])
            if args.save_kml:
                # draw filled contours
                map_fig, ax = gearth_fig(llcrnrlat=args.latitude[0],  urcrnrlat=args.latitude[1],
                                        llcrnrlon=args.longitude[0], urcrnrlon=args.longitude[1],
                                        pixels=args.resolution
                                        )
                if args.gmi_range[0] is not None and args.gmi_range[1] is not None and args.gmi_range[0] != args.gmi_range[1]:
                    levels = np.linspace(min(args.gmi_range[0], args.gmi_range[1]), max(args.gmi_range[0], args.gmi_range[1]), 256)
                else:
                    levels = 256
                cs_gmi = ax.tricontourf(triangles(lons['GMI'][frequency], lats['GMI'][frequency]),
                                        data['GMI'][frequency],
                                        levels,
                                        cmap=cmap
                                        )
                lons['GMI'][frequency] = np.array([])
                lats['GMI'][frequency] = np.array([])
                zorder = 1
                cs_gpm = None
                if not args.no_colorbar:
                    # add colorbar
                    leg_fig = plt.figure(facecolor=None)
                for ds in ['MS', 'HS', 'NS']:
                    if data['GPM'][ds].size > 0:
                        if not triangulated_coordinates[ds]:
                            triangulated_coordinates[ds] = triangles(lons['GPM'][ds], lats['GPM'][ds])
                        lons['GPM'][ds] = np.array([])
                        lats['GPM'][ds] = np.array([])
                        if args.gpm_range[0] is not None and args.gpm_range[1] is not None and args.gpm_range[0] != args.gpm_range[1]:
                            levels = np.linspace(min(args.gpm_range[0], args.gpm_range[1]), max(args.gpm_range[0], args.gpm_range[1]), 256)
                        else:
                            levels = 256
                        cs_gpm = ax.tricontourf(triangulated_coordinates[ds],
                                                data['GPM'][ds],
                                                levels,
                                                cmap=cmap,
                                                zorder=zorder
                                                )
                        if not args.no_colorbar:
                            # add colorbar
                            cbar = leg_fig.colorbar(cs_gpm, format='%d')
                            label = ''
                            if args.precipitation:
                                label = 'Precipitation rate'
                            else:
                                label = 'Backscattering cross section'
                            if units[ds]:
                                label += ' [{unit}]'.format(unit=units[ds])
                            cbar.set_label(label)
                        zorder += 1
                ax.set_axis_off()
                if not args.no_colorbar:
                    # add colorbar
                    cbar = leg_fig.colorbar(cs_gmi, orientation='vertical', format='%d')
                    leg_fig.gca().set_visible(False)
                    label = 'Brightness temperature'
                    if units[ds]:
                        label += ' [{unit}]'.format(unit=units[dataset])
                    cbar.set_label(label)
            if not args.no_output:
                if args.save_image:
                    print('saving image…', end='\r')
                    if args.save_image.endswith('.{format}'.format(format=args.image_format)):
                        imgfilename = '{initfn}.{frequency}.{format}'.format(initfn=args.save_image[:-len(args.image_format)-1],
                                                                            frequency=frequency+1,
                                                                            format=args.image_format)
                    else:
                        imgfilename = '{initfn}.{frequency}.{format}'.format(initfn=args.save_image,
                                                                            frequency=frequency+1,
                                                                            format=args.image_format)
                    try:
                        plt.savefig(imgfilename,
                                    transparent=args.image_transparent,
                                    format=args.image_format,
                                    bbox_inches='tight' if args.tight_image else None,
                                    dpi=args.image_dpi if args.image_dpi > 16 else None
                                    )
                    except:
                        print('Error: failed to save an image as {filename}'.format(filename=imgfilename))
                    else:
                        print('image saved as {filename}'.format(filename=imgfilename))
                if args.save_kml:
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        print('creating heatmap...')
                        # TODO: optionally, create separate overlays for the input files
                        pltfilename = 'overlay.png'
                        pltfilename = os.path.join(tmpdirname, pltfilename)
                        map_fig.savefig(pltfilename, transparent=True, frameon=False, format='png')
                        if not args.no_colorbar:
                            legfilename = 'legend.png'
                            legfilename = os.path.join(tmpdirname, legfilename)
                            leg_fig.savefig(legfilename, transparent=False, frameon=False, format='png', bbox_inches='tight')
                        else:
                            legfilename = None
                        # TODO: strip extension if set in the argument
                        kmzfilename = '{filename}.{frequency}.kmz'.format(frequency=frequency+1, filename=args.save_kml)
                        print('saving the result as {filename}'.format(filename=kmzfilename))
                        make_kmlz(llcrnrlat=args.latitude[0],  urcrnrlat=args.latitude[1],
                                llcrnrlon=args.longitude[0], urcrnrlon=args.longitude[1],
                                fig=pltfilename,
                                colorbar=legfilename,
                                kmzfile=kmzfilename,
                                name=parse_list(description[dataset])[frequency % 9]
                                )
                elif not args.save_image:
                    plt.show()
            plt.close('all')
            if not args.no_colorbar:
                del cbar
            if args.GPM:
                del cs_gpm
            del cs_gmi
            data['GMI'][frequency] = np.array([])
            gc.collect()

elif args.GPM:
    if not args.save_kml or args.save_image:
        # draw coastlines, state and country boundaries, edge of map
        m.drawcoastlines()
        #m.drawstates()
        #m.drawcountries()
        m.drawparallels(parallels, labels=[1, 0, 0, 0])
        m.drawmeridians(meridians, labels=[0, 0, 0, 1])

        # draw filled contours
        print('drawing plot…')
        zorder = 0
        cs_gpm = None
        for ds in ['MS', 'HS', 'NS']:
            if data['GPM'][ds].size > 0:
                x, y = m(lons['GPM'][ds], lats['GPM'][ds])
                if args.gpm_range[0] is not None and args.gpm_range[1] is not None and args.gpm_range[0] != args.gpm_range[1]:
                    levels = np.linspace(min(args.gpm_range[0], args.gpm_range[1]), max(args.gpm_range[0], args.gpm_range[1]), 256)
                else:
                    levels = 256
                cs_gpm = plt.tricontourf(triangles(x, y),
                                         data['GPM'][ds],
                                         levels,
                                         cmap=cmap,
                                         zorder=zorder
                                         )
                if not args.no_colorbar:
                    # add colorbar
                    format = '%d'
                    label = ''
                    if args.precipitation:
                        label = 'Precipitation rate'
                        format = None
                    else:
                        label = 'Backscattering cross section'
                    if units[ds]:
                        label += ' [{unit}]'.format(unit=units[ds])
                    cbar = plt.colorbar(cs_gpm, format=format)
                    cbar.set_label(label)
                zorder += 1
                del x, y
    if args.save_kml:
        # draw filled contours
        map_fig, ax = gearth_fig(llcrnrlat=args.latitude[0],  urcrnrlat=args.latitude[1],
                                 llcrnrlon=args.longitude[0], urcrnrlon=args.longitude[1],
                                 pixels=args.resolution
                                 )
        zorder = 0
        cs_gpm = None
        if not args.no_colorbar:
            # add colorbar
            leg_fig = plt.figure(facecolor=None)
        for ds in ['MS', 'HS', 'NS']:
            if data['GPM'][ds].size > 0:
                if args.gpm_range[0] is not None and args.gpm_range[1] is not None and args.gpm_range[0] != args.gpm_range[1]:
                    levels = np.linspace(min(args.gpm_range[0], args.gpm_range[1]), max(args.gpm_range[0], args.gpm_range[1]), 256)
                else:
                    levels = 256
                cs_gpm = ax.tricontourf(triangles(lons['GPM'][ds], lats['GPM'][ds]),
                               data['GPM'][ds],
                               levels,
                               cmap=cmap,
                               zorder=zorder
                               )
                if not args.no_colorbar:
                    # add colorbar
                    cbar = leg_fig.colorbar(cs_gpm, format='%d')
                    label = ''
                    if args.precipitation:
                        label = 'Precipitation rate'
                    else:
                        label = 'Backscattering cross section'
                    if units[ds]:
                        label += ' [{unit}]'.format(unit=units[ds])
                    cbar.set_label(label)
                zorder += 1
        ax.set_axis_off()
    if not args.no_output:
        if args.save_image:
            print('saving image…', end='\r')
            if args.save_image.endswith('.{format}'.format(format=args.image_format)):
                imgfilename = args.save_image
            else:
                imgfilename = '{initfn}.{format}'.format(initfn=args.save_image,
                                                         format=args.image_format)
            try:
                plt.savefig(imgfilename,
                            transparent=args.image_transparent,
                            format=args.image_format,
                            bbox_inches='tight' if args.tight_image else None,
                            dpi=args.image_dpi if args.image_dpi > 16 else None
                            )
            except:
                print('Error: failed to save an image as {filename}'.format(filename=imgfilename))
            else:
                print('image saved as {filename}'.format(filename=imgfilename))
        if args.save_kml:
            with tempfile.TemporaryDirectory() as tmpdirname:
                print('creating heatmap...')
                # TODO: optionally, create separate overlays for the input files
                pltfilename = 'overlay.png'
                pltfilename = os.path.join(tmpdirname, pltfilename)
                map_fig.savefig(pltfilename, transparent=True, frameon=False, format='png')
                # TODO: strip extension if set as the argument
                kmzfilename = '{filename}.kmz'.format(filename=args.savekml)
                print('saving the result as {filename}'.format(filename=kmzfilename))
                make_kmlz(llcrnrlat=args.latitude[0],  urcrnrlat=args.latitude[1],
                          llcrnrlon=args.longitude[0], urcrnrlon=args.longitude[1],
                          fig=pltfilename,
                          kmzfile=kmzfilename,
                          #name=parse_list(description[dataset])[frequency % 9]
                          )
        elif not args.save_image:
            plt.show()
    plt.close('all')

print('done')
