import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from novatel_gps_msgs.msg import Inspvax

import numpy as np
import pickle
import math
import struct
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

map = 2
if map == 1:
    position0 =[44.26262793084365, -76.51626692489744, 359.1112573857645, 90.78792007733136] #lat, long, azimuth, altitude
    pickle_filename = "map1.pkl"
elif map ==2:
    position0 =[44.2627483711262, -76.5163410045937, 88.12346606005315, 90.47871469985694] #lat, long, azimuth, altitude
    pickle_filename = "map2.pkl"
elif map ==3:
    position0 =[44.26277117948086, -76.51647591060944, 266.9308659366506, 90.25321607664227] #lat, long, azimuth, altitude
    pickle_filename = "map3.pkl"



position = [0,0,0,0]
count = 0

pc = [0,0,0,0]

pcd = o3d.geometry.PointCloud()

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.
    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

def get_displacement(origin, current):
    R = 6378.137 # Radius of earth in KM
    dLat = 1000*R*(current[0] * np.pi / 180 - origin[0] * np.pi / 180)
    dLon = 1000*R*(current[1] * np.pi / 180 - origin[1] * np.pi / 180)
    az = current[2] * np.pi / 180
    # az = (az + np.pi) % (2 * np.pi) - np.pi
    return [dLat, dLon, az, current[3]]

class Map_Sub(Node):
    def __init__(self):
        super().__init__('map')
        self.subscription = self.create_subscription(Inspvax,'inspvax',self.gnss_callback,10)
        self.subscription = self.create_subscription(PointCloud2,'cepton_pcl2',self.pc_callback,10)
        self.subscription  # prevent unused variable warning

    def gnss_callback(self, msg):
        global position
        position = [msg.latitude, msg.longitude, msg.azimuth, msg.altitude]


    def pc_callback(self, msg):
        disp = get_displacement(position0, position)
        rotation = np.array([
        [np.cos(disp[2]), -np.sin(disp[2]), 0],
        [np.sin(disp[2]), np.cos(disp[2]), 0],
        [0, 0, 1]])
        global pc
        global count
        count +=1
        print(count)


        if (count % 4 == 0):
            pcd_npi = np.array(list(read_points(msg)))
            xi = pcd_npi[:,0:1]
            yi = pcd_npi[:,1:2]
            zi = pcd_npi[:,2:3]
            intensity = pcd_npi[:,3:4]
            norm_intensity = np.amax(intensity)/2  *intensity*  255 / (np.amax(intensity) - np.amin(intensity))

            xlims = [-7.5,7.5]
            ylims = [-5,15]
            zlims = [-10,0]
            # intensity_lims = [10, 100]
            filtered_indices = np.where(
                (xi >= xlims[0]) & (xi <= xlims[1]) &
                (yi >= ylims[0]) & (yi <= ylims[1]) & 
                (zi >= zlims[0]) & (zi <= zlims[1]))# & 
                # (intensity >= intensity_lims[0]) & (intensity <= intensity_lims[1]))
            pointsi = np.array([xi, yi, zi])[:, filtered_indices[0]]   
            intensity = intensity[filtered_indices[0]]
            pointsi = np.squeeze(pointsi, axis=-1)
            pointsi = np.matmul(pointsi.T, rotation)
            pointsi = pointsi + np.array([disp[1],disp[0],disp[3]])
            pointsi = np.concatenate((pointsi, intensity), axis=1)
            pc = np.vstack((pc,pointsi))

            if (count % 100 < 4):
                print("pickle")
                with open(pickle_filename, 'wb') as file:

                    pickle.dump(pc, file)
            #     # import pdb;pdb.set_trace()
            #     pcd.points = o3d.utility.Vector3dVector(pc)
            #     o3d.visualization.draw_geometries([pcd])

    #  pcd_npi = np.array(list(read_points(msg)))
    #     xi = pcd_npi[:,0:1]
    #     yi = pcd_npi[:,1:2]
    #     zi = pcd_npi[:,2:3]

  
    #     #transform pc
    #     # import pdb;pdb.set_trace()
    #     pointsi = pointsi + np.array([disp[1],disp[0],0.0])
    #     rotation = Rotation.from_euler('z', disp[2], degrees=False)
    #     pointsi = rotation.apply(pointsi)

def main(args=None):
    rclpy.init(args=args)

    map_sub = Map_Sub()

    rclpy.spin(map_sub)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    map_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()