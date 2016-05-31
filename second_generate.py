import yaml
import sys
from base64 import b64decode
import zlib
import struct
import json
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
import random
from sympy import Point, Point3D, Line, Line3D, Plane

class Observation:
    observationCount = 0
    occupancyCount = 0
    occupancyConfidence = 0.0
    r = 0.0
    g = 0.0
    b = 0.0

class Observation2:
    countRGB = 0.0
    countA = 0.0
    r = 0.0
    g = 0.0
    b = 0.0
    a = 0.0

    def __init__(self):
        return

class GaussianMapChannel:
    binarySize = 40
    unpacker = struct.Struct('ddddd')

    @staticmethod
    def fromBinary(data):
        assert len(data) == GaussianMapChannel.binarySize, (len(data), GaussianMapChannel.binarySize)
        unpackedTuple = GaussianMapChannel.unpacker.unpack(data)
        #print "tuple", unpackedTuple
        return GaussianMapChannel(*unpackedTuple)

    def __init__(self, counts, squaredcounts, mu, sigmasquared, samples):
        self.counts = counts
        self.squaredcounts = squaredcounts
        self.mu = mu
        self.sigmasquared = sigmasquared
        self.samples = samples
    def __repr__(self):
        return "GaussianMapChannel(%.10f, %.10f, %.10f, %.10f, %.10f)" % (self.counts, self.squaredcounts,
                                         self.mu, self.sigmasquared,
                                         self.samples)

class GaussianMapCell:
    binarySize = GaussianMapChannel.binarySize * 4

    @staticmethod
    def fromBinary(data):
        assert len(data) == GaussianMapCell.binarySize
        return GaussianMapCell(GaussianMapChannel.fromBinary(data[0:GaussianMapChannel.binarySize]),
                               GaussianMapChannel.fromBinary(data[GaussianMapChannel.binarySize:GaussianMapChannel.binarySize * 2]),
                               GaussianMapChannel.fromBinary(data[GaussianMapChannel.binarySize * 2:GaussianMapChannel.binarySize * 3]),
                               GaussianMapChannel.fromBinary(data[GaussianMapChannel.binarySize * 3:GaussianMapChannel.binarySize * 4]))

    def __init__(self, red, green, blue, z):
        self.red = red
        self.green = green 
        self.blue = blue
        self.z = z
    def __repr__(self):
        return "GaussianMapCell(%s, %s, %s, %s)" % (repr(self.red), repr(self.green), repr(self.blue), repr(self.z))

class GaussianMap:
    @staticmethod
    def fromYaml(yamlGmap):


        cellData = readBinaryFromYaml(yamlGmap["cells"])
        print "cellData", len(cellData)

        expectedSize = yamlGmap["width"] * yamlGmap["height"] * GaussianMapCell.binarySize
        assert len(cellData) == expectedSize, (len(cellData), expectedSize)
        
        cells = []
        for i in range(yamlGmap["width"] * yamlGmap["height"]):
            cell = GaussianMapCell.fromBinary(cellData[i * GaussianMapCell.binarySize:(i + 1) * GaussianMapCell.binarySize])
            cells.append(cell)
        return GaussianMap(yamlGmap["width"], yamlGmap["height"],
                           yamlGmap["x_center_cell"], yamlGmap["y_center_cell"],
                           yamlGmap["cell_width"], cells)

    def __init__(self, width, height, x_center_cell, y_center_cell, cell_width, cells):
        self.width = width
        self.height = height
        self.x_center_cell = x_center_cell
        self.y_center_cell = y_center_cell
        self.cell_width = cell_width
        self.cells = cells
        



def readBinaryFromYaml(yamlList):
    data = "".join(yamlList)
    decoded = b64decode(data)
    decompressed = zlib.decompress(decoded)
    return decompressed

#hit_points = []
#old_planes = []
#new_planes = []

doubleunpacker = struct.Struct('d')
uintunpacker = struct.Struct('I')
def readMatFromYaml(fs):
    rows = fs["rows"]
    cols = fs["cols"]
    imgtype = fs["type"]
    print "type", imgtype
    m = cv2.cv.CreateMat(rows, cols, imgtype)
    
    if imgtype == 6:
        numpytype = na.float32
        unpacker = doubleunpacker
        #numpytype = na.uint32
        #unpacker = uintunpacker
    else:
        raise ValueError("Unknown image type: " + repr(imgtype))
    array = na.zeros(m.rows * m.cols * m.channels, dtype=numpytype)
    binary = readBinaryFromYaml(fs["data"])
    size = unpacker.size
    for i in range(len(array)):
        data = unpacker.unpack(binary[i*size:(i+1)*size])
        assert len(data) == 1
        array[i] = data[0]
    array = na.transpose(array.reshape((m.rows, m.cols, m.channels)), axes=(1,0,2))
    return array


##############################################################################################
##############################################################################################
#############                           CUSTOM FUNCTIONS                        ##############
##############################################################################################
##############################################################################################
def variance_filter_rgb(r_var, g_var, b_var, r, g, b):
    if (r_var-r)*(r_var-r)+(g_var-g)*(g_var-g)+(b_var-b)*(b_var-b) > 100:
        return False
    return True


def variance_filter_z(z_var, z):
    if (z_var-z)*(z_var-z) > 200:
        return False
    return True

# given a cube/3D grid of confidence interval and a slug from a single perspective, returns a hashtable of info 
# INPUT 
# filename : name of the slug yml file
# max_length : the length of cube to which we are mapping confidence score 
# this input was added to prevent large values of Z for slug scanning from the side(there is no table so we need to limit max z value)
#
# OUTPUT
# returns a hashtable with keys max_x, max_y, max_z, min_x, min_y, min_z, and position info
def get_slug_info(filename, max_length):
    info = {}
    f = open(filename) 
    lines = []
    f.readline() 

    pose_info = ""
    while True:
        c = f.read(1)
        pose_info = pose_info + c
        if c == "}":
            break   

    background_pose = pose_info.split('{')[1].split('}')[0]
    background_pose = background_pose.split(",")
    
    for i in range(len(background_pose)):
        background_pose[i] = background_pose[i].strip()


    start_line = pose_info.split("background_pose")[0].split("\n")
    for line in start_line:
        lines.append(line)

    for line in f:
        if "background_pose" in line:
            continue
        lines.append(line)


    data = "\n".join(lines)
    ymlobject = yaml.load(data)
    scene = ymlobject["Scene"]
    observed_map = GaussianMap.fromYaml(scene["observed_map"])

    row = observed_map.height
    col = observed_map.width
    cell_width = observed_map.cell_width
    width_len = observed_map.width*cell_width
    height_len = observed_map.height*cell_width

    x_max = 0
    y_max = 0
    y_min = float('inf')
    x_min = float('inf')
    z_min = float('inf')
    z_max = 0
    x_avg = 0
    y_avg = 0
    z_avg = 0
    count = 0

    info['position'] = { 'x' : float(background_pose[0].split(':')[1]), 'y' : float(background_pose[1].split(':')[1]), 
                        'z': float(background_pose[2].split(':')[1]), 'qw' : float(background_pose[3].split(':')[1]), 
                        'qx': float(background_pose[4].split(':')[1]), 'qy':  float(background_pose[5].split(':')[1]), 
                        'qz':  float(background_pose[6].split(':')[1])}

    for x in range((-1 * col / 2), (col / 2)):
        for y in range((-1 * row / 2), (row / 2)):
            index = (x + (col / 2)) + col * (y + (row / 2));
            cell = observed_map.cells[index]
            z_mu = float(observed_map.cells[index].z.mu)

            if z_mu > 0 and z_mu < max_length:
                #ray info
                ray_origin = get_ray_origin(info, x, y, cell_width)
                ray_direction = get_ray_direction(info)
                ray_x = ray_origin['x']
                ray_y = ray_origin['y']
                ray_z = ray_origin['z']
                direction_x = ray_direction['x']
                direction_y = ray_direction['y']
                direction_z = ray_direction['z']
                
                final_x = ray_x + direction_x * z_mu
                final_y = ray_y + direction_y * z_mu
                final_z = ray_z + direction_z * z_mu

                x_avg = x_avg + final_x
                y_avg = y_avg + final_y
                z_avg = z_avg + final_z
                count = count + 1

                if final_x > x_max:
                    x_max = final_x
                if final_x < x_min:
                    x_min = final_x

                if final_y > y_max:
                    y_max = final_y
                if final_y < y_min:
                    y_min = final_y
                
                if final_z < z_min:
                    z_min = final_z
                if final_z > z_max:
                    z_max = final_z


    info['cell_len'] = cell_width
    info['rows'] = row
    info['cols'] = col
    info['x_max'] = round(x_max, 3)
    info['x_min'] = round(x_min, 3)
    info['y_max'] = round(y_max, 3)
    info['y_min'] = round(y_min, 3)
    info['z_max'] = round(z_max, 3)
    info['z_min'] = round(z_min, 3)
    info['x_avg'] = round(float(x_avg/count), 3)
    info['y_avg'] = round(float(y_avg/count), 3)
    info['z_avg'] = round(float(z_avg/count), 3)
    

    return info

def get_view_info(filename):
    info = {}
    f = open(filename) 
    lines = []
    f.readline() 

    pose_info = ""
    while True:
        c = f.read(1)
        pose_info = pose_info + c
        if c == "}":
            break   

    background_pose = pose_info.split('{')[1].split('}')[0]
    background_pose = background_pose.split(",")
    
    for i in range(len(background_pose)):
        background_pose[i] = background_pose[i].strip()

    
    info['position'] = { 'x' :  float(background_pose[0].split(':')[1]), 'y' : float(background_pose[1].split(':')[1]), 
                        'z': float(background_pose[2].split(':')[1]), 'qw' : float(background_pose[3].split(':')[1]), 
                        'qx': float(background_pose[4].split(':')[1]), 'qy': float(background_pose[5].split(':')[1]), 
                        'qz': float(background_pose[6].split(':')[1])}


    return info



def get_info_from_top_view(file_name):
    return get_slug_info(file_name, float('inf'))



def get_old_ray_origin(slug_info, x, y, cell_length):
        
    default_pos = slug_info['position']

    rotation_matrix = quaternion_to_rotation_matrix(slug_info['position']['qw'],
                                                    slug_info['position']['qx'], 
                                                    slug_info['position']['qy'], 
                                                    slug_info['position']['qz'])

    #rotation matrix  has property  inverse = transpose
    vector = np.array([x*cell_length, y*cell_length, 0])
    current_vector = np.dot(rotation_matrix, vector)
    

    x = round(float(current_vector[0] + default_pos['x']), 3)
    y = round(float(current_vector[1] + default_pos['y']), 3)
    z = round(float(current_vector[2] + default_pos['z']), 3)

    return {'x' : x, 'y':  y  , 'z' : z}





def get_ray_origin(slug_info, x, y, cell_length):
    z_max = 0.388
    default_pos = slug_info['position']

    rotation_matrix = quaternion_to_rotation_matrix(slug_info['position']['qw'],
                                                    slug_info['position']['qx'], 
                                                    slug_info['position']['qy'], 
                                                    slug_info['position']['qz'])

    #rotation matrix  has property  inverse = transpose
    vector = np.array([x*cell_length, y*cell_length, z_max])
    current_vector = np.dot(rotation_matrix, vector)
    

    x = round(float(current_vector[0] + default_pos['x']), 3)
    y = round(float(current_vector[1] + default_pos['y']), 3)
    z = round(float(current_vector[2] + default_pos['z']), 3)
    return {'x' : x, 'y':  y  , 'z' : z}


#Given a yml file, read all the data, put the confidence score in the sparse map and return the updated sparse map
#INPUT : info hashtable obtained from get_slug_info(), yml file name, and sparse map hashtable 
#
#OUTPUT : updated sparse hash map 
def read_from_yml(file_name, sparse_map, slug_info, cube_info):
    f = open(file_name) 

    lines = []
    f.readline() 

    for line in f:
        if "background_pose" in line:
            continue
        lines.append(line)

    data = "\n".join(lines)

    ymlobject = yaml.load(data)
    scene = ymlobject["Scene"]
    observed_map = GaussianMap.fromYaml(scene["observed_map"])

    row = observed_map.height
    col = observed_map.width

    cell_length = observed_map.cell_width
    ray_direction = get_ray_direction(slug_info)

    print " ====== starting ray casting ========"

    for x in range((-1 * col / 2), (col / 2)):
        for y in range((-1 * row / 2), (row / 2)):
            index = (x + (col / 2)) + col * (y + (row / 2));
            cell = observed_map.cells[index]
            r = float(cell.red.mu)
            g = float(cell.green.mu)
            b = float(cell.blue.mu)
            # if g != 0.0 and r == 0.0 and b == 0.0:
            #     print "***********************************************"
            #     print cell.red.mu
            #     print cell.green.mu
            #     print cell.blue.mu

            # the max for giraffe top down is 0.46596
            z_mu = float(cell.z.mu)

            #old_ray_origin = get_old_ray_origin(slug_info, x, y, cell_length)
            ray_origin = get_ray_origin(slug_info, x, y, cell_length)
            
            #if x % 14 == 0 and y % 14 == 0:
            #    data = { 'x' : ray_origin['x'] , 'y' : ray_origin['y'] , 'z' : ray_origin['z'], 'plane' : True}
            #    new_planes.append(data)

            #    new_data = {"x" : old_ray_origin['x'], 'y' : old_ray_origin['y'], 'z' : old_ray_origin['z']}
            #    old_planes.append(new_data)

            #print "z : :" + str(z_mu) + " insied the loop !"
            if z_mu > 0:  
                sparse_map = ray_cast(sparse_map, ray_origin, ray_direction, z_mu, cube_info, r, g, b)


    print "===== end of ray casting ========= "
    print "length of sparse map is " + str(len(sparse_map))

    return sparse_map


# Multiply 2 quaternions, r and q, together
# INPUT - r : {qx, qy, qz, qw}, q: {qx, qy, qz, qw}
# OUTPUT - {qx, qy, qz, qw}
def multiply_quaternion(r, q):
    t0 = r['qw']*q['qw'] - r['qx']*q['qx'] - r['qy']*q['qy'] - r['qz']*q['qz']
    t1 = r['qw']*q['qx'] + r['qx']*q['qw'] - r['qy']*q['qz'] + r['qz']*q['qy']
    t2 = r['qw']*q['qy'] + r['qx']*q['qz'] + r['qy']*q['qw'] - r['qz']*q['qx'] 
    t3 = r['qw']*q['qz'] - r['qx']*q['qy'] + r['qy']*q['qx'] + r['qz']*q['qw'] 
    return {'qw': t0, 'qx': t1, 'qy': t2, 'qz': t3}

def get_ray_direction(slug_info):
    quaternion = slug_info['position']
    quaternion = multiply_quaternion(quaternion , {"qx": 0, "qy" : 1, "qz" : 0, "qw" : 0})
    qw = quaternion['qw']
    qx = quaternion['qx']
    qy = quaternion['qy']
    qz = quaternion['qz']

    rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    direction_vector = np.dot(rotation_matrix, np.array([0, 0, 1]))

    return {'x': direction_vector[0], 'y': direction_vector[1],'z': direction_vector[2]}



def encode_key(x, y, z):
    return str(int(x)) + "_" + str(int(y)) + "_" + str(int(z))


def decode_key(key):
    temp = key.split('_')
    return {'x': (int(temp[0])) , 'y': (int(temp[1])), 'z':(int(temp[2]))} 


#returns rotation matrix M from quaterniion
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    m = np.zeros((3, 3))
    m[0][0] = 1- 2*qy*qy - 2*qz*qz
    m[0][1] = 2*qx*qy - 2*qw*qz
    m[0][2] = 2*qx*qz + 2*qw*qy

    m[1][0] = 2*qx*qy + 2*qw*qz
    m[1][1] = 1- 2*qx*qx - 2*qz*qz
    m[1][2] = 2*qy*qz - 2*qw*qx

    m[2][0] = 2*qx*qz - 2*qw*qy
    m[2][1] = 2*qy*qz + 2*qw*qx
    m[2][2] = 1- 2*qx*qx - 2*qy*qy
    #return np.linalg.inv(m)
    return m

def convertYCrCB_BGR(y,cr,cb):
    data = []
    delta = 128.0

    r = y + 1.402 * (cr - delta)
    g = y - 0.34414 * (cb - delta) - 0.71414 * (cr - delta) 
    b = y + 1.772 * (cb - delta)
    if r < 0:
        r = 0
    elif r > 255:
        r = 255
    if g < 0:
        g = 0
    elif g > 255:
        g = 255
    if b < 0:
        b = 0
    elif b > 255:
        b = 255
    data.append(r)
    data.append(g)
    data.append(b)
    return data


# TODO : CODY SHOULD IMPLEMENT THIS FUNCTION
# INPUTS
#  - sparse_map : hashtable of data key being "x_y_z" and value being data that we store including confidence score. 
#                  use encode_key(), decode_key() function
#  - origin : hashmap with key x, y, z => global location where the ray starts
#  - direction : hashmap with key x, y, z => represents the unit direction vector of the ray
#  - z : float => represents the length of the ray
#  - cube info => hashtable of key 'size' => length of each edge of the cube 
#                 and 'cube_origin' => global location of (0, 0, 0) of the cube
#                 cube_info['cube_origin'] is also a dictionary with keys "x_origin", "y_origin", "z_origin"
#
# OUTPUT : an updated sparse_map hashtable with updated confidence score 
def ray_cast(sparse_map, origin, direction, z_len, cube_info, r, g, b):
    #ray info
    ray_x = origin['x']
    ray_y = origin['y']
    ray_z = origin['z']
    direction_x = direction['x']
    direction_y = direction['y']
    direction_z = direction['z']
    
    #cube info
    cube_origin_x = cube_info['cube_origin']['x_origin']
    cube_origin_y = cube_info['cube_origin']['y_origin']
    cube_origin_z = cube_info['cube_origin']['z_origin']
    grid_size = cube_info['grid_size']
    cell_width = cube_info['cell_width']

    delta_z = 0.05
    cumulative_z = 0.05
    previous = ""
    
    #final_x = ray_x + direction_x * z_len 
    #final_y = ray_y + direction_y * z_len
    #final_z = ray_z + direction_z * z_len 

    #data = {'x' : final_x, 'y' : final_y, 'z' : final_z }
    #hit_points.append(data)

    while cumulative_z <= z_len:
        
        curr_x = ray_x + direction_x * cumulative_z
        curr_y = ray_y + direction_y * cumulative_z
        curr_z = ray_z + direction_z * cumulative_z

        x = curr_x - cube_origin_x
        y = curr_y - cube_origin_y
        z = curr_z - cube_origin_z

        if (x >= 0) and (x <= grid_size * cell_width):
            if (y >= 0) and (y <= grid_size * cell_width):
                if (z >= 0) and (z <= grid_size * cell_width):
                    key_x = math.floor(x / cell_width)
                    key_y = math.floor(y / cell_width)
                    key_z = math.floor(z / cell_width)
                    key = encode_key(key_x, key_y, key_z)
                    
                    # add an occupied observation to this cell's observation object
                    if cumulative_z == z_len:
                        # in this case, there should already be something in sparse_map, so a null pointer shouldn't be thrown
                        if key == previous:
                            observation = sparse_map[key]
                            observation.occupancyCount = observation.occupancyCount + 1
                            observation.r = (observation.r * observation.observationCount + r) / float(observation.observationCount + 1)
                            observation.g = (observation.g * observation.observationCount + g) / float(observation.observationCount + 1)
                            observation.b = (observation.b * observation.observationCount + b) / float(observation.observationCount + 1)
                            observation.occupancyConfidence = float(float(observation.occupancyCount)/float(observation.observationCount))
                            sparse_map[key] = observation

                        # in this case, we need to check whether or not there's an observation object in sparse_map already
                        else:
                            if key in sparse_map:
                                observation = sparse_map[key]
                                observation.r = (observation.r * observation.observationCount + r) / float(observation.observationCount + 1)
                                observation.g = (observation.g * observation.observationCount + g) / float(observation.observationCount + 1)
                                observation.b = (observation.b * observation.observationCount + b) / float(observation.observationCount + 1)
                                observation.observationCount = observation.observationCount + 1
                                observation.occupancyCount = observation.occupancyCount + 1
                                observation.occupancyConfidence = float(float(observation.occupancyCount) / float(observation.observationCount))
                                sparse_map[key] = observation
 
                            else:
                                observation = Observation()
                                observation.r = (observation.r * observation.observationCount + r) / float(observation.observationCount + 1)
                                observation.g = (observation.g * observation.observationCount + g) / float(observation.observationCount + 1)
                                observation.b = (observation.b * observation.observationCount + b) / float(observation.observationCount + 1)
                                observation.observationCount = 1
                                observation.occupancyCount = 1
                                observation.occupancyConfidence = float(float(observation.occupancyCount) / float(observation.observationCount))
                                sparse_map[key] = observation

                            previous = key

                    # add an unoccupied observation to this cell's observation object
                    elif key != previous:
                        if key in sparse_map:
                            observation = sparse_map[key]
                            observation.observationCount = observation.observationCount + 1
                            observation.occupancyConfidence = float(float(observation.occupancyCount) / float(observation.observationCount))
                            sparse_map[key] = observation
    
                        else:
                            observation = Observation()
                            observation.occupancyConfidence = 0
                            observation.observationCount = 1
                            sparse_map[key] = observation
                        previous = key

        #ensures that we don't skip over checking the exact location of intersection
        if (cumulative_z != z_len) and ((cumulative_z + delta_z) >= z_len):
            cumulative_z = z_len
        else:
            cumulative_z = cumulative_z + delta_z

    return sparse_map


def set_cube_dimension(top_down_view_info, padding, grid_size):

    x_size = max(abs(top_down_view_info['x_min'] - top_down_view_info['x_avg']), 
         abs(top_down_view_info['x_max'] - top_down_view_info['x_avg']) )

    y_size = max(abs(top_down_view_info['y_min'] - top_down_view_info['y_avg']), 
         abs(top_down_view_info['y_max'] - top_down_view_info['y_avg']) )

    z_size = max(abs(top_down_view_info['z_min'] - top_down_view_info['z_avg']), 
         abs(top_down_view_info['z_max'] - top_down_view_info['z_avg']) )

    half = max(x_size, y_size, z_size) * padding
    cube_size = float(2*half)
    cube_origin = { 'x_origin': top_down_view_info['x_avg'] - half,
                    'y_origin': top_down_view_info['y_avg'] - half, 
                    'z_origin': top_down_view_info['z_avg'] - half}

    cube_info = {'size' : cube_size, 'cube_origin' : cube_origin, 'grid_size' : grid_size, 'cell_width' : cube_size/float(grid_size)}
    return cube_info

def intersect_cube(starting_point, direction_x, direction_y, direction_z, cube_info):
    top_point_x = cube_info['cube_origin']['x_origin']
    top_point_y = cube_info['cube_origin']['y_origin']
    top_point_z = cube_info['cube_origin']['z_origin']
    bottom_point_x = cube_info['cube_origin']['x_origin'] + cube_info['size']
    bottom_point_y = cube_info['cube_origin']['y_origin'] + cube_info['size']
    bottom_point_z = cube_info['cube_origin']['z_origin'] + cube_info['size']

    
    bottom_plane = Plane(Point3D(top_point_x, top_point_y, top_point_z), normal_vector=(0, 0, -1))
    back_plane = Plane(Point3D(top_point_x, top_point_y, top_point_z), normal_vector=(0, -1, 0))
    side_plane1 = Plane(Point3D(top_point_x, top_point_y, top_point_z), normal_vector=(-1, 0, 0))

    top_plane = Plane(Point3D(bottom_point_x, bottom_point_y, bottom_point_z), normal_vector=(0, 0, 1))
    front_plane = Plane(Point3D(bottom_point_x, bottom_point_y, bottom_point_z), normal_vector=(0, 1, 0))
    side_plane2 = Plane(Point3D(bottom_point_x, bottom_point_y, bottom_point_z), normal_vector=(1, 0, 0))

    vector = Line3D(Point3D(starting_point['x'], starting_point['y'], starting_point['z']),
            Point3D(starting_point['x'] + direction_x , starting_point['y'] + direction_y , starting_point['z'] + direction_z))

    top_intersection = top_plane.intersection(vector)
    bottom_intersection = bottom_plane.intersection(vector)
    side1_intersection = side_plane1.intersection(vector)
    side2_intersection = side_plane2.intersection(vector)
    front_intersection = front_plane.intersection(vector)
    back_intersection = back_plane.intersection(vector)

    if len(top_intersection) > 0 and type(top_intersection[0]) is Point3D:
        point = top_intersection[0]
        if point.x < bottom_point_x and point.x > top_point_x and point.y < bottom_point_y and point.y > top_point_y:
            print cube_info
            print "top intersected intersectiont point : " + str(float(point.x)) + ', ' + str(float(point.y)) + '\n\n\n'
            return True

    if len(bottom_intersection) > 0 and type(bottom_intersection[0]) is Point3D:
        point = bottom_intersection[0]
        if point.x < bottom_point_x and point.x > top_point_x and point.y < bottom_point_y and point.y > top_point_y:
            print "bottom intersected"
            return True

    if len(side1_intersection) > 0 and type(side1_intersection[0]) is Point3D:
        point = side1_intersection[0]
        if point.z < bottom_point_z and point.z > top_point_z and point.y < bottom_point_y and point.y > top_point_y:
            print "side1 intersected"
            return True

    if len(side2_intersection) > 0 and type(side2_intersection[0]) is Point3D:
        point = side2_intersection[0]
        if point.z < bottom_point_z and point.z > top_point_z and point.y < bottom_point_y and point.y > top_point_y:
            print "side2 intersected"
            return True

    if len(front_intersection) > 0 and type(front_intersection[0]) is Point3D:
        point = front_intersection[0]
        if point.x < bottom_point_x and point.x > top_point_x and point.z < bottom_point_z and point.z > top_point_z:
            print "front intersected"
            return True

    if len(back_intersection) > 0 and type(back_intersection[0]) is Point3D:
        point = back_intersection[0]
        if point.x < bottom_point_x and point.x > top_point_x and point.z < bottom_point_z and point.z > top_point_z:
            print "back intersected"
            return True

    print "no intersection found returning false!"
    return False


def get_ray_info(file_arr, grid_size):
    result = {}
    x_max = 0
    x_avg = 0
    x_min = float('inf')
    y_max = 0
    y_min = float('inf')
    y_avg = 0
    z_max = 0
    z_min  = float('inf')
    z_avg = 0
    count = 0
    for fname in file_arr:
        with open(fname, "r") as f:
            for line in f:

                line = line.strip()
                arr = line.split(" ")

                if int(arr[21]) == 2:
                    count = count + 1
                    ax = float(arr[1])
                    ay = float(arr[3])
                    az = float(arr[5])

                    x_avg = x_avg + ax
                    y_avg = y_avg + ay 
                    z_avg = z_avg + az

                    if(ax > x_max):
                        x_max = ax
                    if(ay > y_max):
                        y_max = ay
                    if(az > z_max):
                        z_max = az
                    if(ax < x_min):
                        x_min = ax
                    if (ay < y_min):
                        y_min = ay
                    if (az < z_min ):
                        z_min = az


    x_avg = x_avg / float(count)
    y_avg = y_avg / float(count)
    z_avg = z_avg / float(count)

    result = {"x_min" : x_min, "x_max" : x_max, "x_avg" : x_avg,
            "y_min" : y_min, "y_max" : y_max, "y_avg" : y_avg,
            "z_min" : z_min, "z_max" : z_max, "z_avg" : z_avg}
            
    return set_cube_dimension(result, 1.2 , grid_size)



def filter_angle(x, y, threshold):
    if float(x) * float(x) + float(y) * float(y) > threshold:
        return True

    return False


# casts a single ray, which is in the format specified by John
def ray_cast(sparse_map2, cube_info, ray):

    #if ray comes as a string:
    array = ray.split(" ")
    ray_components = []
    for i in range(0, 10):
        index = i * 2 + 1
        ray_components.append(array[index])


    ray_x = float(ray_components[0])
    ray_y = float(ray_components[1])
    ray_z = float(ray_components[2])
    
    direction_x = float(ray_components[3]) - ray_x
    direction_y = float(ray_components[4]) - ray_y
    direction_z = float(ray_components[5]) - ray_z
    r = float(ray_components[6])
    g = float(ray_components[7])
    b = float(ray_components[8])
    a = float(ray_components[9])


    vector = [direction_x, direction_y, direction_z]
    norm = np.linalg.norm(vector)
    if norm != 0:
        direction_x = vector[0] / float(norm)
        direction_y = vector[1] / float(norm)
        direction_z = vector[2] / float(norm)

    typeA = True
    if r == 0 and g == 0 and b == 0 and a != 0:
        typeA = False
    
    #cube info
    cube_origin_x = cube_info['cube_origin']['x_origin']
    cube_origin_y = cube_info['cube_origin']['y_origin']
    cube_origin_z = cube_info['cube_origin']['z_origin']
    grid_size = cube_info['grid_size']
    cell_width = cube_info['cell_width']

    if typeA:
        angle_threshold = 0.005
        if filter_angle(direction_x, direction_y, angle_threshold):
            return sparse_map2


        # Used in decideKeepGoing(): Calculate locations of 8 corners. Calculate next point and see if the ray is moving further away from all 8 corners. If so, return False
        length = grid_size * cell_width
            #origin, AKA front bottom left
        corner1 = np.array([cube_origin_x, cube_origin_y, cube_origin_z])
            #front top left
        corner2 = np.array([cube_origin_x, cube_origin_y + length, cube_origin_z])
            #front top right
        corner3 = np.array([cube_origin_x + length, cube_origin_y + length, cube_origin_z])
            #front bottom right
        corner4 = np.array([cube_origin_x + length, cube_origin_y, cube_origin_z])
            #back bottom left
        corner5 = np.array([cube_origin_x, cube_origin_y, cube_origin_z + length])
            #back top left
        corner6 = np.array([cube_origin_x, cube_origin_y + length, cube_origin_z + length])
            #back top right
        corner7 = np.array([cube_origin_x + length, cube_origin_y + length, cube_origin_z + length])
            #back bottom left
        corner8 = np.array([cube_origin_x + length, cube_origin_y, cube_origin_z + length])

        previous = ""
        delta_z = 0.005
        cumulative_z = 0.0
        keepGoing = True

        while keepGoing:
            curr_x = ray_x + direction_x * cumulative_z
            curr_y = ray_y + direction_y * cumulative_z
            curr_z = ray_z + direction_z * cumulative_z

            x = curr_x - cube_origin_x
            y = curr_y - cube_origin_y
            z = curr_z - cube_origin_z

            if (x >= 0) and (x <= grid_size * cell_width):
                if (y >= 0) and (y <= grid_size * cell_width):
                    if (z >= 0) and (z <= grid_size * cell_width):
                        key_x = math.floor(x / cell_width)
                        key_y = math.floor(y / cell_width)
                        key_z = math.floor(z / cell_width)
                        key = encode_key(key_x, key_y, key_z)

                        if key != previous:
                            if key in sparse_map2:
                                # should we still average in the 'a' value since we know it's 0?
                                observation = sparse_map2[key]
                                observation.r = (observation.r * observation.countRGB + r) / float(observation.countRGB + 1)
                                observation.g = (observation.g * observation.countRGB + g) / float(observation.countRGB + 1)
                                observation.b = (observation.b * observation.countRGB + b) / float(observation.countRGB + 1)
                                #observation.a = (observation.a * observation.count + a) / float(observation.count + 1)
                                # observation.r = r
                                # observation.g = g
                                # observation.b = b
                                observation.countRGB = observation.countRGB + 1
                                sparse_map2[key] = observation
                            else:
                                observation = Observation2()
                                observation.countRGB = 1
                                observation.r = r
                                observation.g = g
                                observation.b = b
                                #observation.a = a
                                sparse_map2[key] = observation
                            previous = key

            keepGoing = decideKeepGoing(cube_info, curr_x, curr_y, curr_z, direction_x, direction_y, direction_z, delta_z, corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8)
            cumulative_z += delta_z      

    else:
        x = ray_x - cube_origin_x
        y = ray_y - cube_origin_y
        z = ray_z - cube_origin_z
        if (x >= 0) and (x <= grid_size * cell_width):
            if (y >= 0) and (y <= grid_size * cell_width):
                if (z >= 0) and (z <= grid_size * cell_width):
                    key_x = math.floor(x / cell_width)
                    key_y = math.floor(y / cell_width)
                    key_z = math.floor(z / cell_width)
                    key = encode_key(key_x, key_y, key_z)

                    if key in sparse_map2:
                        # should we still average the r g b channels here since we know they're 0?
                        observation = sparse_map2[key]
                        #observation.r = (observation.r * observation.count + r) / float(observation.count + 1)
                        #observation.g = (observation.g * observation.count + g) / float(observation.count + 1)
                        #observation.b = (observation.b * observation.count + b) / float(observation.count + 1)
                        observation.a = (observation.a * observation.countA + a) / float(observation.countA + 1)
                        observation.countA = observation.countA + 1
                        sparse_map2[key] = observation
                    else:
                        observation = Observation2()
                        observation.countA = 1
                        #observation.r = r
                        #observation.g = g
                        #observation.b = b
                        observation.a = a
                        sparse_map2[key] = observation
    return sparse_map2




def decideKeepGoing(cube_info, x, y, z, dirX, dirY, dirZ, delta_z, corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8):
    starting_point = {'x' : x, 'y' : y, 'z': z}
    if not intersect_cube(starting_point, dirX, dirY, dirZ, cube_info):
        return False

    curr = np.array([x, y, z])
    nextPoint = np.array([x + delta_z * dirX, y + delta_z * dirY, z + delta_z * dirZ])

    curr1 = np.linalg.norm(curr - corner1)
    curr2 = np.linalg.norm(curr - corner2)
    curr3 = np.linalg.norm(curr - corner3)
    curr4 = np.linalg.norm(curr - corner4)
    curr5 = np.linalg.norm(curr - corner5)
    curr6 = np.linalg.norm(curr - corner6)
    curr7 = np.linalg.norm(curr - corner7)
    curr8 = np.linalg.norm(curr - corner8)

    next1 = np.linalg.norm(nextPoint - corner1)
    next2 = np.linalg.norm(nextPoint - corner2)
    next3 = np.linalg.norm(nextPoint - corner3)
    next4 = np.linalg.norm(nextPoint - corner4)
    next5 = np.linalg.norm(nextPoint - corner5)
    next6 = np.linalg.norm(nextPoint - corner6)
    next7 = np.linalg.norm(nextPoint - corner7)
    next8 = np.linalg.norm(nextPoint - corner8)

    if next1 > curr1 and next2 > curr2 and next3 > curr3 and next4 > curr4 and next5 > curr5 and next6 > curr6 and next7 > curr7 and next8 > curr8:
        return False
    else:
        return True


################################################################################################
################################################################################################
################################################################################################

def main():
    # PAREMETER TUNING 
    GRID_SIZE = 100 # there are GRID_SIZE^3 cells in the cube / number of smalls cubes in one edge
    PADDING_RATE = 1.2 # how much more space are we going to consider other than (min - max)
    THRESHOLD  = 0.8
    MIN_OBSERVATION = 3


    # INITIALIZE VARIABLES 
    sparse_map = {}
    fileList = ["duckStrawberryBowlAlpha1.ray"]
    cube_info = get_ray_info(fileList, GRID_SIZE)


    # only type (ALPHA)
    print "starting ray casting for alpha ray...."
    f1 = open("duckStrawberryBowlAlpha1.ray")
    lines1 = f1.readlines()
    for line1 in lines1:
        line1 = line1.strip()
        sparse_map = ray_cast(sparse_map, cube_info, line1)



    # only type (BETA)
    print "starting ray casting for beta rays...."    
    f3 = open("duckStrawberryBowlRGB1.ray")
    lines3 = f3.readlines()

    count = 0
    num_rays = 0
    for line3 in lines3:
        line3 = line3.strip()
        sparse_map = ray_cast(sparse_map, cube_info, line3)
        num_rays += 1


    print "length of final sparse map  : " + str(len(sparse_map))
    print "number of final rays that was included in computation : " + str(num_rays)



    # 3. WRITE SPARSE MAP INTO JSON FILE
    data = []
    n_count = 0
    scoreMap = {}
    print " ======  writing to file ======"
    for key in sparse_map:
        position = decode_key(key)
        y = float(sparse_map[key].r)
        cr = float(sparse_map[key].g)
        cb = float(sparse_map[key].b)
        a = float(sparse_map[key].a)
        countRGB = float(sparse_map[key].countRGB)
        bgr_array = convertYCrCB_BGR(y, cr, cb)
        r_mu = float(bgr_array[0])
        g_mu = float(bgr_array[1])
        b_mu = float(bgr_array[2])

        r2_mu = sparse_map[key].r
        g2_mu = sparse_map[key].g
        b2_mu = sparse_map[key].b

        if sparse_map[key].b == 0 and sparse_map[key].g == 0 and sparse_map[key].r == 0:
            r_mu = 255
            g_mu = 0
            b_mu = 0

        n_count += 1
        data.append({'x': position['x']*cube_info["cell_width"] , 'y': position['y']*cube_info["cell_width"] , 'z': position['z']*cube_info["cell_width"] , 
            'countRGB': countRGB, 'a': a, 'r' : r_mu, 'g': g_mu, 'b': b_mu, 'r2': r2_mu, 'g2': g2_mu, 'b2': b2_mu})

    print n_count
    print ".....done!"

    
    out_file = open("ray_output.json", "w")
    json.dump(data, out_file, indent=4) 
    
    # Close the file
    out_file.close()

    return 


if __name__ == "__main__":
    # top_down_file = sys.argv[1]
    # other_views = []
    # l = len(sys.argv)

    # if l < 2:
    #     print "at least one slug file and output file name is needed"
    # else: 

    #     file_name = sys.argv[l-1];

    #     for i in range(2,l-1):
    #         other_views.append(sys.argv[i])

    main()




   