import yaml
import sys
from base64 import b64decode
import zlib
import struct
import json
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter


class Observation:
    observationCount = 0
    occupancyCount = 0
    occupancyConfidence = 0.0
    r = 0.0
    g = 0.0
    b = 0.0

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
    background_pose  = None
    for line in f:
        if "background_pose" in line:
            background_pose = line.split('{')[1].split('}')[0]
            background_pose = background_pose.replace(".", "")
            background_pose = background_pose.split(",")
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


    for x in range(0, col):
        for y in range(0, row):
            index = x + col * y;
            cell = observed_map.cells[index]
            z_mu = float(observed_map.cells[index].z.mu)
            
            if z_mu > 0 and z_mu < max_length:
                x_len = x*(cell_width) 
                y_len = y*(cell_width)
                x_avg = x_avg + x_len
                y_avg = y_avg + y_len
                z_avg = z_avg + z_mu
                count = count + 1

                if x_len > x_max:
                    x_max = x_len
                if y_len > y_max:
                    y_max = y_len
                if y_len < y_min:
                    y_min = y_len
                if x_len < x_min:
                    x_min = x_len
                if z_mu < z_min:
                    z_min = z_mu
                if z_mu > z_max:
                    z_max = z_mu

    info['cell_len'] = cell_width
    info['rows'] = row
    info['cols'] = col
    info['x_max'] = round(x_max, 3)
    info['x_min'] = round(x_min, 3)
    info['y_max'] = round(y_max, 3)
    info['y_min'] = round(y_min, 3)
    info['z_max'] = round(z_max, 3)
    info['z_min'] = round(z_min, 3)
    info['x_avg'] = round(x_avg/count)
    info['y_avg'] = round(y_avg/count)
    info['z_avg'] = round(z_avg/count)
    
    info['position'] = { 'x' : float(background_pose[0].split(':')[1]), 'y' : float(background_pose[1].split(':')[1]), 
                        'z': float(background_pose[2].split(':')[1]), 'qw' :float(background_pose[3].split(':')[1]), 
                        'qx': float(background_pose[4].split(':')[1]), 'qy':float(background_pose[5].split(':')[1]), 
                        'qz': float(background_pose[6].split(':')[1])}

    return info



def get_info_from_top_view(file_name):
    return get_slug_info(file_name, float('inf'))



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

#def get_ray_origin(slug_info, x, y, cell_length):
#    #origin + cell_length* x * cos(angle)
#    default_pos = slug_info['position']
#    rotation_matrix = quaternion_to_rotation_matrix(slug_info['position']['qw'],
#                                                    slug_info['position']['qx'], 
#                                                    slug_info['position']['qy'], 
#                                                    slug_info['position']['qz'])

#    #rotation matrix  has property  inverse = transpose
#    cell_width = slug_info["cell_len"]
#    vector = np.array([x*cell_width, y*cell_width, 0])
#    current_vector = np.dot(rotation_matrix, vector)

#    x = round(float(current_vector[0] + default_pos['x']), 3)
#    y = round(float(current_vector[1] + default_pos['y']), 3)
#    z = round(float(current_vector[2] + default_pos['z']), 3)
#    return {'x' : x, 'y':  y  , 'z' : z}


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

    print " ====== starting ray casting ========"

    for x in range(0, col):
        for y in range(0, row):
            index = x + col * y;
            cell = observed_map.cells[index]
            r = float(cell.red.mu)
            g = float(cell.green.mu)
            b = float(cell.blue.mu)
            z_mu = float(cell.z.mu)
            ray_direction = get_ray_direction(slug_info)
            
            if z_mu > 0:
                ray_origin = get_ray_origin(slug_info, x, y, cell_length)
                sparse_map = ray_cast(sparse_map, ray_origin, ray_direction, z_mu, cube_info, r, g, b)

    print "===== end of ray casting ========= "

    return sparse_map

# returns a directional unit vector hashtable of {x , y, z}
def q_to_euler(qw, qx, qy, qz):
    rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    direction_vector = np.dot(rotation_matrix, np.array([0, 0, 1]))
    direction_vector = direction_vector/np.sum(direction_vector)

    return {'x': direction_vector[0], 'y': direction_vector[1],'z': direction_vector[2]}

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


#def get_ray_direction(slug_info):
#    quaternion = slug_info['position']
#    qw = quaternion['qw']
#    qx = quaternion['qx']
#    qy = quaternion['qy']
#    qz = quaternion['qz']
#    direction = q_to_euler(qw, qx, qy, qz)
#    return direction


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
                                observation.occupancyConfidence = float(observation.occupancyCount) / float(observation.observationCount)
                                sparse_map[key] = observation
                            else:
                                observation = Observation()
                                observation.r = (observation.r * observation.observationCount + r) / float(observation.observationCount + 1)
                                observation.g = (observation.g * observation.observationCount + g) / float(observation.observationCount + 1)
                                observation.b = (observation.b * observation.observationCount + b) / float(observation.observationCount + 1)
                                observation.observationCount = 1
                                observation.occupancyCount = 1
                                observation.occupancyConfidence = float(observation.occupancyCount) / float(observation.observationCount)
                                sparse_map[key] = observation
                            previous = key

                    # add an unoccupied observation to this cell's observation object
                    elif key != previous:
                        if key in sparse_map:
                            observation = sparse_map[key]
                            observation.observationCount = observation.observationCount + 1
                            observation.occupancyConfidence = float(observation.occupancyCount) / float(observation.observationCount)
                            sparse_map[key] = observation
                        else:
                            observation = Observation()
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

    #x_size = abs(top_down_view_info['x_min'] - top_down_view_info['x_max']) 
    #y_size = abs(top_down_view_info['y_min'] - top_down_view_info['y_max']) 
    #z_size = abs(top_down_view_info['z_min'] - top_down_view_info['z_max']) 
    

    #the global location of  (0, 0, 0) cell in the cube  
    #cube_origin = { 'x_origin': top_down_view_info['position']['x'] + top_down_view_info['x_min'], 
    #                'y_origin': top_down_view_info['position']['y'] + top_down_view_info['y_min'], 
    #                'z_origin': top_down_view_info['position']['z'] + top_down_view_info['z_min'] }

    cube_info = {'size' : cube_size, 'cube_origin' : cube_origin, 'grid_size' : grid_size, 'cell_width' : cube_size/float(grid_size)}
    return cube_info




################################################################################################
################################################################################################
################################################################################################

def main(top_view, other_views, file_name):
    # PAREMETER TUNING 
    GRID_SIZE = 1000 # there are GRID_SIZE^3 cells in the cube / number of smalls cubes in one edge
    PADDING_RATE = 1.2 # how much more space are we going to consider other than (min - max)
    THRESHOLD  = 0.8

    # 0. CREATE CUBE DIMENSION AND SPARSE MAP 
    # setting spase map and size of the cube from the top down view 
    # from the top down view, obtain the dimensions of the cube
    # for now, I set (0, 0, 0) location of the cube as 0.8 * (min x, min y, min z) 
    # also the length (size) of the cube is set to be 1.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    # TODO: CODY needs to change this as necessary
    sparse_map = {}
    top_down_view_info = get_info_from_top_view(top_view)

    cube_info = set_cube_dimension(top_down_view_info, PADDING_RATE, GRID_SIZE)
    print "cube info : " + str(cube_info)

    # 1. RAY CAST FROM TOP DOWN VIEW SLUG
    print "===== reading from top down view ====== "
    sparse_map = read_from_yml(top_view, sparse_map, top_down_view_info, cube_info)



    # 2. RAY CAST FROM OTHER VIEWS 
    
    for other_view in other_views:
        view_info = get_slug_info(other_view, cube_size)
        sparse_map = read_from_yml(other_view, sparse_map, view_info, cube_info)


    # 3. WRITE SPARSE MAP INTO JSON FILE
    data = []

    print " ======  writing to file ======"
    for key in sparse_map:
        position = decode_key(key)
        y = float(sparse_map[key].b)
        cr = float(sparse_map[key].g)
        cb = float(sparse_map[key].r)
        bgr_array = convertYCrCB_BGR(y, cr, cb)
        r_mu = float(bgr_array[0])
        g_mu = float(bgr_array[1])
        b_mu = float(bgr_array[2])
        
        score = sparse_map[key].occupancyConfidence

        if score >= THRESHOLD:
            data.append({'x': position['x']*cube_info["cell_width"] , 'y': position['y']*cube_info["cell_width"] , 'z': position['z']*cube_info["cell_width"] , 
                'score': score , 'r' : r_mu, 'g': g_mu, 'b': b_mu})
        else:
            n_count = n_count + 1

    print " noise count is : " + str(n_count) + " among total of : " + str(len(sparse_map))

    out_file = open(file_name, "w")

    # Save the dictionary into this file
    # (the 'indent=4' is optional, but makes it more readable)
    json.dump(data, out_file, indent=4)                                    

    # Close the file
    out_file.close()

    return 


if __name__ == "__main__":
    top_down_file = sys.argv[1]
    other_views = []
    l = len(sys.argv)

    if l < 2:
        print "at least one slug file and output file name is needed"
    else: 

        file_name = sys.argv[l-1];

        for i in range(2,l-1):
            other_views.append(sys.argv[i])


        print "processing " + str(l-2) + " yaml files and creating " +  str(file_name) + "  json file..." 

        main(top_down_file, other_views, file_name )

   