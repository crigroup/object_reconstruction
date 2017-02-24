#!/usr/bin/env python
import os
import sys
import yaml
import rospy
import rosbag
import criros
import argparse
import tempfile
import subprocess
import numpy as np
import tf.transformations as tr
# Progress bar
import progressbar
# OpenCV
import cv2
# PCL
import pcl
from ros_numpy.point_cloud2 import get_xyz_points, pointcloud2_to_array
# Messages
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import (
  Image, 
  PointCloud2
)

"""
Examples:
rosrun object_reconstruction mesh_object.py -i right_frame.bag --leaf_size 0.002 --seg_z_min 0.01 --final_stat_filter_k 20 --final_stat_filter_std_dev 1 --final_leaf_size 0.001 --stat_filter_k -1 --seg_radius_crop -1

rosrun object_reconstruction mesh_object.py -i left_frame.bag --leaf_size 0.002 --seg_z_min 0.01 --final_stat_filter_k 50 --final_stat_filter_std_dev 1 --final_leaf_size 0.001 --stat_filter_k -1 --seg_radius_crop -1

rosrun object_reconstruction mesh_object.py -i bottle.bag --leaf_size -1 --final_leaf_size 0.001 --seg_z_min 0.01 --seg_radius_crop 0.15 --stat_filter_k -1 --stat_filter_std_dev 0.5 --final_stat_filter_k 10 --final_stat_filter_std_dev 1
"""


FILTER_SCRIPT_PIVOTING="""
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Surface Reconstruction: Ball Pivoting">
  <Param type="RichAbsPerc" value="0.01" min="0" name="BallRadius" max="0.122839"/>
  <Param type="RichFloat" value="20" name="Clustering"/>
  <Param type="RichFloat" value="90" name="CreaseThr"/>
  <Param type="RichBool" value="false" name="DeleteFaces"/>
 </filter>
 <filter name="Laplacian Smooth">
  <Param type="RichInt" value="1" name="stepSmoothNum"/>
  <Param type="RichBool" value="true" name="Boundary"/>
  <Param type="RichBool" value="true" name="cotangentWeight"/>
  <Param type="RichBool" value="false" name="Selected"/>
 </filter>
</FilterScript>
"""

logger = criros.utils.TextColors()

class MeshObject(object):
  def __init__(self, options):
    # Config stuff
    np.set_printoptions(precision=6, suppress=True)
    self.leaf_size = options.leaf_size
    self.seg_radius_crop = options.seg_radius_crop
    self.stat_filter_k = options.stat_filter_k
    self.stat_filter_std_dev = options.stat_filter_std_dev
    self.seg_z_crop = options.seg_z_crop
    self.seg_z_min = options.seg_z_min
    self.bag = rosbag.Bag(options.bag, 'r')
    self.final_stat_filter_k = options.final_stat_filter_k
    self.final_stat_filter_std_dev = options.final_stat_filter_std_dev
    self.final_leaf_size = options.final_leaf_size
    self.stl = options.stl
    self.basename = os.path.splitext(os.path.abspath(options.bag))[0]
  
  def consistent_bag(self):
    # check we have the same number of msgs
    baginfo = yaml.load(self.bag._get_yaml_info())
    msgs = [ topic['messages'] for topic in baginfo['topics'] ]
    return len(set(msgs)) == 1, msgs[0]
  
  def execute(self):
    consistent, num_msgs = self.consistent_bag()
    if not consistent:
      logger.logerr( 'The input bag is not consistent.' 
                    'Check that all the topics have the same number of messages.')
      return
    # Get the pattern poses and estimate the table's center
    poses = []
    positions = [] 
    for _, msg, _ in self.bag.read_messages(topics=['/camera/pattern/pose']):
      poses.append( criros.conversions.from_pose(msg.pose) )
      positions.append( poses[-1][:3,3] )
    table_center = np.mean(positions, axis=0)
    pbar = progressbar.ProgressBar(widgets=['Processed clouds: ', progressbar.SimpleProgress()], maxval=num_msgs).start()
    # Let's process each point cloud and stitch them together
    Tbase = np.array(poses[0])
    all_points = []
    for _, msg, _ in self.bag.read_messages(topics=['/camera/depth/points']):
      idx = num_msgs-len(poses)
      # Get the point cloud with the same shape as the images
      cloud_img = get_xyz_points(pointcloud2_to_array(msg), remove_nans=True, dtype=np.float32)
      cloud_xyz = cloud_img.view()
      cloud_xyz.shape = (-1,3)
      cloud = pcl.PointCloud( cloud_xyz )
      # Voxel grid filter
      if self.leaf_size > 0:
        voxel_filter = cloud.make_voxel_grid_filter()
        leaf_size = np.ones(3)*self.leaf_size
        voxel_filter.set_leaf_size(*leaf_size)
        cloud = voxel_filter.filter()
      # Euclidian filter
      cloud_xyz = np.asarray(cloud)
      Tpattern = poses.pop(0)
      plane = criros.spalg.Plane(normal=-Tpattern[:3,2], point=Tpattern[:3,3])
      inliers = []
      for i,point in enumerate(cloud_xyz):
        inside_z_range = self.seg_z_min < plane.distance(point) < self.seg_z_crop
        if self.seg_radius_crop > 0:
          inside_radius = tr.vector_norm(point[:2]-table_center[:2]) < self.seg_radius_crop
        else:
          # A negative value disables this radius cropping
          inside_radius = True
        if (inside_z_range and inside_radius):
          inliers.append(i)
      cloud = cloud.extract(inliers)
      # Statistical outlier filter
      if self.stat_filter_k > 0:
        stat_filter = cloud.make_statistical_outlier_filter()
        stat_filter.set_mean_k(self.stat_filter_k)
        stat_filter.set_std_dev_mul_thresh(self.stat_filter_std_dev)
        cloud = stat_filter.filter()
      # Match point clouds transformation
      transformed_cloud = []
      for point in np.asarray(cloud):
        hom_point = np.ones(4)
        hom_point[:3] = point
        Tcloud =  np.dot(Tbase, criros.spalg.transform_inv(Tpattern))
        transformed_cloud.append(np.dot(Tcloud, hom_point)[:3])
      all_points += transformed_cloud
      pbar.update(num_msgs-len(poses))
    self.bag.close()
    pbar.finish()
    cloud = pcl.PointCloud( all_points )
    # Final statistical outlier filter
    if self.final_stat_filter_k > 0:
      stat_filter = cloud.make_statistical_outlier_filter()
      stat_filter.set_mean_k(self.final_stat_filter_k)
      stat_filter.set_std_dev_mul_thresh(self.final_stat_filter_std_dev)
      cloud = stat_filter.filter()
    # Final voxel grid filter
    if self.final_leaf_size > 0:
      voxel_filter = cloud.make_voxel_grid_filter()
      leaf_size = np.ones(3)*self.final_leaf_size
      voxel_filter.set_leaf_size(*leaf_size)
      cloud = voxel_filter.filter()
    # Generate files
    devnull = open('/dev/null', 'w')
    pcd_filename = self.basename + '.pcd'
    ply_filename = self.basename + '.ply'
    stl_filename = self.basename + '.stl'
    # PCD
    cloud.to_file(pcd_filename)
    logger.loginfo('Resulting cloud has {0} points'.format(cloud.size))
    logger.loginfo('Saved PCD file: %s' % pcd_filename)
    # PLY
    env = os.environ
    subprocess.Popen('pcl_pcd2ply -format 1 -use_camera 0 ' + pcd_filename + ' ' + ply_filename,
                     env=env, cwd=os.getcwd(), shell=True, stdout=devnull).wait()
    logger.loginfo('Saved PLY file: %s' % ply_filename)
    # STL
    if self.stl:
      f = tempfile.NamedTemporaryFile(delete=False)
      script = f.name
      f.write(FILTER_SCRIPT_PIVOTING)
      f.close()
      env['LC_NUMERIC'] = 'C'
      subprocess.Popen('meshlabserver -i ' + ply_filename + ' -o ' + stl_filename + ' -s ' + script,
                       env=env, cwd=os.getcwd(), shell=True, stdout=devnull, stderr=devnull).wait()
      os.remove(script)
      logger.loginfo('Saved STL file: %s' % stl_filename)


def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description='Computes a surface mesh of an object from a rosbag',
                                  fromfile_prefix_chars='@')
  parser.add_argument('-i', '--input', metavar='BAG_FILE', dest='bag', type=str,
                      default='',
                      help='A bagfile to read from.')
  parser.add_argument('--leaf_size', metavar='LEAF_SIZE', type=float, default= 0.005,
                      help='Leaf size (meters) to be used to downsample + filter' 
                      'the initial point cloud. default=%(default).3f')
  parser.add_argument('--seg_radius_crop', metavar='SEG_RADIUS_CROP', type=float, default= 0.3,
                      help='The amount to keep in the XY plane (meters)' 
                      'relative to the table center. default=%(default).3f')
  parser.add_argument('--stat_filter_k', metavar='STAT_FILTER_K', type=int, default=100,
                      help='The number of neighbors to analyze for each point. default=%(default)')
  parser.add_argument('--stat_filter_std_dev', metavar='STAT_FILTER_STD_DEV', type=float, default=0.002,
                      help='All points who have a distance larger than 1 standard deviation'
                      'of the mean distance to the query point will be marked as outliers '
                      'and removed. default=%(default).4f')
  parser.add_argument('--seg_z_crop', metavar='SEG_Z_CROP', type=float, default= 0.5,
                      help='The amount to keep in the z direction (meters)' 
                      'relative to the coordinate frame defined by the pose. default=%(default).3f')
  parser.add_argument('--seg_z_min', metavar='SEG_Z_MIN', type=float, default= 0.005,
                      help='The amount to crop above the plane, in meters. default=%(default).4f')
  parser.add_argument('--final_stat_filter_k', metavar='FINAL_STAT_FILTER_K', type=int, default=10,
                      help='The number of neighbors to analyze for each point. default=%(default)')
  parser.add_argument('--final_stat_filter_std_dev', metavar='FINAL_STAT_FILTER_STD_DEV',
                      type=float, default=1,
                      help='All points who have a distance larger than 1 standard deviation'
                      'of the mean distance to the query point will be marked as outliers '
                      'and removed. default=%(default).4f')
  parser.add_argument('--final_leaf_size', metavar='FINAL_LEAF_SIZE', type=float, default= 0.002,
                      help='Leaf size (meters) to be used to downsample + filter' 
                      'the initial point cloud. default=%(default).3f')
  parser.add_argument('--stl', action='store_true',
                      help='If set, will generate a STL file from the point cloud')
  parser.add_argument('--debug', action='store_true',
                      help='If set, will show additional debugging information')
  args = parser.parse_args(rospy.myargv()[1:])
  if len(args.bag) < 1:
    parser.print_help()
    logger.logerr('\nYou must supply the input bag name')
    sys.exit(1)
  return args

if "__main__" == __name__:
  options = parse_args()
  log_level= rospy.DEBUG if options.debug else rospy.INFO
  logger.set_log_level(log_level)
  meshit = MeshObject(options)
  meshit.execute()
