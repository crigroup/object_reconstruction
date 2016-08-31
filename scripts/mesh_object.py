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


class MeshObject(object):
  def __init__(self, options):
    # Config stuff
    np.set_printoptions(precision=6, suppress=True)
    self.leaf_size = options.leaf_size
    self.seg_radius_crop = options.seg_radius_crop
    self.seg_z_crop = options.seg_z_crop
    self.seg_z_min = options.seg_z_min
    self.bag = rosbag.Bag(options.bag, 'r')
    self.basename = os.path.splitext(os.path.abspath(options.bag))[0]
    rospy.on_shutdown(self.on_shutdown)
  
  def consistent_bag(self):
    # check we have the same number of msgs
    baginfo = yaml.load(self.bag._get_yaml_info())
    msgs = [ topic['messages'] for topic in baginfo['topics'] ]
    return len(set(msgs)) == 1, msgs[0]
  
  def execute(self):
    consistent, num_msgs = self.consistent_bag()
    if not consistent:
      rospy.logerr( 'The input bag is not consistent.' 
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
        xy_dist = tr.vector_norm(point[:2]-table_center[:2])
        if (inside_z_range and xy_dist < self.seg_radius_crop):
          inliers.append(i)
      cloud = pcl.PointCloud( cloud_xyz[inliers] )
      # Statistical outlier filter x 2
      for i in range(2):
        stat_filter = cloud.make_statistical_outlier_filter()
        stat_filter.set_mean_k(100)
        stat_filter.set_std_dev_mul_thresh(0.002)
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
    # Final statistical outlier filter
    cloud = pcl.PointCloud( all_points )
    stat_filter = cloud.make_statistical_outlier_filter()
    stat_filter.set_mean_k(50)
    stat_filter.set_std_dev_mul_thresh(2)
    cloud = stat_filter.filter()
    # Generate files
    devnull = open('/dev/null', 'w')
    pcd_filename = self.basename + '.pcd'
    ply_filename = self.basename + '.ply'
    stl_filename = self.basename + '.stl'
    # PCD
    cloud.to_file(pcd_filename)
    rospy.loginfo('Saved PCD file: %s' % pcd_filename)
    # PLY
    env = os.environ
    #  object_mesh.pcd object_mesh.ply
    subprocess.Popen('pcl_pcd2ply -format 1 -use_camera 0 ' + pcd_filename + ' ' + ply_filename,
                     env=env, cwd=os.getcwd(), shell=True, stdout=devnull).wait()
    rospy.loginfo('Saved PLY file: %s' % ply_filename)
    # STL
    f = tempfile.NamedTemporaryFile(delete=False)
    script = f.name
    f.write(FILTER_SCRIPT_PIVOTING)
    f.close()
    env['LC_NUMERIC'] = 'C'
    subprocess.Popen('meshlabserver -i ' + ply_filename + ' -o ' + stl_filename + ' -s ' + script,
                     env=env, cwd=os.getcwd(), shell=True, stdout=devnull, stderr=devnull).wait()
    os.remove(script)
    rospy.loginfo('Saved STL file: %s' % stl_filename)
  
  def on_shutdown(self):
    # Close rosbag
    self.bag.close()


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
  parser.add_argument('--seg_z_crop', metavar='SEG_Z_CROP', type=float, default= 0.5,
                      help='The amount to keep in the z direction (meters)' 
                      'relative to the coordinate frame defined by the pose. default=%(default).3f')
  parser.add_argument('--seg_z_min', metavar='SEG_Z_MIN', type=float, default= 0.005,
                      help='The amount to crop above the plane, in meters. default=%(default).4f')
  parser.add_argument('--debug', action='store_true',
                      help='If set, will show additional debugging information')
  args = parser.parse_args(rospy.myargv()[1:])
  if len(args.bag) < 1:
    parser.print_help()
    print '\nYou must supply the input bag name'
    sys.exit(1)
  return args

if "__main__" == __name__:
  options = parse_args()
  log_level= rospy.DEBUG if options.debug else rospy.INFO
  node_name = os.path.splitext(os.path.basename(__file__))[0]
  rospy.init_node(node_name, log_level=log_level)
  meshit = MeshObject(options)
  meshit.execute()
