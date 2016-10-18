#!/usr/bin/env python
import os
import sys
import rospy
import rosbag
import criros
import argparse
import numpy as np
import tf.transformations as tr
# PCL
import pcl
from ros_numpy.point_cloud2 import get_xyz_points, pointcloud2_to_array
# dynamic config
from dynamic_reconfigure.server import Server
from object_reconstruction.cfg import TransformPointcloudConfig
# Messages
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as SensorPointcloudMsg

class TransformPointcloud(object):
  def __init__(self, options):
    # Config stuff
    np.set_printoptions(precision=6, suppress=True)
    self.input_model_filename = options.model_name
    self.model_publisher = rospy.Publisher("/model_pointcloud", PointCloud2)

    
  def execute(self):
    # Load model
    self.model = pcl.PointCloud()
    self.model.from_file(self.input_model_filename +'.pcd')
    if self.model.size!=0:
      Server(TransformPointcloudConfig, self.transform_pointcloud_callback)
    rospy.spin()
    
  def transform_pointcloud_callback(self, config, level):
    rospy.loginfo("----")
    rospy.loginfo("Transform pointcloud params")
    rospy.loginfo("TranslateX: %s" %config['TranslateX'])
    rospy.loginfo("TranslateY: %s" %config['TranslateY'])
    rospy.loginfo("TranslateZ: %s" %config['TranslateZ'])
    rospy.loginfo("RotateX: %s" %config['RotateX'])
    rospy.loginfo("RotateY: %s" %config['RotateY'])
    rospy.loginfo("RotateZ: %s" %config['RotateZ'])
    # if (config.groups.transform.InputFilename != "") and (config.groups.transform.InputFilename!=self.input_model_filename):
    #   Have not implemented
    # Move (transform) the pointcloud transformation
    Tmove = tr.euler_matrix(config['RotateX'], config['RotateY'], config['RotateZ'])
    Tmove[:3,3] = np.array([config['TranslateX'],config['TranslateY'],config['TranslateZ']])
    Tmove = np.linalg.inv(Tmove)
    transformed_model = []
    for point in np.asarray(self.model):
      hom_point = np.ones(4)
      hom_point[:3] = point
      transformed_model.append(np.dot(Tmove, hom_point)[:3])
    # Publish tranformed_model
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera'
    model_pointcloud = SensorPointcloudMsg.create_cloud_xyz32(header, transformed_model)
    self.model_publisher.publish(model_pointcloud)
    # Save transformed model if commanded
    if config['SaveModel']:
      model = pcl.PointCloud(transformed_model)
      model.to_file(config['OutputFilename']+'.pcd')
      rospy.loginfo("Saved model into %s" %(config['OutputFilename']+'.pcd'))
    return config

def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description='Transform the model point cloud',
                                  fromfile_prefix_chars='@')
  parser.add_argument('--model_name', metavar='MODEL_FILENAME', type=str,
                      default='model',
                      help='The file name of the input model (without extenstion!)')
  parser.add_argument('--debug', action='store_true',
                      help='If set, will show additional debugging information')
  args = parser.parse_args(rospy.myargv()[1:])
  return args

if "__main__" == __name__:
  options = parse_args()
  log_level= rospy.DEBUG if options.debug else rospy.INFO
  node_name = os.path.splitext(os.path.basename(__file__))[0]
  rospy.init_node(node_name, log_level=log_level)
  moveit =  TransformPointcloud(options)
  moveit.execute()
