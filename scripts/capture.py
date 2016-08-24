#!/usr/bin/env python
import os
import sys
import math
import rospy
import argparse
import textwrap
import numpy as np
import dynamic_reconfigure.client
# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import StereoCameraModel
# PCL
import pcl
import ros_numpy
from ros_numpy.point_cloud2 import get_xyz_points, pointcloud2_to_array
# Messages
from sensor_msgs.msg import (
  Image, 
  PointCloud2
)


class CaptureServer(object):
  def __init__(self, options):
    # Config stuff
    np.set_printoptions(precision=4, suppress=True)
    self.bridge = CvBridge()
    self.nviews = options.nviews
    self.exposure_time = 1.5
    # Setup publishers and subscribers
    self.reset_snapshots()
    rospy.Subscriber('left/image_raw', Image, self.cb_raw_left)
    rospy.Subscriber('right/image_raw', Image, self.cb_raw_right)
    rospy.Subscriber('left/image_rect', Image, self.cb_rect_left)
    rospy.Subscriber('right/image_rect', Image, self.cb_rect_right)
    rospy.Subscriber('depth/points', PointCloud2, self.cb_point_cloud)
    self.dbg_pub = rospy.Publisher('capture/debug', Image, queue_size=1)
    # Camera configuration client
    self.dynclient = dynamic_reconfigure.client.Client('ensenso_driver', timeout=30, config_callback=self.cb_dynresponse)
    rospy.on_shutdown(self.on_shutdown)
  
  def cb_dynresponse(self, config):
    pass
  
  def cb_point_cloud(self, msg):
    try:
      self.point_cloud = msg
    except:
      self.point_cloud = None
  
  def cb_raw_left(self, msg):
    try:
      self.raw_left = self.bridge.imgmsg_to_cv2(msg, 'mono8')
    except:
      rospy.logdebug('Failed to process left image')
      self.raw_left = None
  
  def cb_raw_right(self, msg):
    try:
      self.raw_right = self.bridge.imgmsg_to_cv2(msg, 'mono8')
    except:
      rospy.logdebug('Failed to process right image')
      self.raw_right = None
  
  def cb_rect_left(self, msg):
    try:
      self.rect_left = self.bridge.imgmsg_to_cv2(msg, 'mono8')
    except:
      rospy.logdebug('Failed to process left image')
      self.rect_left = None
  
  def cb_rect_right(self, msg):
    try:
      self.rect_right = self.bridge.imgmsg_to_cv2(msg, 'mono8')
    except:
      rospy.logdebug('Failed to process right image')
      self.rect_right = None
  
  def cloud_available(self):
    return (self.point_cloud is not None)
  
  def images_available(self):
    raw_available = (self.raw_left is not None) and (self.raw_right is not None)
    rect_available =(self.rect_left is not None) and (self.rect_right is not None)
    return raw_available and rect_available
  
  def execute(self):
    # Start streaming
    self.dynclient.update_configuration({'Cloud':True, 'Images':True, 'Projector':False})
    rospy.sleep(self.exposure_time)
    taken = 0
    while taken < self.nviews and not rospy.is_shutdown():
      self.reset_snapshots()
      while not (self.cloud_available() and self.images_available()):
        rospy.sleep(0.01)
        if rospy.is_shutdown():
          return
      # Locate the A3 page
      image = self.raw_left
      smooth = cv2.medianBlur(image, 7)
      dbg_image = cv2.cvtColor(smooth, cv2.COLOR_GRAY2RGB)
      # Contours
      (thresh, bw) = cv2.threshold(smooth, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      _, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      # Find the biggest rectangle in the image
      max_area = -float('inf')
      rect = None
      for i,cnt in enumerate(contours):
        # Check only promising contours
        epsilon = 0.05*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
          cnt_area = cv2.contourArea(cnt)
          if cnt_area > max_area:
            max_area = cnt_area
            rect = cv2.boundingRect(cnt)
      if rect is None:
        continue
      x,y,w,h = rect
      cv2.rectangle(dbg_image, (x,y), (x+w,y+h), (0,255,0), 4)
      # Crop the image
      roi = image[y:y+h, x:x+w]
      success, centers = cv2.findCirclesGrid(roi, (5, 3), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
      if centers is None:
        continue
      for center in np.uint16( np.squeeze(centers) ):
        cv2.circle(dbg_image, tuple(center), 5, (255,0,0), 10)
      # Publish debugging image
      self.dbg_pub.publish( self.bridge.cv2_to_imgmsg(dbg_image, 'rgb8') )
  
  def on_shutdown(self):
    # Stop streaming
    self.dynclient.update_configuration({'Cloud':False, 'Images':False, 'Projector':False})
  
  def reset_snapshots(self):
    self.point_cloud = None
    self.raw_left = None
    self.raw_right = None
    self.rect_left = None
    self.rect_right = None


def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=textwrap.dedent(
"""Captures data appropriate for training object recognition pipelines.
Assumes that there is a known fiducial in the scene, and captures views of the 
object sparsely, depending on the angle_thresh setting."""),
                                  fromfile_prefix_chars='@')
  parser.add_argument('-o', '--output', metavar='BAG_FILE', dest='bag', type=str,
                      default='',
                      help='A bagfile to write to.')
  parser.add_argument('-a', '--angle_thresh', metavar='RADIANS', dest='angle_thresh', type=float,
                      default= 10 * math.pi / 180, #10 degrees
                      help='The delta angular threshold in pose.'
                           'Frames will not be recorded unless they are not closer to any other pose by this amount. default(%(default)s)')
  parser.add_argument('-n', '--nviews', metavar='NVIEWS', dest='nviews', type=int,
                      default=36,
                      help='Number of desired views. default(%(default)s)')
  parser.add_argument('-p', '--preview', dest='preview', action='store_true',
                      default=False, help='Preview the pose estimator.')
  parser.add_argument('--debug', action='store_true',     
                      help='If set, will show additional debugging information')
  args = parser.parse_args(rospy.myargv()[1:])
  if not args.preview and len(args.bag) < 1:
    parser.print_help()
    print '\nYou must supply a bag name, or run in --preview mode'
    sys.exit(1)
  return args

if "__main__" == __name__:
  options = parse_args()
  log_level= rospy.DEBUG if options.debug else rospy.INFO
  node_name = os.path.splitext(os.path.basename(__file__))[0]
  rospy.init_node(node_name, log_level=log_level)
  server = CaptureServer(options)
  server.execute()
