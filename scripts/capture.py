#!/usr/bin/env python
import os
import sys
import copy
import rospy
import rosbag
import criros
import argparse
import textwrap
import numpy as np
import dynamic_reconfigure.client
import tf.transformations as tr
# Progress bar
import progressbar
# OpenCV
import cv2
# Python ensenso snatcher
from ensenso.snatcher import Snatcher
# Messages
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2


class CaptureServer(Snatcher):
  def __init__(self, options):
    super(CaptureServer, self).__init__(use_cv_types=False)
    # Config stuff
    np.set_printoptions(precision=6, suppress=True)
    self.ns = rospy.get_namespace()
    self.nviews = options.nviews
    self.preview = options.preview
    self.angle_thresh = options.angle_thresh
    self.observations = []
    # Create rosbag if needed
    if not self.preview:
      self.bag = rosbag.Bag(options.bag, 'w')
    # Setup publishers and subscribers
    rospy.Subscriber('pattern/pose', PoseStamped, self.cb_pattern_pose)
    rospy.on_shutdown(self.on_shutdown)
  
  def cb_pattern_pose(self, msg):
    self.pose_msg = msg
    self.pattern_pose = criros.conversions.from_pose(msg.pose)
    self.headers['pattern_pose'] = copy.deepcopy(msg.header)
  
  def execute(self):
    # Start streaming
    self.enable_lights(projector=True, frontlight=False)
    self.enable_streaming(cloud=True, images=True)
    rospy.sleep(2.0)
    pbar = progressbar.ProgressBar(widgets=['Pattern observations: ', progressbar.SimpleProgress()], maxval=self.nviews).start()
    while len(self.observations) < self.nviews and not rospy.is_shutdown():
      # Take snapshot
      self.reset_snapshots()
      self.take_snapshot(exposure_time=0.001, success_fn=self.has_images_and_cloud)
      # Add observations that are theta_delta apart.
      if not self.received_sync_msgs():
        rospy.logdebug('The headers of the images and pattern pose must be equal')
        continue
      if not self.process_observation(self.pattern_pose):
        continue
      # Write rosbag
      if not self.preview:
        stamp = self.rect_left.header.stamp
        self.bag.write('{0}left/image_rect'.format(self.ns),  self.rect_left, stamp)
        self.bag.write('{0}right/image_rect'.format(self.ns), self.rect_right, stamp)
        self.bag.write('{0}depth/points'.format(self.ns), self.point_cloud, stamp)
        self.bag.write('{0}pattern/pose'.format(self.ns), self.pose_msg, stamp)
      pbar.update(len(self.observations))
    pbar.finish()
  
  def process_observation(self, Tpattern):
    min_delta = float('inf')
    novel = False
    for observation in self.observations:
      Rdelta = np.dot( observation[:3,:3], np.transpose(Tpattern[:3,:3]) )
      delta_vect, _ = cv2.Rodrigues(Rdelta)
      theta_delta = tr.vector_norm(delta_vect)
      if (theta_delta < min_delta):
        min_delta = theta_delta
    if min_delta > (self.angle_thresh / 2.):
      novel = True
      self.observations.append(Tpattern)
    return novel
  
  def on_shutdown(self):
    # Stop streaming
    self.enable_lights(projector=False, frontlight=False)
    self.enable_streaming(cloud=False, images=False)
    # Close rosbag
    self.bag.close()
  
  def received_sync_msgs(self):
    required_keys = ['raw_left', 'raw_right', 'rect_left', 'rect_right', 'pattern_pose']
    if not criros.utils.has_keys(self.headers, required_keys):
      return False
    stamps = []
    frames = []
    for header in self.headers.values():
      stamps.append( header.stamp.to_sec() )
      frames.append( header.frame_id )
    all_frames_equal = frames.count(frames[0]) == len(frames)
    all_stamps_equal = np.all([ np.isclose(stamps[0],x,atol=1e-3) for x in stamps ])
    return (all_frames_equal and all_stamps_equal)


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
                      default= 10 * np.pi / 180,
                      help='The delta angular threshold in pose.'
                           'Frames will not be recorded unless they apart to any other pose by this amount. default=%(default).2f')
  parser.add_argument('-n', '--nviews', metavar='NVIEWS', dest='nviews', type=int,
                      default=36,
                      help='Number of desired views. default=%(default)d')
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
