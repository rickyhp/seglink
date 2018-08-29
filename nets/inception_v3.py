import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops

slim = tf.contrib.slim

depth_multiplier = 1
min_depth = 1
final_endpoint = 'Mixed_7c'

def basenet(inputs):
    end_points = {}
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
          # 299 x 299 x 3
          end_point = 'Conv2d_1a_3x3'
          net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
              return net, end_points
          # 149 x 149 x 32
          end_point = 'Conv2d_2a_3x3'
          net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
              return net, end_points
          # 147 x 147 x 32
          end_point = 'Conv2d_2b_3x3'
          net = slim.conv2d(
              net, depth(64), [3, 3], padding='SAME', scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
              return net, end_points
          # 147 x 147 x 64
          end_point = 'MaxPool_3a_3x3'
          net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
              return net, end_points
          # 73 x 73 x 64
          end_point = 'Conv2d_3b_1x1'
          net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
              return net, end_points
          # 73 x 73 x 80.
          #end_point = 'Conv2d_4a_3x3'
          end_point = 'conv4_3'
          net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
              return net, end_points
          # 71 x 71 x 192.
          end_point = 'MaxPool_5a_3x3'
          net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
              return net, end_points
          # 35 x 35 x 192.
        
          # Inception blocks
          with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
              # mixed: 35 x 35 x 256.
              end_point = 'Mixed_5b'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = slim.conv2d(
                      branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.conv2d(
                      net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                  branch_2 = slim.conv2d(
                      branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with variable_scope.variable_scope('Branch_3'):
                  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                  branch_3 = slim.conv2d(
                      branch_3, depth(32), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
            
              # mixed_1: 35 x 35 x 288.
              end_point = 'Mixed_5c'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                  branch_1 = slim.conv2d(
                      branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.conv2d(
                      net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                  branch_2 = slim.conv2d(
                      branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with variable_scope.variable_scope('Branch_3'):
                  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                  branch_3 = slim.conv2d(
                      branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
            
              # mixed_2: 35 x 35 x 288.
              end_point = 'Mixed_5d'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = slim.conv2d(
                      branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.conv2d(
                      net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                  branch_2 = slim.conv2d(
                      branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with variable_scope.variable_scope('Branch_3'):
                  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                  branch_3 = slim.conv2d(
                      branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
            
              # mixed_3: 17 x 17 x 768.
              end_point = 'Mixed_6a'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net,
                      depth(384), [3, 3],
                      stride=2,
                      padding='VALID',
                      scope='Conv2d_1a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = slim.conv2d(
                      branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                  branch_1 = slim.conv2d(
                      branch_1,
                      depth(96), [3, 3],
                      stride=2,
                      padding='VALID',
                      scope='Conv2d_1a_1x1')
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.max_pool2d(
                      net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = array_ops.concat([branch_0, branch_1, branch_2], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
            
              # mixed4: 17 x 17 x 768.
              end_point = 'Mixed_6b'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = slim.conv2d(
                      branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
                  branch_1 = slim.conv2d(
                      branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.conv2d(
                      net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
                  branch_2 = slim.conv2d(
                      branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                  branch_3 = slim.conv2d(
                      branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
            
              # mixed_5: 17 x 17 x 768.
              end_point = 'Mixed_6c'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = slim.conv2d(
                      branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                  branch_1 = slim.conv2d(
                      branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.conv2d(
                      net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                  branch_2 = slim.conv2d(
                      branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                  branch_3 = slim.conv2d(
                      branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
              # mixed_6: 17 x 17 x 768.
              end_point = 'Mixed_6d'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = slim.conv2d(
                      branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                  branch_1 = slim.conv2d(
                      branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.conv2d(
                      net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                  branch_2 = slim.conv2d(
                      branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                  branch_3 = slim.conv2d(
                      branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
            
              # mixed_7: 17 x 17 x 768.
              end_point = 'Mixed_6e'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = slim.conv2d(
                      branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                  branch_1 = slim.conv2d(
                      branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.conv2d(
                      net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
                  branch_2 = slim.conv2d(
                      branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with variable_scope.variable_scope('Branch_3'):
                  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                  branch_3 = slim.conv2d(
                      branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
            
              # mixed_8: 8 x 8 x 1280.
              end_point = 'Mixed_7a'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                  branch_0 = slim.conv2d(
                      branch_0,
                      depth(320), [3, 3],
                      stride=2,
                      padding='VALID',
                      scope='Conv2d_1a_3x3')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = slim.conv2d(
                      branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                  branch_1 = slim.conv2d(
                      branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                  branch_1 = slim.conv2d(
                      branch_1,
                      depth(192), [3, 3],
                      stride=2,
                      padding='VALID',
                      scope='Conv2d_1a_3x3')
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.max_pool2d(
                      net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = array_ops.concat([branch_0, branch_1, branch_2], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
              # mixed_9: 8 x 8 x 2048.
              end_point = 'Mixed_7b'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = array_ops.concat(
                      [
                          slim.conv2d(
                              branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                          slim.conv2d(
                              branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')
                      ],
                      3)
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.conv2d(
                      net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                  branch_2 = array_ops.concat(
                      [
                          slim.conv2d(
                              branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                          slim.conv2d(
                              branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                      ],
                      3)
                with variable_scope.variable_scope('Branch_3'):
                  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                  branch_3 = slim.conv2d(
                      branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
            
              # mixed_10: 8 x 8 x 2048.
              end_point = 'Mixed_7c'
              with variable_scope.variable_scope(end_point):
                with variable_scope.variable_scope('Branch_0'):
                  branch_0 = slim.conv2d(
                      net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with variable_scope.variable_scope('Branch_1'):
                  branch_1 = slim.conv2d(
                      net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                  branch_1 = array_ops.concat(
                      [
                          slim.conv2d(
                              branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                          slim.conv2d(
                              branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')
                      ],
                      3)
                with variable_scope.variable_scope('Branch_2'):
                  branch_2 = slim.conv2d(
                      net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                  branch_2 = slim.conv2d(
                      branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                  branch_2 = array_ops.concat(
                      [
                          slim.conv2d(
                              branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                          slim.conv2d(
                              branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                      ],
                      3)
                with variable_scope.variable_scope('Branch_3'):
                  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                  branch_3 = slim.conv2d(
                      branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
              end_points[end_point] = net
              if end_point == final_endpoint:
                return net, end_points
          raise ValueError('Unknown final endpoint %s' % final_endpoint)