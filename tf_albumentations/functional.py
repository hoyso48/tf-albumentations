import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons import image as image_ops

from tf_albumentations import bboxes as B

_IMAGE_INTERPOLATION = 'bilinear'
_MASK_INTERPOLATION = 'nearest'

##########################
#basic image ops

def wrap(image):
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], 2)
  return extended


def unwrap(image, replace):
  """Unwraps an image produced by wrap.
  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.
  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.
  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = flattened_image[:, -1]

  replace = tf.ones_like(image[0,0,:-1]) * replace
  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
    tf.equal(alpha_channel, 0)[...,None],
    tf.ones_like(flattened_image, dtype=image.dtype) * replace,
    flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], image_shape[2]-1])
  return image
  
def blend(image1, image2, factor):
  """Blend image1 and image2 using 'factor'.
  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.
  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.
  Returns:
    A blended image Tensor of type uint8.
  """
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1_ = tf.cast(image1, tf.float32)
  image2_ = tf.cast(image2, tf.float32)

  difference = image2_ - image1_
  scaled = tf.cast(factor, tf.float32) * difference

  # Do addition in float.
  temp = image1_ + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, image2.dtype)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), image2.dtype)



###############################
#visual transforms
#image only transforms

def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  image = tf.where(image < tf.cast(threshold, image.dtype), image, 255 - image)
  return image

def solarize_add(image, addition=0, threshold=128):
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  added_image = image + tf.cast(addition, image.dtype)
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), image.dtype)
  image = tf.where(image < tf.cast(threshold, image.dtype), added_image, image)
  return image

def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  image = blend(degenerate, image, factor)
  return image

def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, image.dtype))
  image = blend(degenerate, image, factor)
  return image

def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  image = image
  degenerate = tf.zeros_like(image)
  image = blend(degenerate, image, factor)
  return image

def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = tf.cast(8 - bits, tf.int32)
  shifted_image = tf.bitwise.left_shift(tf.bitwise.right_shift(tf.cast(image, tf.int32), shift), shift)
  shifted_image = tf.cast(shifted_image, image.dtype)
  return shifted_image

def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops.
  Args:
    image: A 3D uint8 tensor.
  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  image = tf.transpose(tf.map_fn(scale_channel, tf.transpose(image, [2,0,1], image.dtype)), [1,2,0])
  return image

def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  image = image
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID')
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  image = blend(result, orig_image, factor)
  return image

def equalize(image):
  """Implements Equalize function from PIL using TF ops."""
  def scale_channel(im):
    """Scale the data in the channel to implement equalize."""
    # Compute the histogram of the image channel.
    im = tf.cast(im, tf.int32)
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(tf.equal(step, 0),
                     lambda: im,
                     lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  image = tf.transpose(tf.map_fn(scale_channel, tf.transpose(image, [2,0,1], image.dtype)), [1,2,0])
  return image

def invert(image):
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  image = 255 - image
  return image

def gamma(image, pow):
  image = tf.image.adjust_gamma(image, pow, gain=1.)
  return image

def hue(image, delta):
  image = tf.image.adjust_hue(image, delta)
  return image

def saturation(image, factor):
  image = tf.image.adjust_saturation(image, factor)
  return image

def rgb_shift_scale(image, r_shift, g_shift, b_shift, r_scale, g_scale, b_scale):
  dt = image.dtype
  n_channels = tf.shape(image)[-1]
  tf.debugging.assert_equal(n_channels, 3)
  image = tf.cast(image, tf.float32)/255.
  image = image * tf.convert_to_tensor([r_scale, g_scale, b_scale], dtype=tf.float32)
  image = image + tf.convert_to_tensor([r_shift, g_shift, b_shift], dtype=tf.float32)
  image = tf.clip_by_value(image, tf.constant(0, dtype=tf.float32), tf.constant(1, dtype=tf.float32))
  image = tf.cast(image * 255, dt)
  return image

def hsv_shift_scale(image, h_shift, s_shift, v_shift, h_scale, s_scale, v_scale):
  dt = image.dtype
  n_channels = tf.shape(image)[-1]
  tf.debugging.assert_equal(n_channels, 3)
  image = tf.cast(image, tf.float32)/255.
  image = tf.image.rgb_to_hsv(image)
  image = image * tf.convert_to_tensor([h_scale, s_scale, v_scale], dtype=tf.float32)
  image = image + tf.convert_to_tensor([h_shift, s_shift, v_shift], dtype=tf.float32)
  image = tf.clip_by_value(image, tf.constant(0, dtype=tf.float32), tf.constant(1, dtype=tf.float32))
  image = tf.image.hsv_to_rgb(image)
  image = tf.cast(image * 255, dt)
  return image

def gaussian_noise(image, mean, std, per_channel=True):
  if per_channel:
    shape = tf.shape(image)
  else:
    shape = tf.concat(tf.shape(image)[:2], [1])
  noise = tf.random.normal(shape, tf.cast(mean, tf.float32), tf.cast(std, tf.float32))
  image = tf.cast(tf.clip_by_value(tf.cast(image, tf.float32) + noise, 0, 255), image.dtype)
  return image

def channel_shuffle(image):
  n_channels = tf.shape(image)[-1]
  shuffled_channels = tf.random.shuffle(range(n_channels))
  shuffled_image = tf.gather(image, shuffled_channels, axis = -1)
  image = shuffled_image
  return image




################################
#sparse-geometric transforms
#image, mask, bboxes transforms

def cutout(image, mask, objects, pad_size, replace=0):
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `img`. The pixel values filled in will be of the
  value `replace`. The located where the mask will be applied is randomly
  chosen uniformly over the whole image.
  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.
  Returns:
    An image Tensor that is of type uint8.
  """
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height,
      dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width,
      dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [image_height - (lower_pad + upper_pad),
                  image_width - (left_pad + right_pad)]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  cutout = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims, constant_values=1)
  cutout = tf.expand_dims(cutout, -1)
  image = tf.where(
      tf.equal(cutout, 0),
      tf.ones_like(image, dtype=image.dtype) * replace,
      image)
  if mask is not None:
    mask = tf.where(
      tf.equal(cutout, 0),
      tf.zeros_like(mask, dtype=mask.dtype),
      mask)
  if objects is not None:
      offset = (lower_pad/image_height, left_pad/image_width)
      size = (cutout_shape[0]/image_height, cutout_shape[1]/image_width)
      objects = B.cutout_bboxes(objects, offset, size)
  return image, mask, objects

def rotate(image, mask, objects, degrees, replace=0):
  """Rotates the image by degrees either clockwise or counterclockwise.
  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
    replace: A one or three value 1D tensor to fill empty pixels caused by
      the rotate operation.
  Returns:
    The rotated version of image.
  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = tf.cast(degrees, tf.float32) * degrees_to_radians

  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = image_ops.rotate(wrap(image), radians, interpolation=_IMAGE_INTERPOLATION)
  image = unwrap(image, replace)
  if mask is not None:
    mask = image_ops.rotate(mask, radians, interpolation=_MASK_INTERPOLATION)
  if objects is not None:
    objects = B.rotate_bboxes(objects, radians)
  return image, mask, objects


def translate_x(image, mask, objects, level, replace=0):
  """Equivalent of PIL Translate in X dimension."""
  pixels = tf.cast(tf.cast(tf.shape(image)[1], tf.float32) * level, tf.int32)
  image = image_ops.translate(wrap(image), [-pixels, 0])
  image = unwrap(image, replace)
  if mask is not None:
    mask = image_ops.translate(mask, [-pixels, 0])
  if objects is not None:
    image_width = tf.shape(image)[1]
    objects = B.translate_bboxes(objects,0,pixels/image_width)
  return image, mask, objects


def translate_y(image, mask, objects, level, replace=0):
  """Equivalent of PIL Translate in Y dimension."""
  pixels = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * level, tf.int32)
  image = image_ops.translate(wrap(image), [0,-pixels])
  image = unwrap(image, replace)
  if mask is not None:
    mask = image_ops.translate(mask, [0,-pixels])
  if objects is not None:
    image_height = tf.shape(image)[0]
    objects = B.translate_bboxes(objects,pixels/image_height,0)
  return image, mask, objects


def shear_x(image, mask, objects, level, replace=0):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = image_ops.transform(
      wrap(image), [1., level, 0., 0., 1., 0., 0., 0.], interpolation=_IMAGE_INTERPOLATION)
  image = unwrap(image, replace)
  if mask is not None:
    mask = image_ops.transform(
      mask, [1., level, 0., 0., 1., 0., 0., 0.], interpolation=_MASK_INTERPOLATION)
  if objects is not None:
    objects = B.shear_bboxes(objects, 0, level)
  return image, mask, objects

def shear_y(image, mask, objects, level, replace=0):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = image_ops.transform(
      wrap(image), [1., 0., 0., level, 1., 0., 0., 0.], interpolation=_IMAGE_INTERPOLATION)
  image = unwrap(image, replace)
  if mask is not None:
    mask = image_ops.transform(
      mask, [1., 0., 0., level, 1., 0., 0., 0.], interpolation=_MASK_INTERPOLATION)
  if objects is not None:
    objects = B.shear_bboxes(objects, level, 0)
  return image, mask, objects

def scale(image, mask, objects, scale, replace=0):
  #scale(zoom-in, zoom-out) image with scale while maintaining aspect ratio
  image_shape = tf.shape(image)
  mask_shape = tf.shape(mask)
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  resize_h = tf.cast(tf.cast(image_height, tf.float32)*scale, tf.int32)
  resize_w = tf.cast(tf.cast(image_width, tf.float32)*scale, tf.int32)
  offset_y = tf.random.uniform((),0,tf.abs(image_height-resize_h),tf.int32)
  offset_x = tf.random.uniform((),0,tf.abs(image_width-resize_w), tf.int32)
  image = tf.cast(tf.image.resize(image, [resize_h, resize_w], method=_IMAGE_INTERPOLATION), image.dtype)
  pad = [[offset_y,image_height-offset_y-resize_h],[offset_x,image_width-offset_x-resize_w],[0,0]]
#   if mask is not None:
#     mask = tf.cast(tf.image.resize(mask, [resize_h, resize_w], 'nearest'), mask.dtype)
  
  if scale > 1:
    image = tf.slice(image, [offset_y,offset_x,0], [image_height,image_width,-1])
  else:
    image = tf.pad(wrap(image), pad)
    image = unwrap(image, replace)
    image = tf.reshape(image, image_shape)
  if mask is not None:
    mask = tf.cast(tf.image.resize(mask, [resize_h, resize_w], method=_MASK_INTERPOLATION), mask.dtype)
    if scale > 1:
      mask = tf.slice(mask, [offset_y,offset_x,0], [image_height,image_width,-1])
    else:
      mask = tf.pad(mask, pad)
    mask = tf.reshape(mask, mask_shape)
  if objects is not None:
    objects = B.scale_bboxes(objects, scale, scale, offset_y/image_height, offset_x/image_width)
  return image, mask, objects

def flip_left_right(image, mask, objects):
  image = tf.image.flip_left_right(image)
  if mask is not None:
    mask = tf.image.flip_left_right(mask)
  if objects is not None:
    objects = B.flip_left_right_bboxes(objects)
  return image, mask, objects

def flip_up_down(image, mask, objects):
  image = tf.image.flip_up_down(image)
  if mask is not None:
    mask = tf.image.flip_up_down(mask)
  if objects is not None:
    objects = B.flip_up_down_bboxes(objects)
  return image, mask, objects




################################
#dense-geometric transforms
#image, mask transforms

#base op for dense image warping
def tf_cv2_remap(image, map1, map2, interpolation='bilinear', border_mode='reflect'):
    H, W, C = tf.unstack(tf.shape(image))
    map1 = tf.cast(map1, tf.float32)
    map2 = tf.cast(map2, tf.float32)
    H = tf.cast(H, tf.float32)
    W = tf.cast(W, tf.float32)
    if border_mode == 'constant':
        # map2 = tf.where(tf.logical_or(map2<0, map2>H-1), H, map2)
        # map1 = tf.where(tf.logical_or(map1<0, map1>W-1), W, map1)
        # map2 = tf.pad(map2, [0,1], )
        # image = tf.pad(image, [[0,1],[0,1],[0,0]])
        mask_y = tf.logical_and(map2>=0, map2<H-1)
        mask_x = tf.logical_and(map1>=0, map1<W-1)
        mask = tf.cast(tf.logical_and(mask_y, mask_x), image.dtype)
        map2 = tf.where(mask_y , map2, 0)
        map1 = tf.where(mask_x , map1, 0)
    elif border_mode == 'reflect':
        map2 = tf.where(map2<0, -map2, map2)
        map1 = tf.where(map1<0, -map1, map1)
        map2 = tf.where(map2>H-1, H-1-map2%(H-1), map2)
        map1 = tf.where(map1>W-1, W-1-map1%(W-1), map1)
        mask = tf.ones_like(image[...,0])
    else: raise NotImplementedError

    if interpolation=='nearest':
        map = tf.stack([map2, map1], -1)
        map = tf.cast(map, tf.int32)
        result = tf.gather_nd(image, map)

    elif interpolation=='bilinear':
        map = tf.stack([map2, map1], -1)
        x, y = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32))
        origin = tf.stack([y,x], -1)
        map = origin - map
        result = tfa.image.dense_image_warp(tf.cast(image[None,...], tf.float32), map[None,...])[0]

        # x1 = tf.cast(tf.math.floor(map1), tf.int32)
        # x2 = tf.cast(tf.math.ceil(map1), tf.int32)
        # y1 = tf.cast(tf.math.floor(map2), tf.int32)
        # y2 = tf.cast(tf.math.ceil(map2), tf.int32)
        # q11 = tf.cast(tf.gather_nd(image, tf.stack([y1, x1],-1)), tf.float32)
        # q21 = tf.cast(tf.gather_nd(image, tf.stack([y2, x1],-1)), tf.float32)
        # q12 = tf.cast(tf.gather_nd(image, tf.stack([y1, x2],-1)), tf.float32)
        # q22 = tf.cast(tf.gather_nd(image, tf.stack([y2, x2],-1)), tf.float32)

        # x2_x = tf.clip_by_value(map1-tf.math.floor(map1), 0, 1)
        # x_x1 = tf.clip_by_value(1-x2_x, 0, 1)
        # y2_y = tf.clip_by_value(map2-tf.math.floor(map2), 0, 1)
        # y_y1 = tf.clip_by_value(1-y2_y, 0, 1)

        # result = ((x2_x * y2_y)[...,None] * q11 +
        #         (x_x1 * y2_y)[...,None] * q21 +
        #         (x2_x * y_y1)[...,None] * q12 +
        #         (x_x1 * y_y1)[...,None] * q22)
    else: raise NotImplementedError

    return tf.cast(result, image.dtype) * mask[...,None]

def optical_distortion(image, mask, k, replace=0, border_mode='reflect'):
    H, W = tf.unstack(tf.cast(tf.shape(image), tf.float32)[:2])
    x, y = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32))        
    center_x = W/2
    center_y = H/2
    x = (x - center_x)/W
    y = (y - center_y)/H
    r = x**2 + y**2
    dist = 1 + k*r**2 + k*r
    x = x*dist
    y = y*dist
    x = W*x+center_x
    y = H*y+center_y
    image = tf_cv2_remap(wrap(image), x, y, _IMAGE_INTERPOLATION, border_mode)
    image = unwrap(image, replace)
    if mask is not None:
        mask = tf_cv2_remap(mask, x, y, _MASK_INTERPOLATION, border_mode)
    return image, mask

def grid_distortion(image, mask, distort=0.5, num_steps=5, replace=0, border_mode='reflect'):
    height, width = tf.unstack(tf.shape(image)[:2])

    x_step = width // num_steps
    y_step = height // num_steps
    stepsx = 1 + tf.random.uniform((num_steps+1,), -distort, distort)
    stepsy = 1 + tf.random.uniform((num_steps+1,), -distort, distort)
    xx = tf.zeros((width,), tf.float32)
    prev = 0
    for idx in range(num_steps + 1):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + tf.cast(tf.cast(x_step, tf.float32) * stepsx[idx], tf.int32)

        xx = tf.tensor_scatter_nd_update(xx, tf.range(start, end)[...,None], tf.cast(tf.linspace(prev, cur, end - start),tf.float32))
        prev = cur

    yy = tf.zeros((height,), tf.float32)
    prev = 0
    for idx in range(num_steps + 1):
        x = idx * y_step
        start = int(x)
        end = int(x) +y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + tf.cast(tf.cast(y_step,tf.float32) * stepsy[idx], tf.int32)

        yy = tf.tensor_scatter_nd_update(yy, tf.range(start, end)[...,None], tf.cast(tf.linspace(prev, cur, end - start),tf.float32))
        prev = cur

    map_x, map_y = tf.meshgrid(xx, yy)
    image = tf_cv2_remap(wrap(image), map_x, map_y, _IMAGE_INTERPOLATION, border_mode)
    image = unwrap(image, replace)
    if mask is not None:
        mask = tf_cv2_remap(mask, map_x, map_y, _MASK_INTERPOLATION, border_mode)
    return image, mask

def elastic_transform(image, mask, alpha=1, sigma=50, alpha_affine=50, replace=0, border_mode='reflect'):
    shape = tf.shape(image)
    shape_size = shape[:2]

    # Random affine
    center_square = shape_size // 2
    square_size = tf.minimum(shape_size[0], shape_size[1]) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = tf.convert_to_tensor([center_square + square_size, 
                                [center_square[0] + square_size, center_square[1] - square_size],
                                center_square - square_size], dtype=tf.float32)
    pts2 = pts1 + tf.random.uniform(tf.shape(pts1), -alpha_affine, alpha_affine)#random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    dx = tf.random.uniform(shape_size)*2-1
    dy = tf.random.uniform(shape_size)*2-1
    # truncate = 4
    kernel_size = 17 #min(2*int(truncate*sigma + 0.5) + 1, 17)
    dx = tfa.image.gaussian_filter2d(dx,kernel_size,sigma,padding='reflect') * alpha
    dy = tfa.image.gaussian_filter2d(dy,kernel_size,sigma,padding='reflect') * alpha
    x, y = tf.meshgrid(tf.range(shape[1], dtype=tf.float32), tf.range(shape[0], dtype=tf.float32))
    map_x = x+dx
    map_y = y+dy

    image = wrap(image)
    image = tf.cast(tfa.image.sparse_image_warp(tf.cast(image[None,...],tf.float32), pts1[None,...], pts2[None,...],interpolation_order=1)[0][0], image.dtype)
    image = tf_cv2_remap(image, map_x, map_y, _IMAGE_INTERPOLATION, border_mode)
    image = unwrap(image, replace)
    if mask is not None:
        mask = tf.cast(tfa.image.sparse_image_warp(tf.cast(mask[None,...],tf.float32), pts1[None,...], pts2[None,...],interpolation_order=0)[0][0], mask.dtype)
        mask = tf_cv2_remap(mask, map_x, map_y, _MASK_INTERPOLATION, border_mode)
    return image, mask