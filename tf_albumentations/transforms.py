import tensorflow as tf
import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tf_albumentations import functional as F

def _apply_func_with_params(func, params, **kwargs):
  #set value None for function args not in kwargs
  func_args = kwargs.copy()
  for arg in inspect.getfullargspec(func).args:
    if arg not in kwargs.keys():
      if arg in params.keys():
        func_args[arg] = params[arg]
      else:
        if not arg == 'self':
            func_args[arg] = None
  #do nothing and pass if kwargs not in func
  for arg in kwargs.keys():
    if arg not in inspect.getfullargspec(func).args:
      del func_args[arg]
  #update if output exists
  output = func(**func_args)
  output_dict = kwargs.copy()
  for key in output.keys():
    if key in kwargs.keys():
        output_dict[key] = output[key]
  return output_dict

def _parse_arg(param, log_scale=False):
    #if param is tuple of length 2, return random value in between
    #if param is list, return random element of the list
    #if param is const, return itself
    if isinstance(param, tuple):
        assert len(param)==2
        param = tf.convert_to_tensor(param, dtype=tf.float32)
        if log_scale:
#           tf.debugging.assert_greater(param[0],0.)
#           tf.debugging.assert_greater(param[1],0.)
          p = tf.random.uniform((), tf.math.log(param[0]), tf.math.log(param[1]), tf.float32)
          return tf.exp(p)
        else:
          return tf.random.uniform((), param[0], param[1], tf.float32)
    elif isinstance(param, list):
        assert len(param)>0
        param = tf.convert_to_tensor(param)
        selection = tf.random.uniform((), 0, len(param), tf.int32) 
        return param[selection]
    elif isinstance(param, (int, float)):
        return param
    elif param is None:
        return param
    else: raise ValueError

def tf_random_choice(a, size, replace=True, p=None):
    if replace:
        if p is None:
            idxs = tf.random.uniform((size,), 0, len(a), dtype=tf.int32)
        else:
            cum_dist = tf.cast(tf.math.cumsum(p),tf.float32)
            cum_dist /= cum_dist[-1]  # to account for floating point errors
            unif_samp = tf.random.uniform((size,), 0, 1)
            idxs = tf.searchsorted(cum_dist, unif_samp)
    else:
        assert size <= len(a)
        if p is None:
            idxs = tf.random.shuffle(tf.range(len(a)))[:size]
        else:
            p = tf.cast(p/tf.reduce_sum(tf.cast(p, tf.float32)), tf.float32)
            unif_samp = tf.random.uniform((len(a),), 0, 1)
            unif_samp = tf.math.log(unif_samp)/p
            idxs = tf.math.top_k(unif_samp, k=size, sorted=False).indices
    return idxs

class Transform:
    def __init__(self, p=0, debug=False):
        self.p=p
#         self.p = p
#         self.debug = debug
    def apply(self):
        return
    def get_params(self):
        return {}
    def __call__(self, **kwargs):
        params = self.get_params()
        if self.p == 0:
            return kwargs
        elif self.p == 1:
            return _apply_func_with_params(self.apply, params, **kwargs)
        else:
            p = tf.random.uniform(())
            if p < self.p:
#                 if self.debug:
#                   print()
#                   print(self, f'applied with prob {p} < {self.p}')
                return _apply_func_with_params(self.apply, params, **kwargs)
            else:
#                 if self.debug:
#                   print()
#                   print(self, f'skipped with prob {p} >= {self.p}')
                return kwargs

class NoOp(Transform):
    def __init__(self, p=1.0):
        self.p = p
    def __call__(self, **kwargs):
        return kwargs

class Choice(Transform):
    def __init__(self, transforms:List[Transform], p=1, n=1, sample_weights=None, replace=True, remain_order=False):
        self.transforms = transforms
        self.p = p
        self.n = n
        self.sample_weights = sample_weights
        self.replace = replace
        self.remain_order = remain_order

    def __call__(self, **kwargs):
        if (self.p == 0) or (len(self.transforms)==0):
            return kwargs
        else:
            idxs = tf_random_choice(tf.range(len(self.transforms)), self.n, self.replace, self.sample_weights)
            if tf.random.uniform(()) < self.p:
                if self.remain_order:
                    idxs = tf.sort(idxs)
                for k in range(self.n):
                    for (i, func) in enumerate([x.__call__ for x in self.transforms]):
                        # if i==idxs[k]:
                        #     print('selects', i)
                        kwargs = tf.cond(
                            tf.equal(i, idxs[k]),
                            lambda selected_func=func, selected_args=kwargs: selected_func(**selected_args),
                            lambda: kwargs)
            return kwargs

class Sequence(Transform):
    def __init__(self, transforms:List[Transform], p=1):
        self.transforms = transforms
        self.p = p

    def __call__(self, **kwargs):
        if (self.p == 0) or (len(self.transforms)==0):
            return kwargs
        else:
            if tf.random.uniform(()) < self.p:
                for f in self.transforms:
                    kwargs = f(**kwargs)
            return kwargs

class Solarize(Transform):
    def __init__(self, p=0.5, threshold=128):
        self.p = p
        self.threshold = threshold

    def apply(self, image, threshold):
        image = F.solarize(image, threshold)
        return {'image': image}

    def get_params(self):
        return {
            'threshold':_parse_arg(self.threshold)
        }

class SolarizeAdd(Transform):
    def __init__(self, p=0.5, addition=128, threshold=128):
        self.p = p
        self.addition = addition
        self.threshold = threshold

    def apply(self, image, addition, threshold):
        image = F.solarize_add(image, addition, threshold)
        return {'image': image}

    def get_params(self):
        return {
            'addition':_parse_arg(self.addition),
            'threshold':_parse_arg(self.threshold)
        }

class Color(Transform):
    def __init__(self, p=0.5, factor=(0.7,1.3)):
        self.p = p
        self.factor = factor

    def apply(self, image, factor):
        image = F.color(image, factor)
        return {'image': image}

    def get_params(self):
        return {
            'factor':_parse_arg(self.factor)
        }

class Contrast(Transform):
    def __init__(self, p=0.5, factor=(0.7,1.3)):
        self.p = p
        self.factor = factor

    def apply(self, image, factor):
        image = F.contrast(image, factor)
        return {'image': image}

    def get_params(self):
        return {
            'factor':_parse_arg(self.factor)
        }

class Brightness(Transform):
    def __init__(self, p=0.5, factor=(0.7,1.3)):
        self.p = p
        self.factor = factor

    def apply(self, image, factor):
        image = F.brightness(image, factor)
        return {'image': image}

    def get_params(self):
        return {
            'factor':_parse_arg(self.factor)
        }

class Posterize(Transform):
    def __init__(self, p=0.5, bits=4):
        self.p = p
        self.bits = bits

    def apply(self, image, bits):
        image = F.posterize(image, bits)
        return {'image': image}

    def get_params(self):
        return {
            'bits':_parse_arg(self.bits)
        }

class AutoContrast(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def apply(self, image):
        image = F.autocontrast(image)
        return {'image': image}

    def get_params(self):
        return {
        }

class Sharpness(Transform):
    def __init__(self, p=0.5, factor=(0.7,1.3)):
        self.p = p
        self.factor = factor

    def apply(self, image, factor):
        image = F.sharpness(image, factor)
        return {'image': image}

    def get_params(self):
        return {
            'factor':_parse_arg(self.factor)
        }

class Equalize(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def apply(self, image):
        image = F.equalize(image)
        return {'image': image}

    def get_params(self):
        return {
        }

class Invert(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def apply(self, image):
        image = F.invert(image)
        return {'image': image}

    def get_params(self):
        return {
        }

class Gamma(Transform):
    def __init__(self, p=0.5, pow=(0.7,1.3)):
        self.p = p
        self.pow = pow

    def apply(self, image, pow):
        image = F.gamma(image, pow)
        return {'image': image}

    def get_params(self):
        return {
            'pow':_parse_arg(self.pow)
        }

class Hue(Transform):
    def __init__(self, p=0.5, delta=(-0.3,0.3)):
        self.p = p
        self.delta = delta

    def apply(self, image, delta):
        image = F.hue(image, delta)
        return {'image': image}

    def get_params(self):
        return {
            'delta':_parse_arg(self.delta)
        }

class Saturation(Transform):
    def __init__(self, p=0.5, factor=(0.7,1.3)):
        self.p = p
        self.factor = factor

    def apply(self, image, factor):
        image = F.saturation(image, factor)
        return {'image': image}

    def get_params(self):
        return {
            'factor':_parse_arg(self.factor)
        }

class ChannelShuffle(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def apply(self, image):
        image = F.channel_shuffle(image)
        return {'image': image}

    def get_params(self):
        return {
        }

class GaussianNoise(Transform):
    def __init__(self, p=0.5, mean=0, std=(3,6)):
        self.p = p
        self.mean = mean
        self.std = std

    def apply(self, image, mean, std):
        image = F.gaussian_noise(image, mean, std)
        return {'image': image}

    def get_params(self):
        return {
            'mean':_parse_arg(self.mean),
            'std':_parse_arg(self.std)
        }

class RGBJitter(Transform):
    def __init__(self, p=0.5, shift_limit=(-0.2,0.2), scale_limit=(-0.05,0.05)):
        self.p = p
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit

    def apply(self, image, r_shift, g_shift, b_shift, r_scale, g_scale, b_scale):
        image = F.rgb_shift_scale(image, r_shift, g_shift, b_shift, r_scale, g_scale, b_scale)
        return {'image': image}

    def get_params(self):
        return {
            'r_shift':_parse_arg(self.shift_limit),
            'g_shift':_parse_arg(self.shift_limit),
            'b_shift':_parse_arg(self.shift_limit),
            'r_scale':_parse_arg(self.scale_limit),
            'g_scale':_parse_arg(self.scale_limit),
            'b_scale':_parse_arg(self.scale_limit),
        }

class HSVJitter(Transform):
    def __init__(self, p=0.5, shift_limit=(-0.2,0.2), scale_limit=(-0.05,0.05)):
        self.p = p
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit

    def apply(self, image, h_shift, s_shift, v_shift, h_scale, s_scale, v_scale):
        image = F.hsv_shift_scale(image, h_shift, s_shift, v_shift, h_scale, s_scale, v_scale)
        return {'image': image}

    def get_params(self):
        return {
            'h_shift':_parse_arg(self.shift_limit),
            's_shift':_parse_arg(self.shift_limit),
            'v_shift':_parse_arg(self.shift_limit),
            'h_scale':_parse_arg(self.scale_limit),
            's_scale':_parse_arg(self.scale_limit),
            'v_scale':_parse_arg(self.scale_limit),
        }

class HorizontalFlip(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def apply(self, image, mask, objects):
        image, mask, objects = F.flip_left_right(image, mask, objects)
        return {'image':image, 'mask':mask, 'objects':objects}

    def get_params(self):
        return {}

class VerticalFlip(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def apply(self, image, mask, objects):
        image, mask, objects = F.flip_up_down(image, mask, objects)
        return {'image':image, 'mask':mask, 'objects':objects}

    def get_params(self):
        return {}
      
class Rotate(Transform):
    def __init__(self, p=0.5, degrees=(-30,30), replace=0):
        self.p = p
        self.degrees = degrees
        self.replace = replace

    def apply(self, image, mask, objects, degrees, replace):
        image, mask, objects = F.rotate(image, mask, objects, degrees, replace)
        return {'image':image, 'mask':mask, 'objects':objects}

    def get_params(self):
        return {
            'degrees':_parse_arg(self.degrees),
            'replace':self.replace
        }

class ShearX(Transform):
    def __init__(self, p=0.5, level=(-0.3,0.3), replace=0):
        self.p = p
        self.level = level
        self.replace = replace

    def apply(self, image, mask, objects, level, replace):
        image, mask, objects = F.shear_x(image, mask, objects, level, replace)
        return {'image':image, 'mask':mask, 'objects':objects}

    def get_params(self):
        return {
            'level':_parse_arg(self.level),
            'replace':self.replace
        }

class ShearY(Transform):
    def __init__(self, p=0.5, level=(-0.3,0.3), replace=0):
        self.p = p
        self.level = level
        self.replace = replace

    def apply(self, image, mask, objects, level, replace):
        image, mask, objects = F.shear_y(image, mask, objects, level, replace)
        return {'image':image, 'mask':mask, 'objects':objects}

    def get_params(self):
        return {
            'level':_parse_arg(self.level),
            'replace':self.replace
        }

class TranslateX(Transform):
    def __init__(self, p=0.5, level=(-0.3,0.3), replace=0):
        self.p = p
        self.level = level
        self.replace = replace

    def apply(self, image, mask, objects, level, replace):
        image, mask, objects = F.translate_x(image, mask, objects, level, replace)
        return {'image':image, 'mask':mask, 'objects':objects}

    def get_params(self):
        return {
            'level':_parse_arg(self.level),
            'replace':self.replace
        }

class TranslateY(Transform):
    def __init__(self, p=0.5, level=(-0.3,0.3), replace=0):
        self.p = p
        self.level = level
        self.replace = replace

    def apply(self, image, mask, objects, level, replace):
        image, mask, objects = F.translate_y(image, mask, objects, level, replace)
        return {'image':image, 'mask':mask, 'objects':objects}

    def get_params(self):
        return {
            'level':_parse_arg(self.level),
            'replace':self.replace
        }
      
# class Scale(Transform):
#     def __init__(self, p=0.5, scale=(0.7,1.3), centered=False, replace=0):
#         self.p = p
#         self.scale = scale
#         self.centered = centered
#         self.replace = replace

#     def apply(self, image, mask, objects, scale, centered, replace):
#         image, mask, objects = F.scale_preserved(image, mask, objects, scale, centered, replace)
#         return {'image':image, 'mask':mask, 'objects':objects}

#     def get_params(self):
#         return {
#             'scale':_parse_arg(self.scale),
#             'centered':self.centered,
#             'replace':self.replace
#         }
      
class Scale(Transform):
    def __init__(self, p=0.5, scale=(0.7,1.33), aspect_ratio=(0.7,1.33), centered=False, crop_size=None, output_size=None, replace=0):
        self.p = p
        self.scale = scale
        self.crop_size = crop_size
        self.output_size = output_size
        self.aspect_ratio = aspect_ratio
        self.centered = centered
        self.replace = replace

    def apply(self, image, mask, objects, scale, aspect_ratio, centered, crop_size, output_size, replace):
        image, mask, objects = F.scale_(image, mask, objects, scale, aspect_ratio, centered, crop_size, output_size, replace)
        return {'image':image, 'mask':mask, 'objects':objects}

    def get_params(self):
        return {
            'scale':_parse_arg(self.scale),
            'aspect_ratio':_parse_arg(self.aspect_ratio),
            'centered':self.centered,
            'output_size':self.output_size,
            'crop_size':self.crop_size,
            'replace':self.replace
        }
      
class Cutout(Transform):
    def __init__(self, p=0.5, size=100, replace=0):
        self.p = p
        self.size = size
        self.replace = replace

    def apply(self, image, mask, objects, size, replace):
        image, mask, objects = F.cutout(image, mask, objects, size, replace)
        return {'image':image, 'mask':mask, 'objects':objects}

    def get_params(self):
        return {
            'size':_parse_arg(self.size),
            'replace':self.replace
        }

class OpticalDistortion(Transform):
    def __init__(self, p=0.5, k=0.5, replace=0, border_mode='reflect'):
        self.p = p
        self.k = k
        self.replace = replace
        self.border_mode = border_mode

    def apply(self, image, mask, k, replace, border_mode):
        image, mask = F.optical_distortion(image, mask, k, replace, border_mode)
        return {'image':image, 'mask':mask}

    def get_params(self):
        return {
            'k':_parse_arg(self.k),
            'replace':self.replace,
            'border_mode':self.border_mode,
        }

class GridDistortion(Transform):
    def __init__(self, p=0.5, distort=0.5, num_steps=5, replace=0, border_mode='reflect'):
        self.p = p
        self.distort = distort
        self.num_steps = num_steps
        self.replace = replace
        self.border_mode = border_mode

    def apply(self, image, mask, distort, num_steps, replace, border_mode):
        image, mask = F.grid_distortion(image, mask, distort, num_steps, replace, border_mode)
        return {'image':image, 'mask':mask}

    def get_params(self):
        return {
            'distort':_parse_arg(self.distort),
            'num_steps':_parse_arg(self.num_steps),
            'replace':self.replace,
            'border_mode':self.border_mode,
        }

class ElasticTransform(Transform):
    def __init__(self, p=0.5, alpha=1, sigma=50, alpha_affine=50, replace=0, border_mode='reflect'):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.replace = replace
        self.border_mode = border_mode

    def apply(self, image, mask, alpha, sigma, alpha_affine, replace, border_mode):
        image, mask = F.elastic_transform(image, mask, alpha, sigma, alpha_affine, replace, border_mode)
        return {'image':image, 'mask':mask}

    def get_params(self):
        return {
                'alpha':_parse_arg(self.alpha),
                'alpha_affine':_parse_arg(self.alpha),
                'sigma':_parse_arg(self.alpha_affine),
                'replace':self.replace,
                'border_mode':self.border_mode,
            }

#composite ops
def Translate(p=0.5, level=(-0.3,0.3), replace=0):
  return Choice([TranslateY(p=1, level=level, replace=replace), TranslateX(p=1, level=level, replace=replace)], p=p, n=1)

def Shear(p=0.5, level=(-0.3,0.3), replace=0):
  return Choice([ShearY(p=1, level=level, replace=replace), ShearX(p=1, level=level, replace=replace)], p=p, n=1)
