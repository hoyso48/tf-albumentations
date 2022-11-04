import tensorflow as tf

from tf_albumentations.transforms import *

_MAX_LEVEL = 10.

class config_autoaug:
    cutout_const=100
    translate_const=1#250
    replace=[128,128,128]

class config_randaug:
    cutout_const=40
    translate_const=0.4#100
    replace=[128,128,128]

def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor

def _rotate_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 30.
  # level = _randomly_negate_tensor(level)
  return level

def _shrink_level_to_arg(level):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return level


def _enhance_level_to_arg(level):
  return (level/_MAX_LEVEL) * 1.8 + 0.1


def _shear_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  # level = _randomly_negate_tensor(level)
  return level


def _translate_level_to_arg(level, translate_const):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  # level = _randomly_negate_tensor(level)
  return level


def level_to_arg(params):
  return {
      'AutoContrast': lambda level: {},
      'Equalize': lambda level: {},
      'Invert': lambda level: {},
      'Rotate': lambda level: {'degrees': [-_rotate_level_to_arg(level), _rotate_level_to_arg(level)], 'replace':params.replace},
      'Posterize': lambda level: {'bits': int((level/_MAX_LEVEL) * 4)},
      'Solarize': lambda level: {'threshold': int((level/_MAX_LEVEL) * 256)},
      'SolarizeAdd': lambda level: {'threshold': int((level/_MAX_LEVEL) * 110)},
      'Color': lambda level: {'factor':_enhance_level_to_arg(level)},
      'Contrast': lambda level: {'factor':_enhance_level_to_arg(level)},
      'Brightness': lambda level: {'factor':_enhance_level_to_arg(level)},
      'Sharpness': lambda level: {'factor':_enhance_level_to_arg(level)},
      'ShearX': lambda level: {'level':[-_shear_level_to_arg(level), _shear_level_to_arg(level)], 'replace':params.replace},
      'ShearY': lambda level: {'level':[-_shear_level_to_arg(level), _shear_level_to_arg(level)], 'replace':params.replace},
      'Cutout': lambda level: {'size' :int((level/_MAX_LEVEL) * params.cutout_const), 'replace':params.replace},
      'TranslateX': lambda level: {'level':[-_translate_level_to_arg(level, params.translate_const), _translate_level_to_arg(level, params.translate_const)], 'replace':params.replace},
      'TranslateY': lambda level: {'level':[-_translate_level_to_arg(level, params.translate_const), _translate_level_to_arg(level, params.translate_const)], 'replace':params.replace}
  }

def policy_v0():
  """Autoaugment policy that was used in AutoAugment Paper."""
  # Each tuple is an augmentation operation of the form
  # (operation, probability, magnitude). Each element in policy is a
  # sub-policy that will be applied sequentially on the image.
  policy = [
      [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
      [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
      [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
      [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
      [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
      [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
      [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
      [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
      [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
      [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
      [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
      [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
      [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
      [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
      [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
      [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
      [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
      [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
      [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
      [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
      [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
      [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
      [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
      [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
      [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
  ]
  return policy

def AutoAugment(policies = policy_v0(), config = config_autoaug):
  POLICY = []
  for policy in policies:
    SUB_POLICY = []
    for subpolicy, prob, level in policy:
      args = level_to_arg(config)[subpolicy](level=level)
      op = eval(subpolicy)(p=prob, **args)
      SUB_POLICY.append(op)
    POLICY.append(Sequence(SUB_POLICY, p=1))
  return Sequence([HorizontalFlip(p=0.5), Choice(POLICY, p=1, n=1)])

available_ops = [
    'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
    'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd']

def RandAugment(available_ops=available_ops, magnitude=10, n_layers=2, config=config_randaug):
  """Applies the RandAugment policy to `image`.
  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  Args:
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].
  """
  POLICY = []
  for subpolicy in available_ops:
    args = level_to_arg(config)[subpolicy](level=magnitude)
    op = eval(subpolicy)(p=1, **args)
    POLICY.append(op)
  return Sequence([HorizontalFlip(p=0.5), Choice(POLICY, p=1, n=n_layers, with_replacement=True)])
