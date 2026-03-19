r"""Test cases for `aftools.utils.serialization.py`."""
# Authors: Zilin Song


import typing
import json
import unittest

import jax.numpy as jnp
import     numpy as  np

from aftools.utils.serialization import JSONable


# helpers.
def gen_arr() -> np.ndarray: return np.random.random_sample((100, 20)).astype(np.float32)


class InnerNamedTuple(typing.NamedTuple):
  x = 'x'
  y = {"in_namedtuple": list((1, 2, 3)), }
  z = gen_arr()


class Inner(JSONable):
  _json_attr_ = ['jnp_arr', 'np_arr', 
                 's', 'i', 'b', 'f', 'none', 
                 'dct', 'lst', 'tpl', 'st', 'namedtpl', ]
  def __init__(self):
    self.s, self.i, self.b, self.f, self.none = "{txt}", 7, False, 3.14, None
    self.dct, self.lst, self.tpl, self.st = {"in": 1}, [2, 3], (4, 5), {6, 7}
    self.namedtpl = InnerNamedTuple()
    self. np_arr =  np.asarray(gen_arr())
    self.jnp_arr = jnp.asarray(gen_arr())


class OuterNamedTuple(typing.NamedTuple):
  inner_named_tuple = InnerNamedTuple()
  inner_obj = Inner()


class Outer(JSONable):
  _json_attr_ = ['jnp_arr', 'np_arr', 
                 's', 'i', 'b', 'f', 'none', 
                 'dct', 'lst', 'tpl', 'st', 'namedtpl',
                 'inner0', 'inner1', ]
  def __init__(self):
    self.s, self.i, self.b, self.f, self.none = "{out}", 9, False, 2.71, None
    self.dct, self.lst, self.tpl, self.st = {"out": 8}, [9, 10], (11, 12), {13, 14}
    self. np_arr =  np.asarray(gen_arr())
    self.jnp_arr = jnp.asarray(gen_arr())
    self.namedtpl = OuterNamedTuple()
    self.inner0, self.inner1 = Inner(), Inner()

def gen_jsonables() -> tuple[type[Inner], type[Outer]]: return Inner, Outer


class Test_JSONable(unittest.TestCase):

  def setUp(self):
    self.Inner, self.Outer = gen_jsonables()
  
  def tearDown(self):
    del self.Inner
    del self.Outer

  def test_primitives(self):
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    for attr in ['s', 'i', 'b', 'f', 'none']:
      self.assertEqual(getattr(oo, attr), getattr(oo2, attr))
  
  def test_np_array(self):
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    np.testing.assert_array_equal(oo.np_arr, oo2.np_arr)
  
  def test_jnp_array(self):
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    self.assertTrue(jnp.array_equal(oo.jnp_arr, oo2.jnp_arr))

  def test_jsonable_nested(self):
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    self.assertIsInstance(oo2.inner0, self.Inner)
    self.assertIsInstance(oo2.inner1, self.Inner)
    self.assertIsInstance(oo2.       namedtpl, OuterNamedTuple)
    self.assertIsInstance(oo2.inner0.namedtpl, InnerNamedTuple)
    self.assertIsInstance(oo2.inner1.namedtpl, InnerNamedTuple)
    # primitives.
    for attr in ['s', 'i', 'b', 'f', 'none']:
      self.assertEqual(getattr(oo.inner0, attr), getattr(oo2.inner0, attr))
      self.assertEqual(getattr(oo.inner1, attr), getattr(oo2.inner1, attr))
    # containers.
    for attr in ['dct', 'lst', 'tpl', 'st', 'namedtpl']:
      self.assertEqual(getattr(oo.inner0, attr), getattr(oo2.inner0, attr))
      self.assertEqual(getattr(oo.inner1, attr), getattr(oo2.inner1, attr))
    # np.ndarray.
    np.testing.assert_array_equal(oo.inner0.np_arr, oo2.inner0.np_arr)
    np.testing.assert_array_equal(oo.inner1.np_arr, oo2.inner1.np_arr)
    # jnp.ndarray.
    np.testing.assert_array_equal(oo.inner0.jnp_arr, oo2.inner0.jnp_arr)
    np.testing.assert_array_equal(oo.inner1.jnp_arr, oo2.inner1.jnp_arr)

  def test_containers(self):
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    self.assertIsInstance(oo2.       namedtpl, OuterNamedTuple)
    self.assertIsInstance(oo2.inner0.namedtpl, InnerNamedTuple)
    self.assertIsInstance(oo2.inner1.namedtpl, InnerNamedTuple)
    for attr in ['dct', 'lst', 'tpl', 'st', 'namedtpl', ]:
      self.assertEqual(getattr(oo, attr), getattr(oo2, attr))
  
  def test_full_roundtrip(self):
    oo         = self.Outer()
    oo2: Outer = Outer.from_json(oo.to_json())
    self.assertIsInstance(oo2, self.Outer)
  
  def test_valid_json_string(self):
    oo = self.Outer()
    json_str = oo.to_json()
    parsed = json.loads(json_str) # should not raise.
    self.assertIsInstance(parsed, dict)