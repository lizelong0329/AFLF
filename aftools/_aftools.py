r"""AlphaFoldTools - Interoperable Pretrain Structure Prediction with AlphaFold."""
# Authors: Zilin Song.


import os
import abc
import typing
import logging

import jax
import jax.numpy as jnp
import     numpy as  np

import aftools.utils as _u


class BasePreset(abc.ABC):
  r"""The abstract preset class."""

  @abc.abstractmethod
  def instantiate(self, **kwargs) -> object:
    r"""Instantiate the instance of the class configured by this preset."""
    raise NotImplementedError(r"Abstract method should be implemented by sub-classes.")


class BaseModel(abc.ABC):
  r"""The abstract Model class for AFTools."""

  def __init__(self, 
               config: _u.af.TAFConfig, 
               params: _u.af.TAFParams, 
               key: jax.random.KeyArray, 
               multimer_mode: bool, ):
    r"""Constructor for the abstract Model class for AFTools.
    
      Args:
        config (TAFConfig): The AF configurations, `aftools.utils.af.AFConfig()`.
        params (TAFParams): The AF parameters    , `aftools.utils.af.AFParams()`.
        key (jax.random.KeyArray): The JAX PRNG Key.
        multimer_mode (bool): If the model is for AF-Multimer.
    """
    _msg = 'AF-Multimer' if multimer_mode==True else 'AF' if multimer_mode==False else None
    assert not _msg is None, f"Illegal `multimer_mode=={multimer_mode}`."
    assert config.model.global_config.multimer_mode==multimer_mode, f"Illegal non-{_msg} config."
    self.config = config
    self.params = params
    self.key    = key

  @abc.abstractmethod
  def compile(self, **kwargs) -> None:
    r"""Compile the model for inference."""
    raise NotImplementedError(r"Abstract method should be implemented by sub-classes.")

  @abc.abstractmethod
  def predict(self, **kwargs) -> None:
    r"""Predict with the model."""
    raise NotImplementedError(r"Abstract method should be implemented by sub-classes.")


class BaseRunner(abc.ABC):
  r"""The abstract Runner class for AFTools."""

  def __init__(self, project_dir: str = os.getcwd(), runner_name: str = r'base_runner'):
    r"""Constructor of the abstract Runner class for AFTools.
    
      Args:
        project_dir (str): The project directory for the runner I/O.
        runner_name (str): The identity string for the Runner.
    """
    self._project_dir = os.path.abspath(project_dir)
    runner_name = str(runner_name)
    runtime_dir = _u.DIR_HIERARCHY.get(runner_name, None)
    assert not runtime_dir is None, f"Illegal `runner_name={runner_name}`."
    self._working_dir = os.path.join(self.project_dir, 'aftools_runtime', runtime_dir)
    self._runner_name = f"aftools.{runner_name}"
    self._key = jax.random.PRNGKey(seed=np.random.randint(low=-99999, high=100000))

  @property
  def project_dir(self) -> str: return self._project_dir

  @property
  def working_dir(self) -> str: return self._working_dir
  
  @property
  def runner_name(self) -> str: return self._runner_name
  
  @property
  def key(self) -> jax.random.KeyArray: return self._key
  @key.setter
  def key(self, val: jax.random.KeyArray) -> None: 
    assert isinstance(val, jnp.DeviceArray)
    assert val.shape==(2, )
    self._key = val

  @abc.abstractmethod
  def execute(self, **kwargs) -> typing.Union[None, typing.Any]:
    r"""Execute the Runner."""
    raise NotImplementedError(r"Abstract method should be implemented by sub-classes.")
  

class Runtime:
  r"""The runtime context manager for AFTools."""

  def __init__(self, whoami: str, working_dir: str):
    r"""Create a runtime context manager.
    
      Args:
        whoami      (str): The identity string for the Runner.
        working_dir (str): The directory for the Runner I/O.
    """
    assert isinstance(whoami,      str), f"Illegal `whoami` type: {type(whoami)}."
    assert isinstance(working_dir, str), f"Illegal `working_dir` type: {type(working_dir)}."
    self._whoami = whoami
    self._logger = logging.getLogger(name=self._whoami)
    self._logger.setLevel('INFO')
    self._working_dir = working_dir
    self._io = _u.io.IOManager()

  @property
  def whoami(self) -> str: return self._whoami

  @property
  def working_dir(self) -> str: return self._working_dir

  @property
  def logger(self) -> logging.Logger: return self._logger

  @property
  def io(self) -> _u.io.IOManager: return self._io
  
  def __enter__(self):
    self.mkdir(directory=self.working_dir)
    self.logger.info(f"Executing at working directory: {self.working_dir}.")
    self.logger.info( "Executing kernel ...")
    return self
  
  def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
    self.io.close()
    if not exc_type is None: # If there is an exception raised during Runtime.
      self.logger.info(f"Abnormal exit: Exception raised during Runtime !!!")
    else:
      self.logger.info(f"Normal exit: Done.")

  # FS operations.
  def mkdir(self, directory: str) -> None:
    r"""Create a directory in the file system."""
    if os.path.exists(directory):
      return
    os.makedirs(directory)    

  def todir(self, *files: str):
    r"""Get the directory by the `files`, absolute path string discards all previous directories."""
    return os.path.join(self.working_dir, *files)

