# Copyright 2025 PhaseLock Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PhaseLock + GeoPhys: trajectory geometry in video diffusion internals.

Two things live here:

* **PhaseLock** -- the training-free Latent Delta Guidance method, now backend-agnostic
  so it runs on Wan2.1 as well as CogVideoX.
* **GeoPhys on internal representations** -- the five geometric statistics of a per-frame
  feature trajectory, applied not to a frozen external encoder but to a video diffusion
  model's own internals: DiT hidden states, VAE latents, clean-latent estimates and the
  flow velocity field. Plus a new family of metrics coupling the two notions of
  "velocity" that GeoPhys and flow matching each use.
"""

from .config import Config, load
from .guidance import LatentDeltaGuidance, extract_motion_prior
from .utils import resolve_dtype, set_seed

__version__ = "0.2.0"

__all__ = [
    "Config",
    "LatentDeltaGuidance",
    "extract_motion_prior",
    "load",
    "resolve_dtype",
    "set_seed",
]
