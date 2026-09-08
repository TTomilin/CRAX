# Copyright 2024 The Brax Authors.
# Modifications Copyright 2026 CRAX Authors.
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
#
# This file is derived from Brax 0.12.3 (__init__.py) and has been modified
# by the CRAX Authors: carries CRAX's own version, and records the upstream Brax
# release in `__brax_version__`.

"""Import top-level classes and functions here for encapsulation/clarity."""

__version__ = '0.1.0'

# The Brax release this fork is derived from; see PROVENANCE.md.
__brax_version__ = '0.12.3'

from crax.base import Motion
from crax.base import State
from crax.base import System
from crax.base import Transform
