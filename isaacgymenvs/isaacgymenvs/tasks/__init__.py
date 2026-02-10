# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .anymal import Anymal
from .go2w import Go2w
from .go2w_aerialcrossover import Go2wAerialcrossover
from .go2w_frontflip import Go2wFrontflip
from .go2w_sideflip import Go2wSideflip
from .go2w_walk import Go2wWalk
from .go2 import Go2
from .go2_bound import Go2Bound
from .anymal_terrain import AnymalTerrain


# Mappings from strings to environments
isaacgym_task_map = {
    "Anymal": Anymal,
    "Go2w" : Go2w,
    "Go2wAerialcrossover" : Go2wAerialcrossover,
    "Go2wFrontflip" : Go2wFrontflip,
    "Go2wSideflip" : Go2wSideflip,
    "Go2wWalk" : Go2wWalk,
    "Go2" : Go2,
    "Go2Bound" : Go2Bound,
    "AnymalTerrain": AnymalTerrain,
}

