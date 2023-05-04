# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out

#Analyse GPU memory usage
def printArrayVars(scope): #used to analyse the array sizes in the GPU
    #for debugging purposes
    vardict = {}
    for var in list(scope):
        if "ndarray" in str(type(eval(var, scope))):
            vardict[var] = str(type(eval(var,scope))), eval(var,scope).shape, eval(var,scope).nbytes, eval(var,scope).dtype.name

    cache = {}
    sta = []
    for name in sorted(vardict.keys()):
        if "ndarray" in vardict[name][0]:
            var, shape, nbytes, dtype = vardict[name]
            idv = id(var)
            if idv in cache.keys():
                namestr = '{} ({})'.format(name, cache[idv])
                original = 0
            else:
                cache[idv] = name
                namestr = name
                original = 1
            shapestr = ' x '.join(map(str, shape))
            bytestr = str(nbytes)
            dtype = var.split(' \'')[1].split('.')[0] + " " +  dtype
            sta.append(
                [namestr, shapestr, bytestr, dtype, original]
            )


    maxname = 0
    maxshape = 0
    maxbyte = 0
    totalbytes = 0
    for k in range(len(sta)):
        val = sta[k]
        if maxname < len(val[0]):
            maxname = len(val[0])
        if maxshape < len(val[1]):
            maxshape = len(val[1])
        if maxbyte < len(val[2]):
            maxbyte = len(val[2])
        if val[4]:
            totalbytes += int(val[2])

    if len(sta) > 0:
        sp1 = max(10, maxname)
        sp2 = max(10, maxshape)
        sp3 = max(10, maxbyte)
        prval = 'Name {} Shape {} Bytes {} Type'.format(
            sp1 * ' ', sp2 * ' ', sp3 * ' '
        )
        print("{}\n{}\n".format(prval, "=" * (len(prval) + 5)))

    for k in range(len(sta)):
        val = sta[k]
        print(
            '{} {} {} {} {} {} {}'.format(
                val[0],
                ' ' * (sp1 - len(val[0]) + 4),
                val[1],
                ' ' * (sp2 - len(val[1]) + 5),
                val[2],
                ' ' * (sp3 - len(val[2]) + 5),
                val[3],
            )
        )
    print('\nUpper bound on total bytes  =       {}'.format(totalbytes))

