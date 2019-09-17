# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2019. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import logging

from ._schemas import InputParameters, OutputParameters
from allensdk.brain_observatory.ecephys.file_io.continuous_file import ContinuousFile
from allensdk.brain_observatory.argschema_utilities import ArgSchemaParserPlus
from .subsampling import select_channels, subsample_timestamps, subsample_lfp, remove_lfp_offset, remove_lfp_noise

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
    barrier = comm.Barrier
    gather = comm.gather
except ModuleNotFoundError as e:
    # Run without mpi4py installed
    MPI_rank = 0
    MPI_size = 1
    barrier = lambda : None
    gather = lambda data, root: data


logger = logging.getLogger(__name__)


def subsample(args):
    """

    :param args:
    :return:
    """
    params = args['lfp_subsampling']

    # distribute the processing of each node to a different core
    probes_by_rank = args['probes'][MPI_rank::MPI_size]
    probe_outputs_on_rank = []
    for probe in probes_by_rank:
        logging.info("Sub-sampling LFP for " + probe['name'])
        lfp_data_file = ContinuousFile(probe['lfp_input_file_path'], probe['lfp_timestamps_input_path'],
                                       probe['total_channels'])

        logging.info("loading lfp data...")
        lfp_raw, timestamps = lfp_data_file.load()
        if params['reorder_channels']:
            lfp_channel_order = lfp_data_file.get_lfp_channel_order()
        else:
            lfp_channel_order = np.arange(0, probe['total_channels'])

        logging.info("selecting channels...")
        channels_to_save, actual_channels = select_channels(probe['total_channels'],
                                                            probe['surface_channel'],
                                                            params['surface_padding'],
                                                            params['start_channel_offset'],
                                                            params['channel_stride'],
                                                            lfp_channel_order,
                                                            probe.get('noisy_channels', []),
                                                            params['remove_noisy_channels'],
                                                            probe['reference_channels'],
                                                            params['remove_reference_channels'])

        ts_subsampled = subsample_timestamps(timestamps, params['temporal_subsampling_factor'])

        logging.info("subsampling data...")
        lfp_subsampled = subsample_lfp(lfp_raw, channels_to_save, params['temporal_subsampling_factor'])

        del lfp_raw

        logging.info("removing offset...")
        lfp_filtered = remove_lfp_offset(lfp_subsampled,
                                         probe['lfp_sampling_rate'] / params['temporal_subsampling_factor'],
                                         params['cutoff_frequency'],
                                         params['filter_order'])

        del lfp_subsampled

        logging.info("Surface channel: " + str(probe['surface_channel']))

        logging.info("removing noise...")
        lfp = remove_lfp_noise(lfp_filtered, probe['surface_channel'], actual_channels)
        del lfp_filtered

        if params['remove_channels_out_of_brain']:
            channels_to_keep = actual_channels < (probe['surface_channel'] + 10)
            actual_channels = actual_channels[channels_to_keep]
            lfp = lfp[:, channels_to_keep]

        logging.info('Writing to disk...')
        lfp.tofile(probe['lfp_data_path'])
        np.save(probe['lfp_timestamps_path'], ts_subsampled)
        np.save(probe['lfp_channel_info_path'], actual_channels)

        probe_outputs_on_rank.append(
            {'name': probe['name'],
             'lfp_data_path': probe['lfp_data_path'],
             'lfp_timestamps_path': probe['lfp_timestamps_path'],
             'lfp_channel_info_path': probe['lfp_channel_info_path']
             }
        )

    barrier()
    # Send all the probe info to rank 0 where it will be saved.
    probe_outputs_all = gather(probe_outputs_on_rank, root=0)
    probe_outputs = []
    if MPI_rank == 0:
        for probes_on_rank in probe_outputs_all:
            for probe_data in probes_on_rank:
                probe_outputs.append(probe_data)

    else:
        probe_outputs = []

    return {'probe_outputs': probe_outputs}


def main():
    mod = ArgSchemaParserPlus(schema_type=InputParameters, output_schema_type=OutputParameters)
    output = subsample(mod.args)
    if MPI_rank == 0:
        # Only save the output file on rank 0 so the file isn't being overwritten
        output.update({"input_parameters": mod.args})
        if "output_json" in mod.args:
            mod.output(output, indent=2)
        else:
            logger.info(mod.get_output_json(output))
    barrier()


if __name__ == "__main__":
    main()
