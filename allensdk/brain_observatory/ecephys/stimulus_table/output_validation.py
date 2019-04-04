import numpy as np


def validate_epoch_durations(table, start_key='Start', end_key='End'):
    durations = table[end_key] - table[start_key]
    min_duration_index = durations.idxmin()
    min_duration = durations[min_duration_index]

    assert min_duration >= 0
    if min_duration == 0:
        warnings.warn(f'there is an epoch in this stimulus table (index: {min_duration_index}) with duration = {min_duration}', UserWarning)


def validate_epoch_order(table, time_keys=('Start', 'End')):
    for time_key in time_keys:
        change = np.diff(table[time_key].values)
        assert np.amin(change) > 0


def validate_max_spontaneous_epoch_duration(
    table, max_duration, get_spontanous_epochs=None, index_key='stimulus_index', start_key='Start', end_key='End'
):
    if get_spontanous_epochs is None:
        get_spontanous_epochs = lambda table: table[np.isnan(table[index_key])]

    spontaneous_epochs = get_spontanous_epochs(table)
    durations = spontaneous_epochs[end_key].values - spontaneous_epochs[start_key].values
    if  np.amax(durations) > max_duration:
        warnings.warn(f'there is a spontaneous activity duration longer than {max_duration}', UserWarning)