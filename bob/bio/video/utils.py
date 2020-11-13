import logging

import bob.bio.base
import h5py
import numpy as np

logger = logging.getLogger(__name__)


def select_frames(count, max_number_of_frames=None, selection_style=None, step_size=None):
    """Returns indices of the frames to be selected given the parameters.

    Different selection styles are supported:

    * first : The first frames are selected
    * spread : Frames are selected to be taken from the whole video with equal spaces in
      between.
    * step : Frames are selected every ``step_size`` indices, starting at
      ``step_size/2`` **Think twice if you want to have that when giving FrameContainer
      data!**
    * all : All frames are selected unconditionally.

    Parameters
    ----------
    count : int
        Total number of frames that are available
    max_number_of_frames : int
        The maximum number of frames to be selected. Ignored when selection_style is
        "all".
    selection_style : str
        One of (``first``, ``spread``, ``step``, ``all``). See above.
    step_size : int
        Only useful when ``selection_style`` is ``step``.

    Returns
    -------
    range
        A range of frames to be selected.

    Raises
    ------
    ValueError
        If ``selection_style`` is not one of the supported ones.
    """
    # default values
    if max_number_of_frames is None:
        max_number_of_frames = 20
    if selection_style is None:
        selection_style = "spread"
    if step_size is None:
        step_size = 10

    if selection_style == "first":
        # get the first frames (limited by all frames)
        indices = range(0, min(count, max_number_of_frames))
    elif selection_style == "spread":
        # get frames lineraly spread over all frames
        indices = bob.bio.base.selected_indices(count, max_number_of_frames)
    elif selection_style == "step":
        indices = range(step_size // 2, count, step_size)[:max_number_of_frames]
    elif selection_style == "all":
        indices = range(0, count)
    else:
        raise ValueError(f"Invalid selection style: {selection_style}")

    return indices


class VideoAsArray:
    def __init__(
        self,
        path,
        selection_style=None,
        max_number_of_frames=None,
        step_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path = path
        self.reader = bob.io.video.reader(self.path)
        self.dtype, shape = self.reader.video_type[:2]
        self.ndim = len(shape)
        self.selection_style = selection_style
        indices = select_frames(
            count=self.reader.number_of_frames,
            max_number_of_frames=max_number_of_frames,
            selection_style=selection_style,
            step_size=step_size,
        )
        self.indices = indices
        self.shape = (len(indices),) + shape[1:]

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("reader")
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.reader = bob.io.video.reader(self.path)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        # logger.debug("Getting frame %s from %s", index, self.path)
        if isinstance(index, int):
            idx = self.indices[index]
            return self.reader[idx]

        if not (isinstance(index, tuple) and len(index) == self.ndim):
            raise NotImplementedError(f"Indxing like {index} is not supported yet!")

        if all(i == slice(0, 0) for i in index):
            return np.array([], dtype=self.dtype)

        if self.selection_style == "all":
            return np.asarray(self.reader.load())[index]

        idx = self.indices[index[0]]
        video = []
        for i, frame in enumerate(self.reader):
            if i not in idx:
                continue
            video.append(frame)
            if i == idx[-1]:
                break

        index = (slice(len(video)),) + index[1:]
        return np.asarray(video)[index]

    def __repr__(self):
        return f"{self.reader!r} {self.dtype!r} {self.ndim!r} {self.shape!r} {self.indices!r}"


class VideoLikeContainer:
    def __init__(self, data, indices, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __array__(self, dtype=None, *args, **kwargs):
        return np.asarray(self.data, dtype, *args, **kwargs)

    @classmethod
    def save(cls, other, file):
        with h5py.File(file, mode="w") as f:
            f["data"] = other.data
            f["indices"] = other.indices

    @classmethod
    def load(cls, file):
        with h5py.File(file, mode="r") as f:
            data = np.array(f["data"])
            indices = np.array(f["indices"])
        self = cls(data=data, indices=indices)
        return self
