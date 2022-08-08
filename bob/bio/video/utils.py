import logging
import pickle

import h5py
import imageio
import numpy as np

from bob.bio.base import selected_indices
from bob.io.image import to_bob
from bob.pipelines import wrap

from .transformer import VideoWrapper

logger = logging.getLogger(__name__)


def video_wrap_skpipeline(sk_pipeline):
    """
    This function takes a `sklearn.Pipeline` and wraps each estimator inside of it with
    :any:`bob.bio.video.transformer.VideoWrapper`
    """

    for i, name, estimator in sk_pipeline._iter():

        # 1. Unwrap the estimator
        # If the estimator is `Sample` wrapped takes `estimator.estimator`.
        transformer = (
            estimator.estimator
            if hasattr(estimator, "estimator")
            else estimator
        )

        # 2. do a video wrap
        transformer = VideoWrapper(transformer)

        # 3. Sample wrap again
        transformer = wrap(
            ["sample"],
            transformer,
            fit_extra_arguments=estimator.fit_extra_arguments,
            transform_extra_arguments=estimator.transform_extra_arguments,
        )

        sk_pipeline.steps[i] = (name, transformer)

    return sk_pipeline


def select_frames(
    count, max_number_of_frames=None, selection_style=None, step_size=None
):
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
        indices = selected_indices(count, max_number_of_frames)
    elif selection_style == "step":
        indices = range(step_size // 2, count, step_size)[:max_number_of_frames]
    elif selection_style == "all":
        indices = range(0, count)
    else:
        raise ValueError(f"Invalid selection style: {selection_style}")

    return indices


def no_transform(x):
    return x


class VideoAsArray:
    """A memory efficient class to load only select video frames.
    It also supports efficient conversion to dask arrays.
    """

    def __init__(
        self,
        path,
        selection_style=None,
        max_number_of_frames=None,
        step_size=None,
        transform=None,
        **kwargs,
    ):
        """init

        Parameters
        ----------
        path : str
            Path to the video file
        selection_style : str, optional
            See :any:`select_frames`, by default None
        max_number_of_frames : int, optional
            See :any:`select_frames`, by default None
        step_size : int, optional
            See :any:`select_frames`, by default None
        transform : callable, optional
            A function that transforms the loaded video. This function should
            not change the video shape or its dtype. For example, you may flip
            the frames horizontally using this function, by default None
        """
        super().__init__(**kwargs)
        self.path = path
        self.reader = imageio.get_reader(path)
        self.dtype = np.uint8
        shape = (self.reader.count_frames(), 3) + self.reader.get_meta_data()[
            "size"
        ][::-1]
        self.ndim = len(shape)
        self.selection_style = selection_style

        indices = select_frames(
            count=self.reader.count_frames(),
            max_number_of_frames=max_number_of_frames,
            selection_style=selection_style,
            step_size=step_size,
        )

        self.indices = indices
        self.shape = (len(indices),) + shape[1:]
        self.transform = transform or no_transform

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("reader")
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.reader = imageio.get_reader(self.path)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        # logger.debug("Getting frame %s from %s", index, self.path)

        # In this method, someone is requesting indices thinking this video has
        # the shape of self.shape but self.shape is determined through
        # select_frames parameters. What we want to do here is to translate
        # ``index`` to real indices of the video file given that we want to load
        # only the selected frames. List of the selected frames are stored in
        # self.indices

        # If only one frame is requested, first translate the index to the real
        # frame number in the video file and load that

        if isinstance(index, int):
            idx = self.indices[index]
            return self.transform(
                np.asarray([to_bob(self.reader.get_data(idx))])
            )[0]

        if not (
            isinstance(index, tuple)
            and len(index) == self.ndim
            and all(isinstance(idx, slice) for idx in index)
        ):
            raise NotImplementedError(
                f"Indexing like {index} is not supported yet!"
            )

        # dask.array.from_array sometimes requests empty arrays
        if all(i == slice(0, 0) for i in index):
            return np.array([], dtype=self.dtype)

        def _frames_generator():
            # read the frames one by one and yield them
            real_frame_numbers = self.indices[index[0]]
            for i, frame in enumerate(self.reader):

                frame = to_bob(frame)
                if i not in real_frame_numbers:
                    continue
                # make sure arrays are loaded in C order because we reshape them
                # by C order later. Also, index into the frames here
                frame = np.ascontiguousarray(frame)[index[1:]]
                # return a tuple of flat array to match what is expected by
                # field_dtype
                yield (frame.ravel(),)
                if i == real_frame_numbers[-1]:
                    break

        iterable = _frames_generator()
        # compute the final shape given self.shape and index
        # see https://stackoverflow.com/a/36188683/1286165
        shape = [
            len(range(*idx.indices(dim))) for idx, dim in zip(index, self.shape)
        ]
        # field_dtype contains information about dtype and shape of each frame
        # numpy black magic: https://stackoverflow.com/a/12473478/1286165 allows
        # us to yield frame by frame in _frames_generator which greatly speeds
        # up loading the video
        field_dtype = [("", (self.dtype, (np.prod(shape[1:]),)))]
        total_number_of_frames = shape[0]
        video = np.fromiter(iterable, field_dtype, total_number_of_frames)
        # view the array as self.dtype to remove the field_dtype
        video = np.reshape(video.view(self.dtype), shape, order="C")

        return self.transform(video)

    def __repr__(self):
        return f"VideoAsArray: {self.path!r} {self.dtype!r} {self.ndim!r} {self.shape!r} {self.indices!r}"


class VideoLikeContainer:
    def __init__(self, data, indices, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.indices = indices

    def __repr__(self) -> str:
        return f"VideoLikeContainer: {self.data!r} {self.indices!r}"

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # we need to throw IndexErrors here because h5py throws ValueErrors
        # instead and this breaks loops on this class
        if isinstance(item, int) and item >= len(self):
            raise IndexError(f"Index ({item}) out of range (0-{len(self)-1})")
        return self.data[item]

    def __array__(self, dtype=None, *args, **kwargs):
        return np.asarray(self.data, dtype, *args, **kwargs)

    def __eq__(self, o: object) -> bool:
        return np.array_equal(self.data, o.data) and np.array_equal(
            self.indices, o.indices
        )

    def save(self, file):
        self.save_function(self, file)

    @staticmethod
    def save_function(other, file):
        try:
            with h5py.File(file, mode="w") as f:
                f["data"] = other.data
                f["indices"] = other.indices
        # revert to saving data in pickles when the dtype is not supported by hdf5
        except TypeError:
            with open(file, "wb") as f:
                pickle.dump({"data": other.data, "indices": other.indices}, f)

    @classmethod
    def load(cls, file):
        try:
            # weak closing of the hdf5 file so we don't load all the data into
            # memory https://docs.h5py.org/en/stable/high/file.html#closing-files
            f = h5py.File(file, mode="r")
            loaded = {"data": f["data"], "indices": list(f["indices"])}
        except OSError:
            with open(file, "rb") as f:
                loaded = pickle.load(f)
        self = cls(**loaded)
        return self
