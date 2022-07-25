import copy
import logging
import os

from functools import partial

import bob.io.base

from bob.bio.base.pipelines.abstract_classes import Database
from bob.bio.video.utils import VideoLikeContainer, select_frames
from bob.extension import rc
from bob.extension.download import get_file
from bob.pipelines import DelayedSample, SampleSet

logger = logging.getLogger(__name__)


class YoutubeDatabase(Database):
    """
    This package contains the access API and descriptions for the `YouTube Faces` database.
    It only contains the Bob accessor methods to use the DB directly from python, with our certified protocols.
    The actual raw data for the `YouTube Faces` database should be downloaded from the original URL (though we were not able to contact the corresponding Professor).

    .. warning::

      To use this dataset protocol, you need to have the original files of the YOUTUBE datasets.
      Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.bio.face.youtube.directory [YOUTUBE PATH]



    In this interface we implement the 10 original protocols of the `YouTube Faces` database ('fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10')


    The code below allows you to fetch the gallery and probes of the "fold0" protocol.

    .. code-block:: python

        >>> from bob.bio.video.database import YoutubeDatabase
        >>> youtube = YoutubeDatabase(protocol="fold0")
        >>>
        >>> # Fetching the gallery
        >>> references = youtube.references()
        >>> # Fetching the probes
        >>> probes = youtube.probes()


    Parameters
    ----------

        protocol: str
           One of the Youtube above mentioned protocols

        annotation_type: str
           One of the supported annotation types

        original_directory: str
           Original directory

        extension: str
           Default file extension

        annotation_extension: str

        frame_selector:
           Pointer to a function that does frame selection.

    """

    def __init__(
        self,
        protocol,
        annotation_type="bounding-box",
        fixed_positions=None,
        original_directory=rc.get("bob.bio.video.youtube.directory", ""),
        extension=".jpg",
        annotation_extension=".labeled_faces.txt",
        frame_selector=None,
    ):

        self._check_protocol(protocol)

        original_directory = original_directory or ""
        if not os.path.exists(original_directory):
            logger.warning(
                "Invalid or non existent `original_directory`: f{original_directory}."
                "Please, do `bob config set bob.bio.video.youtube.directory PATH` to set the Youtube data directory."
            )

        urls = YoutubeDatabase.urls()
        cache_subdir = os.path.join("datasets", "youtube_protocols")
        self.filename = get_file(
            "youtube_protocols-6962cd2e.tar.gz",
            urls,
            file_hash="8a4792872ff30b37eab7f25790b0b10d",
            extract=True,
            cache_subdir=cache_subdir,
        )
        self.protocol_path = os.path.dirname(self.filename)

        self.references_dict = {}
        self.probes_dict = {}

        # Dict that holds a `subject_id` as a key and has
        # filenames as values
        self.subject_id_files = {}
        self.reference_id_to_subject_id = None
        self.reference_id_to_sample = None
        self.load_file_client_id()
        self.original_directory = original_directory
        self.extension = extension
        self.annotation_extension = annotation_extension
        self.frame_selector = frame_selector

        super().__init__(
            name="youtube",
            protocol=protocol,
            score_all_vs_all=False,
            annotation_type=annotation_type,
            fixed_positions=None,
            memory_demanding=True,
        )

    def load_file_client_id(self):

        self.subject_id_files = {}

        # List containing the client ID
        # Each element of this file matches a line in Youtube_names.txt
        self.reference_id_to_subject_id = bob.io.base.load(
            os.path.join(self.protocol_path, "Youtube_labels.mat.hdf5")
        )[0].astype("int")
        self.reference_id_to_sample = [
            x.rstrip("\n")
            for x in open(
                os.path.join(self.protocol_path, "Youtube_names.txt")
            ).readlines()
        ]

        for ll, n in zip(
            self.reference_id_to_subject_id, self.reference_id_to_sample
        ):
            key = int(ll)
            if key not in self.subject_id_files:
                self.subject_id_files[key] = []

            self.subject_id_files[key].append(n.rstrip("\n"))

    def _load_pairs(self):
        fold = int(self.protocol[-1])

        split = bob.io.base.load(
            os.path.join(self.protocol_path, "Youtube_splits.mat.hdf5")
        )[:, :, fold].astype(int)

        return split[:, 0], split[:, 1]

    def _load_video_from_path(self, path):
        files = sorted(
            [x for x in os.listdir(path) if os.path.splitext(x)[1] == ".jpg"]
        )

        # If there's no frame selector, uses all frames
        files_indices = (
            select_frames(
                len(files),
                max_number_of_frames=None,
                selection_style="all",
                step_size=None,
            )
            if self.frame_selector is None
            else self.frame_selector(len(files))
        )

        data, indices = [], []
        for i, file_name in enumerate(files):
            if i not in files_indices:
                continue
            file_name = os.path.join(path, file_name)
            indices.append(os.path.basename(file_name))
            data.append(bob.io.base.load(file_name))

        return VideoLikeContainer(data=data, indices=indices)

    def _make_sample_set(
        self, reference_id, subject_id, sample_path, references=None
    ):

        path = os.path.join(self.original_directory, sample_path)

        kwargs = {} if references is None else {"references": references}

        # Delaying the annotation loading
        delayed_annotations = partial(self._annotations, path)
        return SampleSet(
            key=str(reference_id),
            reference_id=str(reference_id),
            subject_id=str(subject_id),
            **kwargs,
            samples=[
                DelayedSample(
                    key=str(sample_path),
                    load=partial(self._load_video_from_path, path),
                    delayed_attributes={"annotations": delayed_annotations},
                )
            ],
        )

    def _annotations(self, path):
        """Returns the annotations for the given file id as a dictionary of dictionaries, e.g. {'1.56.jpg' : {'topleft':(y,x), 'bottomright':(y,x)}, '1.57.jpg' : {'topleft':(y,x), 'bottomright':(y,x)}, ...}.
        Here, the key of the dictionary is the full image file name of the original image.

        Parameters
        ----------

        path: str
            The path containing the frame sequence of a user

        """

        if self.original_directory is None:
            raise ValueError(
                "Please specify the 'original_directory' in the constructor of this class to get the annotations."
            )

        directory = os.path.dirname(path)
        # shot_id = os.path.basename(path)

        annotation_file = os.path.join(directory + self.annotation_extension)

        annots = {}

        with open(annotation_file) as f:
            for line in f:
                splits = line.rstrip().split(",")
                # shot_id = int(splits[0].split("\\")[1])
                index = splits[0].split("\\")[2]

                # coordinates are: center x, center y, width, height
                (center_y, center_x, d_y, d_x) = (
                    float(splits[3]),
                    float(splits[2]),
                    float(splits[5]) / 2.0,
                    float(splits[4]) / 2.0,
                )
                # extract the bounding box information
                annots[index] = {
                    "topleft": (center_y - d_y, center_x - d_x),
                    "bottomright": (center_y + d_y, center_x + d_x),
                }

        # return the annotations as returned by the call function of the
        # Annotation object
        return annots

    def background_model_samples(self):
        """ """
        return None

    def references(self, group="dev"):
        self._check_group(group)
        if self.protocol not in self.references_dict:
            self.references_dict[self.protocol] = []
            pairs = self._load_pairs()

            for i, (e, _) in enumerate(zip(pairs[0], pairs[1])):
                reference_id = e
                subject_id = self.reference_id_to_subject_id[reference_id]
                sample_path = self.reference_id_to_sample[reference_id]
                sampleset = self._make_sample_set(
                    reference_id, subject_id, sample_path
                )
                self.references_dict[self.protocol].append(sampleset)

        return self.references_dict[self.protocol]

    def probes(self, group="dev"):
        self._check_group(group)
        if self.protocol not in self.probes_dict:
            self.probes_dict[self.protocol] = []
            pairs = self._load_pairs()

            # Computing reference list
            probe_to_reference_id_dict = dict()
            for e, p in zip(pairs[0], pairs[1]):
                if p not in probe_to_reference_id_dict:
                    probe_to_reference_id_dict[p] = []
                probe_to_reference_id_dict[p].append(str(e))

            # Now assembling the samplesets
            for _, p in zip(pairs[0], pairs[1]):
                reference_id = p
                subject_id = self.reference_id_to_subject_id[reference_id]
                sample_path = self.reference_id_to_sample[reference_id]
                references = copy.deepcopy(probe_to_reference_id_dict[p])
                sampleset = self._make_sample_set(
                    reference_id, subject_id, sample_path, references
                )
                self.probes_dict[self.protocol].append(sampleset)

        return self.probes_dict[self.protocol]

    def all_samples(self):
        return self.references() + self.probes()

    def groups(self):
        return ["dev"]

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/youtube_protocols-6962cd2e.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/youtube_protocols-6962cd2e.tar.gz",
        ]

    @staticmethod
    def protocols():
        return [f"fold{fold}" for fold in range(10)]

    def _check_protocol(self, protocol):
        assert (
            protocol in self.protocols()
        ), "Invalid protocol `{}` not in {}".format(protocol, self.protocols())

    def _check_group(self, group):
        assert group in self.groups(), "Invalid group `{}` not in {}".format(
            group, self.groups()
        )
