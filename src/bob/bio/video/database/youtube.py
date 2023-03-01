import logging
import os

from clapper.rc import UserDefaults

from bob.bio.base.database import CSVDatabase

logger = logging.getLogger(__name__)
rc = UserDefaults("bobrc.toml")


class YoutubeDatabase(CSVDatabase):

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

    name = "youtube"
    category = "video"
    dataset_protocols_name = "youtube.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/video/youtube-51c1fb2a.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/video/youtube-51c1fb2a.tar.gz",
    ]
    dataset_protocols_hash = "51c1fb2a"

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
        original_directory = original_directory or ""
        if not os.path.exists(original_directory):
            logger.warning(
                f"Invalid or non existent `original_directory`: {original_directory}."
                "Please, do `bob config set bob.bio.video.youtube.directory PATH` to set the Youtube data directory."
            )

        self.references_dict = {}
        self.probes_dict = {}

        # Dict that holds a `subject_id` as a key and has
        # filenames as values
        self.subject_id_files = {}
        self.template_id_to_subject_id = None
        self.template_id_to_sample = None
        self.original_directory = original_directory
        self.extension = extension
        self.annotation_extension = annotation_extension
        self.frame_selector = frame_selector

        super().__init__(
            name=self.name,
            protocol=protocol,
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
            memory_demanding=True,
        )
