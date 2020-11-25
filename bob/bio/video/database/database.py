from bob.bio.base.database.file import BioFile

from ..utils import VideoAsArray


class VideoBioFile(BioFile):
    def __init__(
        self,
        client_id,
        path,
        file_id,
        original_directory=None,
        original_extension=".avi",
        annotation_directory=None,
        annotation_extension=None,
        annotation_type=None,
        selection_style=None,
        max_number_of_frames=None,
        step_size=None,
        **kwargs,
    ):
        """
        Initializes this File object with an File equivalent for
        VoxForge database.
        """
        super().__init__(
            client_id=client_id,
            path=path,
            file_id=file_id,
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_directory=annotation_directory,
            annotation_extension=annotation_extension,
            annotation_type=annotation_type,
            **kwargs,
        )
        self.selection_style = selection_style or "all"
        self.max_number_of_frames = max_number_of_frames
        self.step_size = step_size

    def load(self):
        path = self.make_path(self.original_directory, self.original_extension)
        return VideoAsArray(
            path,
            selection_style=self.selection_style,
            max_number_of_frames=self.max_number_of_frames,
            step_size=self.step_size,
        )
