from bob.bio.video.database import YoutubeBioDatabase


database = YoutubeBioDatabase(
    protocol="fold1",
    models_depend_on_protocol=True,
    training_depends_on_protocol=True,
    all_files_options={"subworld": "fivefolds"},
    extractor_training_options={"subworld": "fivefolds"},
    projector_training_options={"subworld": "fivefolds"},
    enroller_training_options={"subworld": "fivefolds"},
)
