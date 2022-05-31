from functools import partial

from bob.bio.video.database import YoutubeDatabase
from bob.bio.video.utils import select_frames

# Defining frame selection bit
# If you want to customize this, please, create a new config file and do
# bob bio pipeline simple `my-new-config-file.py` `baseline`......
selection_style = "first"
max_number_of_frames = None
step_size = None


frame_selector = partial(
    select_frames,
    max_number_of_frames=max_number_of_frames,
    selection_style=selection_style,
    step_size=step_size,
)


database = YoutubeDatabase(protocol="fold0", frame_selector=frame_selector)
