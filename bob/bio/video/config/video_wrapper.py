from bob.bio.video.transformer import VideoWrapper
from bob.bio.video.utils import video_wrap_skpipeline
from bob.pipelines import wrap

# Fetaching the pipeline from the chain-loading
pipeline = locals().get("pipeline")


pipeline.transformer = video_wrap_skpipeline(pipeline.transformer)

