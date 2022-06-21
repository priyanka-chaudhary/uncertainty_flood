from dotenv import load_dotenv
from pathlib import Path
import os

rain_const = 150.4
waterdepth_diff_const = 0.01
max_depth_709_const = 10.65

env_path = Path(__file__).parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path.resolve(), override=True, verbose=True)
    
class LazyEnv:
    """Lazy environment variable."""

    def __init__(
        self,
        env_var,
        default=None,
        return_type=str,
        after_eval=None,
    ):
        """Construct lazy evaluated environment variable."""
        self.env_var = env_var
        self.default = default
        self.return_type = return_type
        self.after_eval = after_eval

    def eval(self):
        """Evaluate environment variable."""
        value = self.return_type(os.environ.get(self.env_var, self.default))

        if self.after_eval:
            self.after_eval(value)

        return value
    
PATH_ROOT = str(Path(__file__).parents[1])
PATH_DATA = LazyEnv("PATH_DATA", Path(PATH_ROOT) / Path("data")).eval()
PATH_TOY = LazyEnv("PATH_TOY", Path(PATH_DATA) / Path("toy")).eval()
PATH_709 = LazyEnv("PATH_709", Path(PATH_DATA) / Path("709")).eval()
PATH_709_new = LazyEnv("PATH_709_new", Path(PATH_DATA) / Path("709_new")).eval()
PATH_684 = LazyEnv("PATH_684", Path(PATH_DATA) / Path("684")).eval()
PATH_709wall = LazyEnv("PATH_709wall", Path(PATH_DATA) / Path("709wall")).eval()
PATH_709wall_0threshold = LazyEnv("PATH_709wall_0threshold", Path(PATH_DATA) / Path("709wall_0threshold")).eval()
PATH_709_0threshold = LazyEnv("PATH_709_0threshold", Path(PATH_DATA) / Path("709_0threshold")).eval()

PATH_GENERATED = LazyEnv("PATH_GENERATED", Path(PATH_DATA) / Path("generated_datasets")).eval()
PATH_SUMMARY = LazyEnv("PATH_SUMMARY", Path(PATH_DATA) / Path("summary")).eval()
PATH_CHECKPOINTS = LazyEnv("PATH_CHECKPOINTS", Path(PATH_DATA) / Path("checkpoints")).eval()

Path(PATH_GENERATED).mkdir(parents=True, exist_ok=True)
Path(PATH_SUMMARY).mkdir(parents=True, exist_ok=True)
Path(PATH_CHECKPOINTS).mkdir(parents=True, exist_ok=True)

