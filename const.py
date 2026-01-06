
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
MESH_DIR = DATA_DIR / "mesh"
OUTPUT_DIR = DATA_DIR / "output"
LOG_DIR = DATA_DIR / "logs"
VISUALIZATION_DIR = DATA_DIR / "visualization"