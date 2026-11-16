from pathlib import Path
import sys
sys.path.insert(0, str(Path.home() / "runtime-lab"))
from app import app as application
