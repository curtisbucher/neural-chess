# For reletive imports
import sys
sys.path.append('src/engine')

from src.engine.ui import start

if __name__ == "__main__":
    try:
        start()
    except KeyboardInterrupt:
        pass
