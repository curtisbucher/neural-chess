# For reletive imports
import sys
sys.path.append('src/engine') #TODO: relative imports

from src.engine import start

if __name__ == "__main__":
    try:
        start()
    except KeyboardInterrupt:
        pass
