import signal
import sys
from train import train

if __name__ == "__main__":
    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    train()
