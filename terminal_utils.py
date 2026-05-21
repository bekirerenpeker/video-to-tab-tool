import sys


def draw_progress_bar(percent, prefix="Progress", length=40):
    """
    Draws a progress bar in the terminal.
    percent: float between 0.0 and 1.0
    """
    filled_length = int(length * percent)
    bar = "#" * filled_length + "-" * (length - filled_length)

    # \r goes to the start of the line
    sys.stdout.write(f"\r{prefix}: |{bar}| {percent:.1%}")
    sys.stdout.flush()

    # Move to a new line once 100% is reached
    if percent >= 1.0:
        print()
