import time
from collections import deque
from rich.console import Console
from rich.live import Live

def read_new_lines(file_handle, lines_buffer, max_lines):
    """
    Reads new lines from the given file handle and updates the lines buffer.
    
    Args:
        file_handle: The file handle to read from.
        lines_buffer: The buffer (deque) to store the last lines.
        max_lines: The maximum number of lines to keep in the buffer.
    """
    new_lines = file_handle.readlines()
    if new_lines:
        lines_buffer.extend(new_lines)
        while len(lines_buffer) > max_lines:
            lines_buffer.popleft()

def display_lines(console, lines):
    """
    Displays the lines using rich console.
    
    Args:
        console: The rich console object.
        lines: A list of lines to be displayed.
    """
    console.clear()
    for line in lines:
        console.print(line.strip())

def tail_f(file_path, max_lines=20, refresh_interval=1):
    """
    Continuously reads a file and displays the last 'max_lines' lines using rich.
    
    Args:
        file_path: The path to the file to be tailed.
        max_lines: The maximum number of lines to display.
        refresh_interval: The interval (in seconds) to refresh the display.
    """
    console = Console()
    lines_buffer = deque(maxlen=max_lines)

    try:
        with open(file_path, "r") as file_handle:
            file_handle.seek(0, 2)  # Move to the end of the file
            with Live(console=console, refresh_per_second=4):
                while True:
                    current_position = file_handle.tell()
                    read_new_lines(file_handle, lines_buffer, max_lines)
                    if file_handle.tell() != current_position:
                        display_lines(console, lines_buffer)
                    time.sleep(refresh_interval)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")

# Example usage:
# Replace 'stats/usage_statistics.csv' with the path to the log file you want to tail.
tail_f("stats/usage_statistics.csv")

