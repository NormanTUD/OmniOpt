import time
import subprocess
from collections import deque
from rich.console import Console
from rich.live import Live

def read_new_lines_from_command(process, lines_buffer, max_lines):
    """
    Reads new lines from the given subprocess process and updates the lines buffer.
    
    Args:
        process: The subprocess process to read from.
        lines_buffer: The buffer (deque) to store the last lines.
        max_lines: The maximum number of lines to keep in the buffer.
    """
    new_lines = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        new_lines.append(line.decode('utf-8'))
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

def tail_command(command, max_lines=20, refresh_interval=1):
    """
    Continuously runs a command and displays the last 'max_lines' lines of its output using rich.
    
    Args:
        command: The command to run and tail.
        max_lines: The maximum number of lines to display.
        refresh_interval: The interval (in seconds) to refresh the display.
    """
    console = Console()
    lines_buffer = deque(maxlen=max_lines)

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    try:
        with Live(console=console, refresh_per_second=4):
            while True:
                read_new_lines_from_command(process, lines_buffer, max_lines)
                display_lines(console, lines_buffer)
                if process.poll() is not None:
                    read_new_lines_from_command(process, lines_buffer, max_lines)
                    display_lines(console, lines_buffer)
                    break
                time.sleep(refresh_interval)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
    finally:
        process.stdout.close()
        process.stderr.close()
        process.terminate()

# Example usage:
# Replace 'your_command' with the bash command you want to tail.
tail_command("tail -f stats/usage_statistics.csv")
