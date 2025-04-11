import sys
import os
import json

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")
        sys.exit(1)

    path = sys.argv[1]
    json_file_path = os.path.join(path, 'input.json')

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            print(json.dumps(data, indent=4))
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        sys.exit(2)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {json_file_path}.")
        sys.exit(3)

    sys.exit(10)

if __name__ == "__main__":
    main()
