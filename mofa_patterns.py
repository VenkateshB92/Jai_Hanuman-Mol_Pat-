import os
import json
from collections import defaultdict

# Directory where the JSON files are stored
json_dir = 'graph_json'

def load_json_files(json_dir):
    """Load all JSON files from the specified directory."""
    json_data = []
    for file in os.listdir(json_dir):
        if file.endswith('.json'):
            with open(os.path.join(json_dir, file), 'r') as f:
                data = json.load(f)
                json_data.append((file, data))  # Store filename along with data
    return json_data

def extract_atom_patterns(json_data, pattern_size):
    """Extract atom patterns of a specified size from the loaded JSON data."""
    pattern_occurrences = defaultdict(lambda: defaultdict(int))  # {pattern: {file: count}}

    for filename, graph in json_data:
        if 'node_labels' in graph:
            atoms = list(graph['node_labels'].values())
            # Generate patterns of the specified size
            for i in range(len(atoms) - pattern_size + 1):
                pattern = tuple(atoms[i:i + pattern_size])  # Get a sub-list of the specified size
                pattern_occurrences[pattern][filename] += 1  # Count each pattern for the specific file

    return pattern_occurrences

def find_common_patterns(pattern_occurrences, min_count=2):
    """Filter patterns that appear in two or more files."""
    common_patterns = {}
    for pattern, occurrences in pattern_occurrences.items():
        if len(occurrences) >= min_count:  # Check if the pattern appears in 2 or more files
            common_patterns[pattern] = occurrences
    return common_patterns

def main():
    # Load JSON files
    json_data = load_json_files(json_dir)

    # Extract and find common patterns for sizes 3, 4, and 5
    for size in range(3, 6):  # Pattern sizes 3, 4, 5
        pattern_occurrences = extract_atom_patterns(json_data, size)
        common_patterns = find_common_patterns(pattern_occurrences, min_count=2)  # At least 2 files

        print(f"\nCommon Atom Patterns of Size {size} Across JSON Files:")
        for pattern, occurrences in common_patterns.items():
            print(f"Pattern: {pattern}, Occurrences: {dict(occurrences)}")

if __name__ == "__main__":
    main()
