"""
Helper script to run the examples with the correct Python path.

Usage:
    python run_examples.py [example_name]

Example:
    python run_examples.py input_processor_usage
    python run_examples.py test_api_with_processor
    python run_examples.py analyzer_integration
    python run_examples.py all  # Run all examples
"""

import sys
import os
import importlib.util
import subprocess

# List of available examples
EXAMPLES = [
    'input_processor_usage',
    'test_api_with_processor',
    'analyzer_integration',
]

def print_header(title):
    """Print a formatted header for each example."""
    print("\n" + "=" * 80)
    print(f"  RUNNING EXAMPLE: {title}")
    print("=" * 80 + "\n")

def run_example(example_name):
    """Run a specific example module."""
    if example_name not in EXAMPLES:
        print(f"Error: Example '{example_name}' not found.")
        print(f"Available examples: {', '.join(EXAMPLES)}")
        return False
    
    try:
        # Construct the path to the example file
        example_path = os.path.join('examples', f'{example_name}.py')
        
        # Check if the file exists
        if not os.path.exists(example_path):
            print(f"Error: Example file '{example_path}' not found.")
            return False
        
        print_header(example_name)
        
        # Run the example as a subprocess to ensure it has the correct environment
        result = subprocess.run([sys.executable, example_path], check=False)
        
        if result.returncode != 0:
            print(f"\nExample '{example_name}' exited with error code {result.returncode}")
            return False
            
        return True
    except Exception as e:
        print(f"Error running example '{example_name}': {e}")
        return False

def main():
    """Main function to run examples."""
    # Check if an example name was provided
    if len(sys.argv) < 2:
        print("Please specify an example to run:")
        print(f"  python run_examples.py [{'|'.join(EXAMPLES)}|all]")
        return
    
    example_name = sys.argv[1].lower()
    
    # Run all examples if requested
    if example_name == 'all':
        success = True
        for example in EXAMPLES:
            if not run_example(example):
                success = False
        
        if not success:
            print("\nSome examples failed. Please check the output above.")
            sys.exit(1)
    else:
        # Run the specified example
        if not run_example(example_name):
            sys.exit(1)
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main()
