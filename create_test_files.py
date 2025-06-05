# Create a test script to generate sample files
import os
from pathlib import Path

def create_test_files():
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create some text files
    (test_dir / "python_code.txt").write_text("""
    def hello_world():
        print("Hello, World!")
        
    class Calculator:
        def add(self, x, y):
            return x + y
    """)
    
    (test_dir / "meeting_notes.txt").write_text("""
    Team Meeting Notes - 2024
    Agenda:
    1. Project updates
    2. Timeline review
    3. Resource allocation
    """)
    
    (test_dir / "shopping_list.txt").write_text("""
    Grocery Shopping List:
    - Milk
    - Eggs
    - Bread
    - Fruits
    - Vegetables
    """)
    
    # Create a subdirectory with more files
    docs_dir = test_dir / "documents"
    docs_dir.mkdir(exist_ok=True)
    
    (docs_dir / "report.txt").write_text("""
    Quarterly Report 2024
    Sales increased by 15%
    New market expansion planned
    """)
    
    (docs_dir / "todo.txt").write_text("""
    TODO List:
    1. Finish project
    2. Review code
    3. Update documentation
    """)

if __name__ == "__main__":
    create_test_files()