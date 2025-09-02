#!/usr/bin/env python3
"""
Test File Organization Script
=============================
This script categorizes and organizes test files into logical subdirectories
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def categorize_test_files():
    """Categorize test files based on their names and purposes"""
    
    # Define categories and their patterns
    categories = {
        'api': {
            'patterns': ['api', 'endpoint', 'postman'],
            'description': 'API endpoint and integration tests'
        },
        'identity': {
            'patterns': ['identity', 'who_are_you'],
            'description': 'Identity detection and response tests'
        },
        'personality': {
            'patterns': ['personality', 'traits', 'classification'],
            'description': 'Personality analysis and classification tests'
        },
        'conversation': {
            'patterns': ['conversation', 'history', 'flow', 'tracking'],
            'description': 'Conversation management and flow tests'
        },
        'greeting': {
            'patterns': ['greeting', 'hello'],
            'description': 'Greeting detection and response tests'
        },
        'language': {
            'patterns': ['arabic', 'english', 'language'],
            'description': 'Language-specific tests'
        },
        'detection': {
            'patterns': ['detection', 'enhanced', 'smart', 'improved'],
            'description': 'Detection algorithms and improvements'
        },
        'integration': {
            'patterns': ['integration', 'full', 'complete', 'comprehensive'],
            'description': 'Integration and comprehensive tests'
        },
        'edge_cases': {
            'patterns': ['edge', 'scenario', 'case', 'issue', 'fix'],
            'description': 'Edge cases and specific scenarios'
        },
        'utilities': {
            'patterns': ['analyzer', 'organization', 'cleanup'],
            'description': 'Test utilities and helper scripts'
        }
    }
    
    tests_dir = Path("tests")
    
    # Get all test files
    test_files = list(tests_dir.glob("test_*.py"))
    
    # Categorize files
    categorized_files = defaultdict(list)
    uncategorized_files = []
    
    for test_file in test_files:
        filename = test_file.name.lower()
        categorized = False
        
        # Try to match with categories
        for category, info in categories.items():
            for pattern in info['patterns']:
                if pattern in filename:
                    categorized_files[category].append(test_file)
                    categorized = True
                    break
            if categorized:
                break
        
        if not categorized:
            uncategorized_files.append(test_file)
    
    return categories, categorized_files, uncategorized_files

def create_organized_structure():
    """Create organized directory structure and move files"""
    
    print("="*80)
    print("TEST DIRECTORY ORGANIZATION")
    print("="*80)
    print()
    
    categories, categorized_files, uncategorized_files = categorize_test_files()
    tests_dir = Path("tests")
    
    # Show categorization plan
    print("📋 CATEGORIZATION PLAN:")
    print("-" * 50)
    
    total_files = 0
    for category, files in categorized_files.items():
        if files:
            print(f"📁 {category:15} | {len(files):3} files | {categories[category]['description']}")
            total_files += len(files)
    
    if uncategorized_files:
        print(f"❓ uncategorized    | {len(uncategorized_files):3} files | Files that don't fit clear categories")
        total_files += len(uncategorized_files)
    
    print(f"\n📊 Total files to organize: {total_files}")
    print()
    
    # Show sample files for each category
    print("📂 SAMPLE FILES PER CATEGORY:")
    print("-" * 50)
    for category, files in categorized_files.items():
        if files:
            print(f"\n📁 {category.upper()}:")
            for i, file in enumerate(sorted(files)[:5]):  # Show first 5 files
                print(f"   • {file.name}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
    
    if uncategorized_files:
        print(f"\n❓ UNCATEGORIZED:")
        for file in sorted(uncategorized_files)[:10]:  # Show first 10
            print(f"   • {file.name}")
        if len(uncategorized_files) > 10:
            print(f"   ... and {len(uncategorized_files) - 10} more files")
    
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with organizing the files? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print()
        print("🚀 ORGANIZING FILES...")
        print()
        
        # Create directories and move files
        moved_count = 0
        created_dirs = set()
        
        for category, files in categorized_files.items():
            if files:
                category_dir = tests_dir / category
                category_dir.mkdir(exist_ok=True)
                created_dirs.add(category)
                
                print(f"📁 Created directory: tests/{category}/")
                
                for file in files:
                    try:
                        destination = category_dir / file.name
                        shutil.move(str(file), str(destination))
                        moved_count += 1
                        print(f"   ✅ Moved: {file.name}")
                    except Exception as e:
                        print(f"   ❌ Failed to move {file.name}: {e}")
        
        # Handle uncategorized files
        if uncategorized_files:
            misc_dir = tests_dir / "misc"
            misc_dir.mkdir(exist_ok=True)
            created_dirs.add("misc")
            
            print(f"\n📁 Created directory: tests/misc/")
            
            for file in uncategorized_files:
                try:
                    destination = misc_dir / file.name
                    shutil.move(str(file), str(destination))
                    moved_count += 1
                    print(f"   ✅ Moved: {file.name}")
                except Exception as e:
                    print(f"   ❌ Failed to move {file.name}: {e}")
        
        # Move non-test files to utilities
        non_test_files = [f for f in tests_dir.glob("*.py") if not f.name.startswith("test_")]
        if non_test_files:
            if "utilities" not in created_dirs:
                utils_dir = tests_dir / "utilities"
                utils_dir.mkdir(exist_ok=True)
                created_dirs.add("utilities")
                print(f"\n📁 Created directory: tests/utilities/")
            else:
                utils_dir = tests_dir / "utilities"
            
            for file in non_test_files:
                try:
                    destination = utils_dir / file.name
                    shutil.move(str(file), str(destination))
                    moved_count += 1
                    print(f"   ✅ Moved utility: {file.name}")
                except Exception as e:
                    print(f"   ❌ Failed to move {file.name}: {e}")
        
        print(f"\n✅ ORGANIZATION COMPLETED!")
        print(f"📊 Total files moved: {moved_count}")
        print(f"📁 Directories created: {len(created_dirs)}")
        print("=" * 80)
        
        # Create README files for each directory
        create_readme_files(categories, created_dirs)
        
    else:
        print("\n❌ Organization cancelled by user")

def create_readme_files(categories, created_dirs):
    """Create README files for each directory"""
    print("\n📝 Creating README files...")
    
    tests_dir = Path("tests")
    
    for dir_name in created_dirs:
        readme_path = tests_dir / dir_name / "README.md"
        
        if dir_name in categories:
            description = categories[dir_name]['description']
            patterns = ", ".join(categories[dir_name]['patterns'])
        elif dir_name == "misc":
            description = "Miscellaneous test files that don't fit into specific categories"
            patterns = "various"
        elif dir_name == "utilities":
            description = "Test utilities, cleanup scripts, and helper tools"
            patterns = "utility, helper, cleanup"
        else:
            description = f"Test files for {dir_name}"
            patterns = "various"
        
        readme_content = f"""# {dir_name.title()} Tests

## Description
{description}

## File Patterns
Files in this directory typically contain: {patterns}

## Purpose
This directory contains test files related to {dir_name} functionality.

---
*Auto-generated on {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}*
"""
        
        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            print(f"   ✅ Created: {dir_name}/README.md")
        except Exception as e:
            print(f"   ❌ Failed to create {dir_name}/README.md: {e}")

if __name__ == "__main__":
    create_organized_structure()
