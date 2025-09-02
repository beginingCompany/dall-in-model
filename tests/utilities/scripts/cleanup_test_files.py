#!/usr/bin/env python3
"""
Test File Management Script
============================
This script organizes and cleans up test files by:
1. Identifying empty test files (size = 0 bytes)
2. Moving non-empty test files from root to tests/ directory
3. Removing empty test files
4. Creating a summary report
"""

import os
import shutil
from pathlib import Path

def get_file_size(filepath):
    """Get file size in bytes"""
    try:
        return os.path.getsize(filepath)
    except:
        return 0

def organize_test_files():
    """Main function to organize and clean test files"""
    root_dir = Path("c:/Users/aalma/Desktop/python/dall-in-model")
    tests_dir = root_dir / "tests"
    
    # Ensure tests directory exists
    tests_dir.mkdir(exist_ok=True)
    
    # Find all test files
    all_test_files = []
    
    # Files in root directory
    root_test_files = list(root_dir.glob("test_*.py"))
    for f in root_test_files:
        all_test_files.append(("root", f, get_file_size(f)))
    
    # Files in tests directory
    tests_test_files = list(tests_dir.glob("test_*.py"))
    for f in tests_test_files:
        all_test_files.append(("tests", f, get_file_size(f)))
    
    # Also check for demo/debug/verify files
    other_files = []
    for pattern in ["demo_*.py", "debug_*.py", "final_*.py", "*_verification.py"]:
        other_files.extend(list(root_dir.glob(pattern)))
    
    for f in other_files:
        all_test_files.append(("root", f, get_file_size(f)))
    
    # Separate empty and non-empty files
    empty_files = []
    non_empty_files = []
    root_files_to_move = []
    
    for location, filepath, size in all_test_files:
        if size == 0:
            empty_files.append((location, filepath, size))
        else:
            non_empty_files.append((location, filepath, size))
            if location == "root":
                root_files_to_move.append((filepath, size))
    
    print("="*80)
    print("TEST FILE CLEANUP AND ORGANIZATION REPORT")
    print("="*80)
    print()
    
    # Report empty files
    if empty_files:
        print(f"🗑️  EMPTY FILES TO DELETE ({len(empty_files)} files):")
        print("-" * 50)
        for location, filepath, size in sorted(empty_files, key=lambda x: x[1].name):
            print(f"   📁 {location:5} | {filepath.name}")
    else:
        print("✅ No empty test files found")
    
    print()
    
    # Report files to move from root to tests
    if root_files_to_move:
        print(f"📁 NON-EMPTY ROOT FILES TO MOVE TO tests/ ({len(root_files_to_move)} files):")
        print("-" * 50)
        for filepath, size in sorted(root_files_to_move, key=lambda x: x[1]):
            size_kb = size / 1024
            print(f"   📄 {filepath.name:40} | {size_kb:6.1f} KB")
    else:
        print("✅ No test files to move from root")
    
    print()
    
    # Report non-empty files already in tests directory
    tests_files = [(fp, sz) for loc, fp, sz in non_empty_files if loc == "tests"]
    if tests_files:
        print(f"📋 NON-EMPTY FILES ALREADY IN tests/ ({len(tests_files)} files):")
        print("-" * 50)
        # Show only largest files to avoid spam
        tests_files_sorted = sorted(tests_files, key=lambda x: x[1], reverse=True)
        for filepath, size in tests_files_sorted[:10]:  # Top 10 largest
            size_kb = size / 1024
            print(f"   📄 {filepath.name:40} | {size_kb:6.1f} KB")
        if len(tests_files) > 10:
            print(f"   ... and {len(tests_files) - 10} more files")
    else:
        print("ℹ️  No non-empty files in tests directory")
    
    print()
    
    # Ask for confirmation
    print("📋 SUMMARY:")
    print(f"   • Empty files to delete: {len(empty_files)}")
    print(f"   • Root files to move to tests/: {len(root_files_to_move)}")
    print(f"   • Total non-empty files in tests/: {len(tests_files)}")
    print()
    
    response = input("Do you want to proceed with cleanup? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print()
        print("🚀 STARTING CLEANUP...")
        print()
        
        # Delete empty files
        deleted_count = 0
        for location, filepath, size in empty_files:
            try:
                filepath.unlink()
                print(f"   ✅ Deleted: {filepath.name}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {filepath.name}: {e}")
        
        print(f"\n🗑️  Deleted {deleted_count} empty files")
        
        # Move root files to tests directory
        moved_count = 0
        for filepath, size in root_files_to_move:
            try:
                destination = tests_dir / filepath.name
                # Check if file already exists in tests
                if destination.exists():
                    print(f"   ⚠️  Skipped {filepath.name} (already exists in tests/)")
                    continue
                
                shutil.move(str(filepath), str(destination))
                print(f"   ✅ Moved: {filepath.name} -> tests/")
                moved_count += 1
            except Exception as e:
                print(f"   ❌ Failed to move {filepath.name}: {e}")
        
        print(f"\n📁 Moved {moved_count} files to tests/ directory")
        
        print()
        print("✅ CLEANUP COMPLETED!")
        print("=" * 80)
        
    else:
        print("\n❌ Cleanup cancelled by user")
        print("=" * 80)

if __name__ == "__main__":
    organize_test_files()
