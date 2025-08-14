#!/usr/bin/env python3
"""
Test script for LIBERO_90 BDDL conversion to AutoGPT+P format
"""

import os
import sys

# Add the AutoGPT+P evaluation directory to the path
sys.path.append('/Users/stefantabakov/Desktop/autogpt-p/autogpt_p')

from autogpt_p.evaluation.libero_bddl_converter import BDDLConverter

def test_libero_90_conversion():
    """Test the LIBERO_90 BDDL to AutoGPT+P conversion"""
    
    # Paths
    libero_bddl_dir = "/Users/stefantabakov/Desktop/autogpt-p/LIBERO/libero/libero/bddl_files"
    output_scenes_dir = "/Users/stefantabakov/Desktop/autogpt-p/autogpt_p/autogpt_p/evaluation/data/libero/scenes"
    output_scenarios_dir = "/Users/stefantabakov/Desktop/autogpt-p/autogpt_p/autogpt_p/evaluation/data/libero/scenarios"
    
    print("🚀 Starting LIBERO_90 BDDL Conversion Test")
    print(f"LIBERO BDDL Directory: {libero_bddl_dir}")
    print(f"Output Scenes Directory: {output_scenes_dir}")
    print(f"Output Scenarios Directory: {output_scenarios_dir}")
    
    # Create converter instance
    converter = BDDLConverter(libero_bddl_dir, output_scenes_dir, output_scenarios_dir)
    
    # Test parsing a single BDDL file first
    print("\n📝 Testing single BDDL file parsing...")
    test_bddl_file = os.path.join(libero_bddl_dir, "libero_90", "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl")
    
    if os.path.exists(test_bddl_file):
        print(f"Parsing: {test_bddl_file}")
        try:
            parsed_data = converter.parse_bddl_file(test_bddl_file)
            print("✅ Successfully parsed BDDL file:")
            print(f"  Language: {parsed_data['language']}")
            print(f"  Objects: {parsed_data['objects']}")
            print(f"  Fixtures: {parsed_data['fixtures']}")
            print(f"  Init State: {parsed_data['init_state']}")
            print(f"  Goal: {parsed_data['goal']}")
            
            # Test scene file generation
            print("\n🏗️ Generating scene file...")
            scene_content = converter.generate_scene_file(parsed_data)
            print("✅ Successfully generated scene content:")
            print("--- Scene Content ---")
            print(scene_content[:500] + ("..." if len(scene_content) > 500 else ""))
            print("--- End Scene Content ---")
            
        except Exception as e:
            print(f"❌ Error parsing BDDL file: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"❌ Test BDDL file not found: {test_bddl_file}")
        return False
    
    # Test converting the libero_90 suite
    print("\n📦 Converting LIBERO_90 suite...")
    try:
        csv_file = converter.convert_suite_to_csv("libero_90")
        print(f"✅ Successfully converted libero_90 suite")
        print(f"Generated CSV: {csv_file}")
        
        # Check if files were created
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                print(f"CSV file contains {len(lines)} lines (including header)")
                if len(lines) > 1:
                    print("First scenario:")
                    print(f"  Header: {lines[0].strip()}")
                    print(f"  Sample: {lines[1].strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting suite: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_libero_90_dependencies():
    """Check if all required dependencies for LIBERO_90 are available"""
    print("🔍 Checking LIBERO_90 dependencies...")
    
    try:
        # Check if LIBERO directory exists
        libero_dir = "/Users/stefantabakov/Desktop/autogpt-p/LIBERO"
        if not os.path.exists(libero_dir):
            print(f"❌ LIBERO directory not found: {libero_dir}")
            return False
        
        # Check if LIBERO_90 BDDL files exist
        bddl_dir = os.path.join(libero_dir, "libero/libero/bddl_files/libero_90")
        if not os.path.exists(bddl_dir):
            print(f"❌ LIBERO_90 BDDL directory not found: {bddl_dir}")
            return False
        
        # Count total BDDL files in libero_90
        bddl_files = [f for f in os.listdir(bddl_dir) if f.endswith('.bddl')]
        print(f"📊 Found {len(bddl_files)} BDDL files in libero_90")
        
        # Check output directories exist
        output_dirs = [
            "/Users/stefantabakov/Desktop/autogpt-p/autogpt_p/autogpt_p/evaluation/data/libero/scenes",
            "/Users/stefantabakov/Desktop/autogpt-p/autogpt_p/autogpt_p/evaluation/data/libero/scenarios"
        ]
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                print(f"✅ Output directory exists: {output_dir}")
            else:
                print(f"❌ Output directory missing: {output_dir}")
                return False
        
        print("✅ All LIBERO_90 dependencies check passed")
        return True
        
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False

if __name__ == "__main__":
    print("🧪 LIBERO_90-to-AutoGPT+P Conversion Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_libero_90_dependencies():
        print("❌ Dependency check failed. Please ensure LIBERO_90 is properly set up.")
        sys.exit(1)
    
    # Run conversion test
    success = test_libero_90_conversion()
    
    if success:
        print("\n🎉 LIBERO_90 conversion test completed successfully!")
        print("You can now run the full LIBERO_90 evaluation using libero_planner_evaluation.py")
    else:
        print("\n💥 LIBERO_90 conversion test failed!")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)