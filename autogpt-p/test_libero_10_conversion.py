#!/usr/bin/env python3
"""
Test script for LIBERO_10 BDDL conversion to AutoGPT+P format
"""

import os
import sys

# Add the AutoGPT+P evaluation directory to the path
sys.path.append('/Users/stefantabakov/Desktop/autogpt-p/autogpt_p')

from autogpt_p.evaluation.libero_bddl_converter import BDDLConverter

def test_libero_10_conversion():
    """Test the LIBERO_10 BDDL to AutoGPT+P conversion"""
    
    # Paths
    libero_bddl_dir = "/Users/stefantabakov/Desktop/autogpt-p/LIBERO/libero/libero/bddl_files"
    output_scenes_dir = "/Users/stefantabakov/Desktop/autogpt-p/autogpt_p/autogpt_p/evaluation/data/libero/scenes"
    output_scenarios_dir = "/Users/stefantabakov/Desktop/autogpt-p/autogpt_p/autogpt_p/evaluation/data/libero/scenarios"
    
    print("🚀 Starting LIBERO_10 BDDL Conversion Test")
    print(f"LIBERO BDDL Directory: {libero_bddl_dir}")
    print(f"Output Scenes Directory: {output_scenes_dir}")
    print(f"Output Scenarios Directory: {output_scenarios_dir}")
    
    # Create converter instance
    converter = BDDLConverter(libero_bddl_dir, output_scenes_dir, output_scenarios_dir)
    
    # Test parsing a single BDDL file first
    print("\n📝 Testing single BDDL file parsing...")
    test_bddl_file = os.path.join(libero_bddl_dir, "libero_10", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl")
    
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
    
    # Test converting the libero_10 suite
    print("\n📦 Converting LIBERO_10 suite...")
    try:
        csv_file = converter.convert_suite_to_csv("libero_10")
        print(f"✅ Successfully converted libero_10 suite")
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

def check_libero_10_dependencies():
    """Check if all required dependencies for LIBERO_10 are available"""
    print("🔍 Checking LIBERO_10 dependencies...")
    
    try:
        # Check if LIBERO directory exists
        libero_dir = "/Users/stefantabakov/Desktop/autogpt-p/LIBERO"
        if not os.path.exists(libero_dir):
            print(f"❌ LIBERO directory not found: {libero_dir}")
            return False
        
        # Check if LIBERO_10 BDDL files exist
        bddl_dir = os.path.join(libero_dir, "libero/libero/bddl_files/libero_10")
        if not os.path.exists(bddl_dir):
            print(f"❌ LIBERO_10 BDDL directory not found: {bddl_dir}")
            return False
        
        # Check for some sample BDDL files
        sample_files = [
            "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
            "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl",
            "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl"
        ]
        
        for sample_file in sample_files:
            sample_path = os.path.join(bddl_dir, sample_file)
            if os.path.exists(sample_path):
                print(f"✅ Found sample BDDL file: {sample_file}")
            else:
                print(f"⚠️ Sample BDDL file not found: {sample_file}")
        
        # Count total BDDL files in libero_10
        bddl_files = [f for f in os.listdir(bddl_dir) if f.endswith('.bddl')]
        print(f"📊 Found {len(bddl_files)} BDDL files in libero_10")
        
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
        
        print("✅ All LIBERO_10 dependencies check passed")
        return True
        
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False

if __name__ == "__main__":
    print("🧪 LIBERO_10-to-AutoGPT+P Conversion Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_libero_10_dependencies():
        print("❌ Dependency check failed. Please ensure LIBERO_10 is properly set up.")
        sys.exit(1)
    
    # Run conversion test
    success = test_libero_10_conversion()
    
    if success:
        print("\n🎉 LIBERO_10 conversion test completed successfully!")
        print("You can now run the full LIBERO_10 evaluation using libero_planner_evaluation.py")
    else:
        print("\n💥 LIBERO_10 conversion test failed!")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)