#!/usr/bin/env python3
"""
Test script for LIBERO_GOAL BDDL conversion to AutoGPT+P format
"""

import os
import sys

# Add the AutoGPT+P evaluation directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'autogpt_p'))

from autogpt_p.evaluation.libero_bddl_converter import BDDLConverter

def test_libero_goal_conversion():
    """Test the LIBERO_GOAL BDDL to AutoGPT+P conversion"""
    
    # Paths (relative to script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libero_bddl_dir = os.path.join(script_dir, "LIBERO/libero/libero/bddl_files")
    output_scenes_dir = os.path.join(script_dir, "autogpt_p/autogpt_p/evaluation/data/libero/scenes")
    output_scenarios_dir = os.path.join(script_dir, "autogpt_p/autogpt_p/evaluation/data/libero/scenarios")
    
    print("🚀 Starting LIBERO_GOAL BDDL Conversion Test")
    print(f"LIBERO BDDL Directory: {libero_bddl_dir}")
    print(f"Output Scenes Directory: {output_scenes_dir}")
    print(f"Output Scenarios Directory: {output_scenarios_dir}")
    
    # Create converter instance
    converter = BDDLConverter(libero_bddl_dir, output_scenes_dir, output_scenarios_dir)
    
    # Test parsing a single BDDL file first
    print("\n📝 Testing single BDDL file parsing...")
    test_bddl_file = os.path.join(libero_bddl_dir, "libero_goal", "put_the_bowl_on_the_plate.bddl")
    
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
    
    # Test converting the libero_goal suite
    print("\n📦 Converting LIBERO_GOAL suite...")
    try:
        csv_file = converter.convert_suite_to_csv("libero_goal")
        print(f"✅ Successfully converted libero_goal suite")
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

def check_libero_goal_dependencies():
    """Check if all required dependencies for LIBERO_GOAL are available"""
    print("🔍 Checking LIBERO_GOAL dependencies...")
    
    try:
        # Check if LIBERO directory exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        libero_dir = os.path.join(script_dir, "LIBERO")
        if not os.path.exists(libero_dir):
            print(f"❌ LIBERO directory not found: {libero_dir}")
            return False
        
        # Check if LIBERO_GOAL BDDL files exist
        bddl_dir = os.path.join(libero_dir, "libero/libero/bddl_files/libero_goal")
        if not os.path.exists(bddl_dir):
            print(f"❌ LIBERO_GOAL BDDL directory not found: {bddl_dir}")
            return False
        
        # Count total BDDL files in libero_goal
        bddl_files = [f for f in os.listdir(bddl_dir) if f.endswith('.bddl')]
        print(f"📊 Found {len(bddl_files)} BDDL files in libero_goal")
        
        # Check output directories exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dirs = [
            os.path.join(script_dir, "autogpt_p/autogpt_p/evaluation/data/libero/scenes"),
            os.path.join(script_dir, "autogpt_p/autogpt_p/evaluation/data/libero/scenarios")
        ]
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                print(f"✅ Output directory exists: {output_dir}")
            else:
                print(f"❌ Output directory missing: {output_dir}")
                return False
        
        print("✅ All LIBERO_GOAL dependencies check passed")
        return True
        
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False

if __name__ == "__main__":
    print("🧪 LIBERO_GOAL-to-AutoGPT+P Conversion Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_libero_goal_dependencies():
        print("❌ Dependency check failed. Please ensure LIBERO_GOAL is properly set up.")
        sys.exit(1)
    
    # Run conversion test
    success = test_libero_goal_conversion()
    
    if success:
        print("\n🎉 LIBERO_GOAL conversion test completed successfully!")
        print("You can now run the full LIBERO_GOAL evaluation using libero_planner_evaluation.py")
    else:
        print("\n💥 LIBERO_GOAL conversion test failed!")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)