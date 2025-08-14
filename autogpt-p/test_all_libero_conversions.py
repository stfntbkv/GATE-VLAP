#!/usr/bin/env python3
"""
Master script to convert all LIBERO suites to AutoGPT+P format
"""

import os
import sys

# Add the AutoGPT+P evaluation directory to the path
sys.path.append('/Users/stefantabakov/Desktop/autogpt-p/autogpt_p')

from autogpt_p.evaluation.libero_bddl_converter import BDDLConverter

def convert_all_libero_suites():
    """Convert all LIBERO suites to AutoGPT+P format"""
    
    # Paths
    libero_bddl_dir = "/Users/stefantabakov/Desktop/autogpt-p/LIBERO/libero/libero/bddl_files"
    output_scenes_dir = "/Users/stefantabakov/Desktop/autogpt-p/autogpt_p/autogpt_p/evaluation/data/libero/scenes"
    output_scenarios_dir = "/Users/stefantabakov/Desktop/autogpt-p/autogpt_p/autogpt_p/evaluation/data/libero/scenarios"
    
    print("🚀 Starting All LIBERO Suites BDDL Conversion")
    print("=" * 60)
    print(f"LIBERO BDDL Directory: {libero_bddl_dir}")
    print(f"Output Scenes Directory: {output_scenes_dir}")
    print(f"Output Scenarios Directory: {output_scenarios_dir}")
    
    # Create converter instance
    converter = BDDLConverter(libero_bddl_dir, output_scenes_dir, output_scenarios_dir)
    
    # Define all suites to convert
    suites = ['libero_goal', 'libero_10', 'libero_90', 'libero_spatial', 'libero_object']
    
    results = {}
    total_scenarios = 0
    
    for suite in suites:
        print(f"\n📦 Converting {suite.upper()} suite...")
        print("-" * 40)
        
        try:
            # Check if suite directory exists
            suite_dir = os.path.join(libero_bddl_dir, suite)
            if not os.path.exists(suite_dir):
                print(f"❌ Suite directory not found: {suite_dir}")
                results[suite] = "FAILED - Directory not found"
                continue
            
            # Count BDDL files
            bddl_files = [f for f in os.listdir(suite_dir) if f.endswith('.bddl')]
            print(f"📊 Found {len(bddl_files)} BDDL files in {suite}")
            
            # Convert suite
            csv_file = converter.convert_suite_to_csv(suite)
            
            # Verify conversion results
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    scenario_count = len(lines) - 1  # Subtract header
                    total_scenarios += scenario_count
                    print(f"✅ Successfully converted {scenario_count} scenarios")
                    print(f"📄 Generated CSV: {csv_file}")
                    results[suite] = f"SUCCESS - {scenario_count} scenarios"
                    
                    # Show first scenario as example
                    if len(lines) > 1:
                        print(f"📝 Sample scenario: {lines[1].strip()[:80]}...")
            else:
                print(f"❌ CSV file not created: {csv_file}")
                results[suite] = "FAILED - No CSV output"
                
        except Exception as e:
            print(f"❌ Error converting {suite}: {e}")
            results[suite] = f"FAILED - {str(e)[:50]}..."
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n🎯 CONVERSION SUMMARY")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for suite, result in results.items():
        status = "✅" if "SUCCESS" in result else "❌"
        print(f"{status} {suite.upper()}: {result}")
        if "SUCCESS" in result:
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Overall Results:")
    print(f"   ✅ Successful conversions: {successful}/{len(suites)}")
    print(f"   ❌ Failed conversions: {failed}/{len(suites)}")
    print(f"   📝 Total scenarios generated: {total_scenarios}")
    
    if successful == len(suites):
        print("\n🎉 All LIBERO suite conversions completed successfully!")
        print("You can now run evaluations on any LIBERO suite using libero_planner_evaluation.py")
        return True
    else:
        print("\n⚠️ Some conversions failed. Please check the errors above.")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check if LIBERO directory exists
    libero_dir = "/Users/stefantabakov/Desktop/autogpt-p/LIBERO"
    if not os.path.exists(libero_dir):
        print(f"❌ LIBERO directory not found: {libero_dir}")
        return False
    
    # Check if BDDL files directory exists
    bddl_dir = os.path.join(libero_dir, "libero/libero/bddl_files")
    if not os.path.exists(bddl_dir):
        print(f"❌ LIBERO BDDL directory not found: {bddl_dir}")
        return False
    
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
    
    # Check available suites
    suites = ['libero_goal', 'libero_10', 'libero_90', 'libero_spatial', 'libero_object']
    available_suites = []
    
    for suite in suites:
        suite_dir = os.path.join(bddl_dir, suite)
        if os.path.exists(suite_dir):
            bddl_count = len([f for f in os.listdir(suite_dir) if f.endswith('.bddl')])
            print(f"✅ Found {suite}: {bddl_count} BDDL files")
            available_suites.append(suite)
        else:
            print(f"❌ Missing {suite}")
    
    if len(available_suites) == len(suites):
        print("✅ All dependencies check passed")
        return True
    else:
        print(f"⚠️ Only {len(available_suites)}/{len(suites)} suites available")
        return False

if __name__ == "__main__":
    print("🧪 LIBERO All Suites Conversion Test")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Dependency check failed. Please ensure all LIBERO suites are properly set up.")
        sys.exit(1)
    
    # Run all conversions
    success = convert_all_libero_suites()
    
    if success:
        print("\n🎉 All LIBERO conversions completed successfully!")
    else:
        print("\n💥 Some LIBERO conversions failed!")
        sys.exit(1)