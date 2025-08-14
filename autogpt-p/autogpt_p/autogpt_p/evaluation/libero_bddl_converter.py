import os
import re
from typing import Dict, List, Tuple, Optional
import csv

# Mapping from LIBERO predicates to AutoGPT+P PDDL predicates
LIBERO_TO_PDDL_PREDICATES = {
    "On": "on",
    "In": "in", 
    "Turnon": "turn_on",
    "Turnoff": "turn_off",
    "Open": "opened",
    "Close": "closed",
    "And": "and",
    "Or": "or",
    "Not": "not"
}

# Note: Object types are now handled by the OAM system (libero_oam.json) 
# which provides multi-affordance classification instead of single-type mapping


class BDDLConverter:
    """Converts LIBERO BDDL files to AutoGPT+P scene files and CSV scenarios"""
    
    def __init__(self, libero_bddl_dir: str, output_scenes_dir: str, output_scenarios_dir: str):
        self.libero_bddl_dir = libero_bddl_dir
        self.output_scenes_dir = output_scenes_dir
        self.output_scenarios_dir = output_scenarios_dir
        
        # Ensure output directories exist
        os.makedirs(output_scenes_dir, exist_ok=True)
        os.makedirs(output_scenarios_dir, exist_ok=True)
    
    def parse_bddl_file(self, bddl_path: str) -> Dict:
        """Parse a LIBERO BDDL file and extract relevant information"""
        with open(bddl_path, 'r') as file:
            content = file.read()
        
        # Extract components using regex patterns
        parsed = {
            'language': self._extract_language(content),
            'objects': self._extract_objects(content),
            'fixtures': self._extract_fixtures(content),
            'init_state': self._extract_init_state(content),
            'goal': self._extract_goal(content),
            'filename': os.path.basename(bddl_path).replace('.bddl', '')
        }
        
        return parsed
    
    def _extract_language(self, content: str) -> str:
        """Extract the natural language task description"""
        match = re.search(r'\(:language\s+([^)]+)\)', content)
        if match:
            return match.group(1).strip().strip('"')
        return ""
    
    def _extract_objects(self, content: str) -> List[Tuple[str, str]]:
        """Extract objects and their types"""
        objects = []
        
        # Extract from (:objects ...) section
        objects_match = re.search(r'\(:objects\s+(.*?)\)', content, re.DOTALL)
        if objects_match:
            objects_text = objects_match.group(1)
            # Parse "name - type" pairs, including multiple objects of same type on one line
            for line in objects_text.strip().split('\n'):
                line = line.strip()
                if '-' in line and not line.startswith('('):
                    parts = line.split('-')
                    if len(parts) == 2:
                        names_part = parts[0].strip()
                        obj_type = parts[1].strip()
                        
                        # Handle multiple object names on one line (e.g., "plate_1 plate_2 - plate")
                        names = names_part.split()
                        for name in names:
                            objects.append((name.strip(), obj_type))
        
        return objects
    
    def _extract_fixtures(self, content: str) -> List[Tuple[str, str]]:
        """Extract fixtures (fixed environment objects) and their types"""
        fixtures = []
        
        # Extract from (:fixtures ...) section
        fixtures_match = re.search(r'\(:fixtures\s+(.*?)\)', content, re.DOTALL)
        if fixtures_match:
            fixtures_text = fixtures_match.group(1)
            # Parse "name - type" pairs
            for line in fixtures_text.strip().split('\n'):
                line = line.strip()
                if '-' in line:
                    parts = line.split('-')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        fixture_type = parts[1].strip()
                        fixtures.append((name, fixture_type))
        
        return fixtures
    
    def _extract_init_state(self, content: str) -> List[str]:
        """Extract initial state predicates"""
        init_predicates = []
        
        # Extract from (:init ...) section
        init_match = re.search(r'\(:init\s+(.*?)\n\s*\)', content, re.DOTALL)
        if init_match:
            init_text = init_match.group(1)
            # Find all predicates in parentheses
            predicates = re.findall(r'\([^)]+\)', init_text)
            for predicate in predicates:
                # Clean up the predicate
                clean_pred = predicate.strip('()')
                if clean_pred.strip():  # Only add non-empty predicates
                    init_predicates.append(clean_pred)
        
        return init_predicates
    
    def _extract_goal(self, content: str) -> str:
        """Extract goal state"""
        goal_match = re.search(r'\(:goal\s+(.*?)\n\s*\)', content, re.DOTALL)
        if goal_match:
            return goal_match.group(1).strip()
        return ""
    
    def convert_libero_predicates(self, predicate_text: str) -> str:
        """Convert LIBERO predicates to AutoGPT+P PDDL format"""
        # Replace LIBERO predicates with PDDL equivalents
        converted = predicate_text
        for libero_pred, pddl_pred in LIBERO_TO_PDDL_PREDICATES.items():
            converted = re.sub(r'\b' + libero_pred + r'\b', pddl_pred, converted)
        
        # Handle complex region names - they all refer to table locations
        # e.g., "kitchen_table_plate_right_region" -> "kitchen_table"
        # e.g., "study_table_desk_caddy_right_region" -> "study_table"
        # These will be mapped to table:0 later via object_mapping
        if '_table_' in converted and '_region' in converted:
            # Extract the table type (kitchen_table, study_table, etc)
            # This will be properly mapped later
            converted = re.sub(r'(\w+_table)_\w+_region', r'\1', converted)
        
        # Remove remaining region suffixes (e.g., "kitchen_table_region" → "kitchen_table")
        converted = re.sub(r'_[a-zA-Z_]*region', '', converted)
        converted = re.sub(r'_[a-zA-Z_]*_region', '', converted)
        
        # Debug: print conversion steps (commented out for production)
        # print(f"Converting predicate: '{predicate_text}' -> '{converted}'")
        
        # Convert to lowercase and clean up
        converted = converted.lower()
        
        return converted
    
    def _fix_goal_format(self, goal: str) -> str:
        """Fix common goal formatting issues for AutoGPT+P compatibility"""
        # Replace location references
        goal = goal.replace("main", "table0")
        goal = goal.replace("living_room_table", "table0") 
        goal = goal.replace("living", "table0")
        
        # Fix specific object name patterns  
        goal = re.sub(r"_\d+_top_side", "0", goal)  # cabinet_1_top_side -> cabinet0
        goal = re.sub(r"_top_side", "", goal)  # remove _top_side suffix
        
        # Convert LIBERO object naming (_1) to PDDL Object format (0, 1, 2...)
        goal = re.sub(r"(\w+)_1\b", r"\g<1>0", goal)  # object_1 -> object0
        goal = re.sub(r"(\w+)_2\b", r"\g<1>1", goal)  # object_2 -> object1
        goal = re.sub(r"(\w+)_3\b", r"\g<1>2", goal)  # object_3 -> object2
        
        # Remove outer parentheses if present to match SAYCAN format
        if goal.startswith("(") and goal.endswith(")"):
            goal = goal[1:-1]
        
        # Fix multi-part AND goals - ensure each predicate is properly parenthesized
        if goal.lower().startswith("and"):
            # Find all predicates after "and"
            predicates = re.findall(r'\([^)]+\)', goal)
            if predicates:
                # Reconstruct with proper formatting
                goal = "and " + " ".join(predicates)
        
        # Add proper spacing - the SAYCAN format has extra spaces
        goal = re.sub(r"(\w+)\s+(\w+)\s+(\w+)", r"\1  \2 \3", goal)
        
        # Convert colon notation to PDDL Object format (object:0 -> object0)
        goal = re.sub(r"(\w+):(\d+)", r"\1\2", goal)
        
        return goal
    
    def generate_scene_file(self, parsed_data: Dict) -> str:
        """Generate AutoGPT+P scene file content"""
        scene_content = "SCENE-DESCRIPTION\nOBJECTS\n"
        
        # Add objects with mapped types - create object name mapping
        all_objects = parsed_data['objects'] + parsed_data['fixtures']
        object_mapping = {}  # Maps LIBERO names to AutoGPT+P names
        object_counter = {}
        
        # Add human and robot first
        scene_content += "human:0\nrobot:0\n"
        
        for obj_name, obj_type in all_objects:
            # Map table-like objects to generic "table"
            if obj_type in ["kitchen_table", "living_room_table", "study_table"]:
                base_name = "table"
            else:
                # Use LIBERO object type as the base name (this matches the OAM system)
                base_name = obj_type
                
            if base_name not in object_counter:
                object_counter[base_name] = 0
            else:
                object_counter[base_name] += 1
            
            # Create consistent object name mapping - use colon format to match SayCan
            autogpt_name = f"{base_name}:{object_counter[base_name]}"
            object_mapping[obj_name] = autogpt_name
            
            scene_content += f"{autogpt_name}\n"
        
        scene_content += "END-OBJECTS\n"
        
        # Add relations section
        scene_content += "RELATIONS\n"
        
        # Convert initial state predicates with proper object name mapping
        for init_pred in parsed_data['init_state']:
            converted_pred = self.convert_libero_predicates(init_pred)
            
            # Replace LIBERO object names with AutoGPT+P names (order matters!)
            for libero_name, autogpt_name in object_mapping.items():
                # Use word boundaries to avoid partial replacements
                converted_pred = re.sub(r'\b' + re.escape(libero_name) + r'\b', autogpt_name, converted_pred)
            
            # Replace main table references with table:0
            converted_pred = re.sub(r'\bmain_table\b', "table:0", converted_pred)
            converted_pred = re.sub(r'\bmain\b', "table:0", converted_pred)
            
            # Handle living room table references
            converted_pred = re.sub(r'\bliving_room_table\b', "table:0", converted_pred)
            converted_pred = re.sub(r'\bliving\b', "table:0", converted_pred)
            
            # Replace various table types with table:0 if not already mapped
            converted_pred = re.sub(r'\bkitchen_table\b', "table:0", converted_pred)
            converted_pred = re.sub(r'\bstudy_table\b', "table:0", converted_pred)
            
            # Replace standalone location names (from region removal) with table:0
            converted_pred = re.sub(r'\bkitchen\b', "table:0", converted_pred)
            converted_pred = re.sub(r'\bstudy\b', "table:0", converted_pred)
            
            # Fix regional object references that weren't caught by region removal
            # Replace references like floor_0, floor_1 with the actual object
            for libero_name, autogpt_name in object_mapping.items():
                base_name = libero_name
                # Replace patterns like "floor_0", "floor_1" with the mapped object name
                converted_pred = re.sub(r'\b' + re.escape(base_name) + r'_\d+\b', autogpt_name, converted_pred)
            
            # Handle compound object-part references like microwave_1_top_side
            # These should be converted to just the base object (microwave:0)
            for libero_name, autogpt_name in object_mapping.items():
                # Extract base object name without the instance number
                base_obj = re.sub(r'_\d+$', '', libero_name)  # Remove trailing _number
                # Replace patterns like "microwave_1_top_side", "desk_caddy_1_right_region" 
                pattern = rf'\b{re.escape(base_obj)}_\d+_[a-zA-Z_]+\b'
                converted_pred = re.sub(pattern, autogpt_name, converted_pred)
            
            # Handle complex regional references like "study_table_desk_caddy_front_left_contain_region"
            # These should be simplified to the target object
            if 'contain_region' in converted_pred or '_region' in converted_pred:
                # Try to find the target object in the compound reference
                for libero_name, autogpt_name in object_mapping.items():
                    base_obj = re.sub(r'_\d+$', '', libero_name)
                    # Look for patterns where the object name is embedded in the region reference
                    pattern = rf'\b[a-zA-Z_]*{re.escape(base_obj)}[a-zA-Z_]*_region\b'
                    if re.search(pattern, converted_pred):
                        converted_pred = re.sub(pattern, autogpt_name, converted_pred)
                        break
            
            if converted_pred.strip():
                # Remove parentheses for relations format
                clean_pred = converted_pred.strip('()')
                scene_content += f"{clean_pred}\n"
        
        # Add default actor locations - use table:0 if it exists, otherwise floor:0
        primary_location = "table:0" if "table:0" in [autogpt_name for _, autogpt_name in object_mapping.items()] else "floor:0"
        scene_content += f"at robot:0 {primary_location}\n"
        scene_content += f"at human:0 {primary_location}\n"
        
        # Add default states for openable/closeable objects (only if not already specified)
        openable_types = ['wooden_cabinet', 'microwave']
        
        # Track which objects already have opened/closed states
        existing_states = set()
        for init_pred in parsed_data['init_state']:
            if 'open' in init_pred.lower() or 'close' in init_pred.lower():
                # Extract object name from the predicate
                words = init_pred.split()
                if len(words) >= 2:
                    obj_name = words[1]
                    # Convert to AutoGPT+P format
                    for libero_name, autogpt_name in object_mapping.items():
                        if libero_name in obj_name:
                            existing_states.add(autogpt_name)
        
        for libero_name, autogpt_name in object_mapping.items():
            # Extract the base type from the LIBERO object name
            base_type = libero_name.split('_')[0] + '_' + libero_name.split('_')[1] if len(libero_name.split('_')) > 1 else libero_name
            if any(openable in base_type for openable in openable_types) and autogpt_name not in existing_states:
                scene_content += f"closed {autogpt_name}\n"
        
        scene_content += "END-RELATIONS\n"
        
        # Add locations section - group objects by their location
        scene_content += "LOCATIONS\n"
        
        # Use table0 as the location name for all scenes to match the goal format
        location_name = "table0"
        
        # Create a main location with all objects - use table if available, otherwise floor
        has_table = "table:0" in [autogpt_name for _, autogpt_name in object_mapping.items()]
        if has_table:
            location_objects = ["table:0"]  # Include the table itself first
            for obj_name, autogpt_name in object_mapping.items():
                if "table" not in autogpt_name:  # Don't duplicate the table
                    location_objects.append(autogpt_name)
        else:
            location_objects = ["floor:0"]  # Include the floor itself first
            for obj_name, autogpt_name in object_mapping.items():
                if "floor" not in autogpt_name:  # Don't duplicate the floor
                    location_objects.append(autogpt_name)
        
        scene_content += f"{location_name} {' '.join(location_objects)}\n"
        scene_content += "END-LOCATIONS\n"
        
        # Store object mapping for goal conversion
        parsed_data['object_mapping'] = object_mapping
        
        return scene_content
    
    def estimate_plan_cost(self, goal: str, objects_count: int) -> int:
        """Estimate optimal plan cost based on goal complexity"""
        # Simple heuristic based on goal structure
        goal_lower = goal.lower()
        
        # Count predicates in goal
        predicate_count = len(re.findall(r'\([^)]+\)', goal))
        
        # Base cost per predicate
        base_cost = predicate_count * 2
        
        # Add complexity factors
        if "and" in goal_lower:
            base_cost += 2  # Multiple objectives
        if "turnon" in goal_lower or "opened" in goal_lower:
            base_cost += 1  # Action complexity
        if objects_count > 5:
            base_cost += 1  # Scene complexity
        
        return max(base_cost, 3)  # Minimum cost of 3
    
    def convert_suite_to_csv(self, suite_name: str) -> str:
        """Convert a LIBERO suite to CSV format"""
        suite_dir = os.path.join(self.libero_bddl_dir, suite_name)
        if not os.path.exists(suite_dir):
            print(f"Warning: Suite directory {suite_dir} does not exist")
            return ""
        
        csv_file = os.path.join(self.output_scenarios_dir, f"{suite_name}.csv")
        csv_rows = []
        
        # Process all BDDL files in the suite
        for bddl_file in os.listdir(suite_dir):
            if bddl_file.endswith('.bddl'):
                bddl_path = os.path.join(suite_dir, bddl_file)
                
                try:
                    # Parse BDDL file
                    parsed = self.parse_bddl_file(bddl_path)
                    
                    # Generate scene file
                    scene_content = self.generate_scene_file(parsed)
                    scene_filename = f"{suite_name}_{parsed['filename']}.txt"
                    scene_path = os.path.join(self.output_scenes_dir, scene_filename)
                    
                    # Write scene file
                    with open(scene_path, 'w') as f:
                        f.write(scene_content)
                    
                    # Convert goal to PDDL format with proper object mapping
                    converted_goal = self.convert_libero_predicates(parsed['goal'])
                    
                    # Apply object name mapping to goal if available
                    if 'object_mapping' in parsed:
                        for libero_name, autogpt_name in parsed['object_mapping'].items():
                            converted_goal = converted_goal.replace(libero_name, autogpt_name)
                    
                    # Fix common goal formatting issues
                    converted_goal = self._fix_goal_format(converted_goal)
                    
                    # Estimate plan cost
                    objects_count = len(parsed['objects']) + len(parsed['fixtures'])
                    min_cost = self.estimate_plan_cost(parsed['goal'], objects_count)
                    
                    # Add to CSV data
                    csv_rows.append({
                        'task': parsed['language'],
                        'scene_file': scene_filename,
                        'desired_goal': converted_goal,
                        'min_costs': min_cost
                    })
                    
                    print(f"Converted: {bddl_file}")
                    
                except Exception as e:
                    print(f"Error processing {bddl_file}: {e}")
        
        # Write CSV file
        if csv_rows:
            with open(csv_file, 'w', newline='') as csvfile:
                fieldnames = ['task', 'scene_file', 'desired_goal', 'min_costs']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            print(f"Generated CSV: {csv_file} with {len(csv_rows)} tasks")
        
        return csv_file
    
    def convert_all_suites(self):
        """Convert all LIBERO suites to AutoGPT+P format"""
        suites = ['libero_goal', 'libero_10', 'libero_90', 'libero_spatial', 'libero_object']
        
        for suite in suites:
            print(f"\nConverting {suite}...")
            self.convert_suite_to_csv(suite)