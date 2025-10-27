import json
import string
import traceback

# ===========================================================================
# Step 1: Global Element Indexing (Algorithm 1)
# ===========================================================================
def index_global_elements(model_data):
    """Traverse model data to index all atomic components along with their failure rates and operating times."""
    D_ac = {}
    U = set()
    print("\n--- Step 1: Starting to index global atomic components ---")

    if not isinstance(model_data, dict) or 'components' not in model_data:
        print("Error: Invalid model data format or missing 'components' list.")
        return D_ac, U
    if not isinstance(model_data['components'], list):
        print("Error: 'components' field must be a list.")
        return D_ac, U

    for component in model_data['components']:
        if not isinstance(component, dict):
            print(f"Warning: Found invalid component entry (not a dictionary), skipped: {component}")
            continue

        comp_id = component.get('id')
        comp_name = component.get('name', f"Unnamed_{comp_id}" if comp_id else "Unknown")

        if not comp_id or not isinstance(comp_id, str):
            print(f"Warning: Found component with invalid or missing ID, skipped: {component}")
            continue
        if comp_id in D_ac:
            print(f"Warning: Found duplicate component ID '{comp_id}', will overwrite the original entry.")

        # Handling FailureRate
        rate = component.get('failureRate')
        if isinstance(rate, str) and rate.strip() == "": rate = None
        try:
            if rate is not None: rate = float(rate)
        except (ValueError, TypeError):
            print(f"Warning: Invalid failureRate '{rate}' for component '{comp_id}', setting to None.")
            rate = None

        # (Priority 2: logic such as external references is omitted here, and can be added as needed)
        if rate is None:
            print(f"Warning: Failure rate for component '{comp_id}' is not defined.")

        # --- Added handling for 'Time' ---
        time = component.get('Time')
        if isinstance(time, str) and time.strip() == "": time = None

        try:
            if time is not None:
                time = float(time)
            else:
                # If 'Time' is not provided in JSON, provide a default value
                print(f"Info: Component '{comp_id}' does not provide 'Time', defaulting to 1.0 (unit time).")
                time = 1.0
        except (ValueError, TypeError):
            print(f"Warning: Invalid operating time (Time) '{time}' for component '{comp_id}', setting to None.")
            time = None  # Setting to None will cause subsequent probability calculations to skip this component

        D_ac[comp_id] = {'Name': comp_name, 'Rate': rate, 'Time': time}  # <-- (Modified)
        U.add(comp_id)
        # print(f"  Indexed: ID='{comp_id}', Rate={rate}, Time={time}")

    print(f"--- Step 1: Completed. Indexed {len(U)} atomic components. ---")
    return D_ac, U


# ===========================================================================
# Step 2: Pattern Instantiation and Component Categorization (Algorithm 2)
# ===========================================================================
def instantiate_patterns_and_categorize(model_data):
    """Identify pattern instances, generate logic strings, and categorize components."""
    L_logic = []  # Will store tuples of (pattern_id, instantiated_string)
    Pi = set()    # Stores IDs of pattern instances
    S = set()     # Stores IDs of components within patterns
    print("\n--- Step 2: Starting to instantiate safety patterns and categorize components ---")

    if not isinstance(model_data, dict) or 'patterns' not in model_data:
        print("Error: Invalid model data format or missing 'patterns' list.")
        return L_logic, Pi, S
    if not isinstance(model_data['patterns'], list):
        print("Error: 'patterns' field must be a list.")
        return L_logic, Pi, S

    for pattern_instance in model_data['patterns']:
        if not isinstance(pattern_instance, dict):
            print(f"Warning: Found invalid pattern entry (not a dictionary), skipped: {pattern_instance}")
            continue

        pattern_id = pattern_instance.get('id')
        pattern_type = pattern_instance.get('type')
        contained_components_map = pattern_instance.get('contained_components')
        fta_template = pattern_instance.get('ftaTemplate')

        # --- Data validation ---
        if not pattern_id or not isinstance(pattern_id, str):
            print(f"Warning: Found pattern instance with invalid or missing ID, skipped: {pattern_instance}")
            continue
        if pattern_id in Pi:
            print(f"Warning: Found duplicate pattern instance ID '{pattern_id}'. Ensure instance IDs are unique.")
        if not pattern_type or not isinstance(pattern_type, str):
            print(f"Warning: Pattern instance '{pattern_id}' lacks a valid 'type', skipped.")
            continue
        if not contained_components_map or not isinstance(contained_components_map, dict):
            print(f"Warning: Pattern instance '{pattern_id}' lacks a valid 'contained_components' dictionary, skipped.")
            continue
        if not fta_template or not isinstance(fta_template, str):
            print(f"Warning: Pattern instance '{pattern_id}' (type '{pattern_type}') lacks 'ftaTemplate', cannot generate logic.")
            # Still record the pattern and components but don't generate logic
            Pi.add(pattern_id)
            component_ids_in_pattern = set(contained_components_map.values())
            S.update(component_ids_in_pattern)
            continue  # Skip remaining part of current loop
        # --- End of validation ---

        Pi.add(pattern_id)
        # print(f"Processing pattern instance: ID='{pattern_id}', Type='{pattern_type}'")  # Optional debug output

        # Determine G(pi_i), i.e., the set of component IDs contained in the pattern
        component_ids_in_pattern = set(contained_components_map.values())
        S.update(component_ids_in_pattern)  # Merge these IDs into set S
        # print(f"Contained component IDs: {component_ids_in_pattern}")  # Optional debug output

        # Instantiate the template
        try:
            # Use .format(**dict) for replacement
            instantiated_string = fta_template.format(** contained_components_map)
            L_logic.append((pattern_id, instantiated_string))  # Store as (id, logic_string) tuple
            # print(f"Generated logic for '{pattern_id}': {instantiated_string}")  # Optional debug output

        except KeyError as e:
            print(f"Error: Failed to instantiate template for pattern '{pattern_id}': Template requires placeholder '{e}', "
                  f"but no corresponding key was found in the 'contained_components' dictionary.")
        except Exception as e:  # Catch other potential formatting errors
            print(f"Error: Unknown error occurred while instantiating template for pattern '{pattern_id}': {e}")

    print(f"--- Step 2: Completed. Processed {len(Pi)} pattern instances. Total number of component IDs in patterns: {len(S)}. Generated {len(L_logic)} logic entries. ---")
    return L_logic, Pi, S

# ===========================================================================
# Step3：Functional Dependency Topology Analysis (Algorithm 3 Revised)
# ===========================================================================

# --- Helper Function  ---
memo_owner = {}
def find_relevant_owner_id(element_id, model, patterns_map, analysis_nodes_set):
    """Find the high-level component ID (pattern instance or free component) that the element belongs to based on its ID"""
    if element_id in memo_owner:
        return memo_owner[element_id]

    # 1. Check if element_id itself is an analysis node
    if element_id in analysis_nodes_set:
        memo_owner[element_id] = element_id
        return element_id

    # 2. Check if element_id is a component within a pattern instance
    for pattern_id, pattern_data in patterns_map.items():
        contained = pattern_data.get('contained_components', {})
        if isinstance(contained, dict):
            if element_id in contained.values():
                if pattern_id in analysis_nodes_set:
                    memo_owner[element_id] = pattern_id
                    return pattern_id
                else:
                    # This case should ideally not happen if Pi is correctly passed
                    # print(f"Debug warning: Component {element_id} is within pattern {pattern_id}, but the pattern is not in AnalysisNodes {analysis_nodes_set}?")
                    break  # Stop searching this pattern
        elif isinstance(contained, list):  # Less ideal, assumes simple list
            if element_id in contained:
                if pattern_id in analysis_nodes_set:
                    memo_owner[element_id] = pattern_id
                    return pattern_id
                else:
                    # print(f"Debug warning: Component {element_id} is within pattern {pattern_id}, but the pattern is not in AnalysisNodes?")
                    break

    # 3. If not found above, it's not relevant for high-level topology
    memo_owner[element_id] = None
    return None


memo_upstream = {}
# def find_upstream_analysis_node_ids(component_id, model, patterns_map, analysis_nodes_set):
#     """Find the upstream 'analysis node' ID connected to component_id"""
#     owner_id = find_relevant_owner_id(component_id, model, patterns_map, analysis_nodes_set)
#     if not owner_id:
#         return []
#     if owner_id in memo_upstream:
#          return memo_upstream[owner_id]
#
#     upstream_owners = set()
#     for conn in model.get('connections', []):
#         target_conn_id = conn.get('target')
#         source_conn_id = conn.get('source')
#
#         target_owner_id = find_relevant_owner_id(target_conn_id, model, patterns_map, analysis_nodes_set)
#
#         if target_owner_id == owner_id:
#             source_owner_id = find_relevant_owner_id(source_conn_id, model, patterns_map, analysis_nodes_set)
#             if source_owner_id and source_owner_id != owner_id:
#                 upstream_owners.add(source_owner_id)
#
#     result = list(upstream_owners)
#     memo_upstream[owner_id] = result
#     return result
# --- End Helper Functions ---

def analyze_dependency_topology(model_data, Pi_instance_ids, U, S):
    """Analyze functional dependency relationships between all pattern instances and unlinked components."""
    print("\n--- Step 3: Starting to organize functional dependency topology ---")
    memo_owner.clear()

    U_unlinked_ids = U - S
    print(f"Identified unlinked component IDs (U-S): {U_unlinked_ids}")
    AnalysisNodes_ids = Pi_instance_ids | U_unlinked_ids
    print(f"Topology analysis subject IDs (Pi | U_unlinked): {AnalysisNodes_ids}")

    G_dep = {}  # {downstream ID: [upstream ID list]}
    patterns_map_by_id = {p.get('id'): p for p in model_data.get('patterns', []) if p.get('id')}

    print("Traversing all connections to build G_dep...")
    for conn in model_data.get('connections', []):
        target_conn_id = conn.get('target')
        source_conn_id = conn.get('source')

        # Find the analysis subject that each end of the connection belongs to
        target_owner_id = find_relevant_owner_id(target_conn_id, model_data, patterns_map_by_id, AnalysisNodes_ids)
        source_owner_id = find_relevant_owner_id(source_conn_id, model_data, patterns_map_by_id, AnalysisNodes_ids)

        # If both ends of the connection belong to our concerned analysis subjects and are not self-connected
        if target_owner_id and source_owner_id and target_owner_id != source_owner_id:
            # This is a valid dependency relationship: source -> target
            if target_owner_id not in G_dep:
                G_dep[target_owner_id] = []

            if source_owner_id not in G_dep[target_owner_id]:
                G_dep[target_owner_id].append(source_owner_id)
                # print(f"Adding dependency to G_dep: {source_owner_id} -> {target_owner_id}")

        # (Note: This method assumes that 'connections' in JSON do not contain feedback loops)


    print(f"--- Step 3: Completed. Constructed functional dependency graph G_dep: {G_dep} ---")
    return G_dep

# ===========================================================================
# Step4：Dependency-Based FTA Assembly (Algorithm 4)
# ===========================================================================

# --- Helper Function ---
memo_parse = {} # Cache for parsed logic strings

def split_arguments(content_string):
    """Used to split parameters similar to 'arg1, OR (arg2, arg3), arg4'."""
    args = []
    balance = 0
    current_arg = ""
    for char in content_string:
        if char == ',' and balance == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
    args.append(current_arg.strip()) # Add the last argument
    return [arg for arg in args if arg] # Filter out empty strings


def parse_logic_string(logic_string, D_ac):
    """Recursively parse instantiated FTA logic string into FtaNode object."""
    logic_string = logic_string.strip()
    if not logic_string:
        return None
    if logic_string in memo_parse:  # Cache parsed gates/structures too
        return memo_parse[logic_string]

    node = None
    original_string_repr = logic_string[:40] + ('...' if len(logic_string) > 40 else '')  # For logging

    try:
        if logic_string.upper().startswith("OR("):
            content = logic_string[3:-1]
            node = FtaNode(f"OR_{hash(content)}", "OR", name=f"OR: {original_string_repr}")
            args = split_arguments(content)
            # print(f"Parsing OR: content='{content}', args={args}") # Optional debug
            for arg in args:
                child_node = parse_logic_string(arg, D_ac)
                node.add_child(child_node)

        elif logic_string.upper().startswith("AND("):
            content = logic_string[4:-1]
            node = FtaNode(f"AND_{hash(content)}", "AND", name=f"AND: {original_string_repr}")
            args = split_arguments(content)
            # print(f"Parsing AND: content='{content}', args={args}") # Optional debug
            for arg in args:
                child_node = parse_logic_string(arg, D_ac)
                node.add_child(child_node)

        elif logic_string.upper().startswith("BE:"):
            be_id = logic_string[3:]
            if be_id in D_ac:
                comp_info = D_ac[be_id]
                node = FtaNode(be_id, "BE", name=comp_info.get('Name', be_id), rate=comp_info.get('Rate'))
                # print(f"Parsing BE: id='{be_id}', rate={node.rate}") # Optional debug
            else:
                print(f"Error: Cannot find information for basic event ID '{be_id}' during parsing. Will create an unknown node.")
                node = FtaNode(be_id, "BE", name=f"Unknown_{be_id}", rate=None)

        else:
            print(f"Error: Unable to parse logic string fragment: '{logic_string}'")
            node = FtaNode(f"PARSE_ERROR_{hash(logic_string)}", "BE", name=f"Parse Error: {original_string_repr}", rate=0.0)  # Dummy node

    except Exception as e:
        print(f"Critical error occurred while parsing '{logic_string}': {e}")
        node = FtaNode(f"PARSE_EXCEPTION_{hash(logic_string)}", "BE", name=f"Parse Exception: {original_string_repr}", rate=1.0)  # Dummy critical node

    if node:
        memo_parse[logic_string] = node
    return node

memo_build = {}  # Cache for built subtrees


def build_subtree_recursive(current_id, G_dep, L_logic_map, D_ac, Pi_instance_ids):
    """
    Recursively build the fault tree for the failure of 'current_id'.
    Failure = Internal Failure OR Upstream Failure
    """
    if current_id in memo_build:
        return memo_build[current_id]

    # print(f"Recursively building subtree for: {current_id}")

    # 1. Get internal failure
    N_internal = None
    if current_id in Pi_instance_ids:  # Pattern instance
        logic_string = L_logic_map.get(current_id)
        if logic_string:
            N_internal = parse_logic_string(logic_string, D_ac)
        else:
            print(f"Error: Cannot find logic for pattern {current_id}.")
            N_internal = FtaNode(f"{current_id}_Internal_MissingLogic", "BE", name="Internal Logic Missing", rate=1.0)
    elif current_id in D_ac:  # Unlinked component (actuator in our example)
        comp_info = D_ac[current_id]
        N_internal = FtaNode(current_id, "BE", name=comp_info.get('Name', current_id), rate=comp_info.get('Rate'))
    else:
        print(f"Error: Cannot find information for ID '{current_id}' during construction.")
        return None  # Cannot build

    # 2. Get upstream failure
    upstream_ids = G_dep.get(current_id, [])

    # 3. Combine logic
    if not upstream_ids:
        # If there are no upstream dependencies (sensors in our example), total failure = internal failure
        print(f"Node {current_id} has no upstream dependencies, returning internal failure tree.")
        memo_build[current_id] = N_internal
        return N_internal
    else:
        # If there are upstream dependencies, total failure = OR(Internal Failure, Upstream Failure)
        N_component_failure = FtaNode(f"{current_id}_Failure", "OR", name=f"{current_id} Failure")
        N_component_failure.add_child(N_internal)

        # --- (Key !! ) ---
        # The logic for upstream failure depends on system design.
        # For redundant inputs (e.g., fcc1/fcc2 -> L_RoCon1), the failure logic should use an AND gate.
        # For single inputs (e.g., L_RoCon1 -> LU_Motor), the failure logic uses an OR gate (or direct link).
        # To simplify the prototype, we temporarily assume any upstream failure causes upstream failure (OR gate).
        # (A more advanced implementation would check the "port type" of connections in G_dep to determine AND/OR in the future.)

        print(f"Processing upstream dependencies for {current_id}: {upstream_ids}")
        N_upstream_failure = FtaNode(f"{current_id}_Upstream_Failure", "OR", name="Upstream Dependency Failure")
        for up_id in upstream_ids:
            N_upstream_subtree = build_subtree_recursive(up_id, G_dep, L_logic_map, D_ac, Pi_instance_ids)
            N_upstream_failure.add_child(N_upstream_subtree)
        N_component_failure.add_child(N_upstream_failure)

        memo_build[current_id] = N_component_failure
        return N_component_failure

import traceback  # Ensure traceback is imported for exception printing

def parse_system_level_logic(logic_string, G_dep, L_logic_map, D_ac, Pi_instance_ids):
    """ Recursively parse high-level system failure logic (supports FAIL: syntax). It calls build_subtree_recursive to construct subtrees. """
    logic_string = logic_string.strip()
    if not logic_string:
        return None

    node = None

    try:
        if logic_string.upper().startswith("OR("):
            content = logic_string[3:-1]
            node = FtaNode(f"SYS_OR_{hash(content)}", "OR", name=f"System OR: {content[:50]}...")
            args = split_arguments(content)  # Reuse the existing split_arguments function
            for arg in args:
                child_node = parse_system_level_logic(arg, G_dep, L_logic_map, D_ac, Pi_instance_ids)
                node.add_child(child_node)

        elif logic_string.upper().startswith("AND("):
            content = logic_string[4:-1]
            node = FtaNode(f"SYS_AND_{hash(content)}", "AND", name=f"System AND: {content[:50]}...")
            args = split_arguments(content)
            for arg in args:
                child_node = parse_system_level_logic(arg, G_dep, L_logic_map, D_ac, Pi_instance_ids)
                node.add_child(child_node)

        elif logic_string.upper().startswith("FAIL:"):
            comp_id = logic_string[5:].strip()
            if not comp_id:
                raise ValueError("Component ID missing after FAIL: syntax")
            # print(f"Building complete failure subtree for component '{comp_id}'...")
            # Key: Call the *original* recursive builder from Step 4
            node = build_subtree_recursive(comp_id, G_dep, L_logic_map, D_ac, Pi_instance_ids)

        else:
            # (Optional) You can also support BE: here if the top event needs to reference a basic event directly
            print(f"Error: Unable to parse system top event logic fragment: '{logic_string}'")
            node = FtaNode(f"SYS_PARSE_ERROR_{hash(logic_string)}", "BE", name=f"Parse Error: {logic_string}", rate=1.0)

    except Exception as e:
        print(f"Critical error occurred while parsing system logic '{logic_string}': {e}")
        traceback.print_exc()
        node = FtaNode(f"SYS_PARSE_EXCEPTION_{hash(logic_string)}", "BE", name=f"Parse Exception: {logic_string}", rate=1.0)

    return node

def assemble_fta_hierarchically(model_data, G_dep, L_logic_tuples, D_ac, Pi_instance_ids, U, S):
    """Build the fault tree based on 'analysisTopEventLogic' explicitly defined in the JSON."""
    print("\n--- Step 4: Starting to build hierarchical fault tree based on dependencies ---")
    FtaNode.clear_be_memo()
    memo_build.clear()
    memo_parse.clear()

    L_logic_map = dict(L_logic_tuples)  # Convert to dictionary {pattern_id: logic_string}

    # 1. Retrieve top event logic string from the model
    top_event_logic = model_data.get("analysisTopEventLogic")
    top_event_name = model_data.get("analysisTopEventName", "Top System Failure")

    if not top_event_logic:
        print("Error: 'analysisTopEventLogic' is not defined in the JSON model. Cannot build the fault tree.")
        print("Please add the following to the JSON, for example: \"analysisTopEventLogic\": \"OR(FAIL:component_id_1, FAIL:component_id_2)\"")
        return None

    print(f"Analysis Target: Build based on the top event logic defined in the JSON.")
    print(f"Top Event Name: '{top_event_name}'")
    print(f"Top Event Logic: '{top_event_logic}'")

    # 2. Call the new system-level parser
    N_root = parse_system_level_logic(
        top_event_logic,
        G_dep,
        L_logic_map,
        D_ac,
        Pi_instance_ids
    )

    # 3. Check the result
    if not N_root:
        print("--- Step 4: Failed. Unable to build a valid fault tree from the top event logic. ---")
        return None

    # Set the root node's name and ID to values defined in the JSON (Optional)
    N_root.name = top_event_name
    N_root.id = top_event_name.replace(" ", "_")

    print(f"--- Step 4: Completed. Global fault tree root node: {N_root} ---")
    return N_root

def print_fta_tree(node, indent=""):
    """Recursively print the structure of the FtaNode tree for debugging purposes."""
    if not node:
        print(indent + "Tree is empty (None)")
        return

    rate_info = f" (Rate: {node.rate})" if node.type == 'BE' and node.rate is not None else ""
    print(f"{indent}+ [{node.type}] {node.name} (ID: {node.id}){rate_info}")

    # Recursively print child nodes
    for child in node.children:
        print_fta_tree(child, indent + "  |")


# ===========================================================================
# Step5：Quantitative Analysis and Artifact Export (Algorithm 5)
# ===========================================================================

# --- Helper Function ---

def minimize_cutsets(cutsets):
    """Minimization (absorption law). For example, if {A} and {A, B} exist, {A, B} will be removed."""
    # Sort cut sets by size, prioritizing smaller sets for processing
    sorted_sets = sorted(list(cutsets), key=len)
    minimized = set()

    for s_new in sorted_sets:
        # Check if s_new is "subsumed" by any existing set in minimized
        is_subsumed = False
        for s_existing in minimized:
            if s_existing.issubset(s_new):  # s_existing is a subset of s_new
                is_subsumed = True
                break

        # If s_new is not subsumed by any existing set, it is minimal
        if not is_subsumed:
            # Conversely, it might "subsume" larger existing sets in minimized (low probability after sorting, but checked for rigor)
            # This step can be simplified; only check during addition, as MOCUS AND-gate logic ensures this
            minimized.add(s_new)

    return minimized


def get_cutsets_recursive(node):
    """Core recursive function for the MOCUS algorithm."""
    global memo_cutsets_cache  # Use global cache

    # 1. Check cache
    if node.id in memo_cutsets_cache:
        return memo_cutsets_cache[node.id]

    result = set()  # Initialize empty set of cut sets

    # 2. Rule 1: Basic Event
    if node.type == 'BE':
        # The cut set of a basic event is the event itself
        result = {frozenset({node.id})}  # frozenset is immutable and can be stored in a set

    # 3. Rule 2: OR Gate
    elif node.type == 'OR':
        # The cut sets of an OR gate are the "union" of all child node cut sets
        for child in node.children:
            child_cutsets = get_cutsets_recursive(child)
            result.update(child_cutsets)  # set.update() performs union
        # Perform minimization after processing OR gate to handle logic like A + (A*B)
        result = minimize_cutsets(result)

    # 4. Rule 3: AND Gate
    elif node.type == 'AND':
        if not node.children:
            result = {frozenset()}  # Empty AND gate? Logically "true", cut set is empty set
        else:
            # The cut sets of an AND gate are the "Cartesian product" of all child node cut sets

            # Start with the first child
            result = get_cutsets_recursive(node.children[0])

            # Iteratively compute Cartesian product with subsequent children
            for child in node.children[1:]:
                child_cutsets = get_cutsets_recursive(child)

                # If any child's cut sets are empty (e.g., child never fails), AND gate result is also empty
                if not child_cutsets:
                    result = set()
                    break

                # Compute Cartesian product
                new_result = set()
                for s1 in result:
                    for s2 in child_cutsets:
                        new_result.add(s1.union(s2))  # Combine cut sets

                result = new_result  # Update result to the new Cartesian product

                # Key: Perform minimization after each AND product step to prevent combinatorial explosion
                result = minimize_cutsets(result)

    else:  # Unknown node type
        print(f"Warning: Unknown node type '{node.type}' (ID: {node.id}), skipping it when calculating cut sets.")
        result = set()  # Return empty set

    # 5. Store in cache and return
    memo_cutsets_cache[node.id] = result
    return result


def _write_cutsets_to_file(cutsets, D_ac, filepath):
    """
    Write the calculated list of minimal cut sets to a text file in a human-readable format.
    Cut sets are sorted by size (order) and include component names.
    """
    print(f"Writing minimal cut sets to: {filepath} ...")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Minimal Cutsets Analysis Results\n")
            f.write(f"Total Minimal Cutsets: {len(cutsets)}\n")
            f.write("=" * 40 + "\n")

            # Convert frozensets to regular sets for sorting and processing
            list_of_sets = [set(cs) for cs in cutsets]
            # Sort cut sets by size (order)
            sorted_cutsets = sorted(list_of_sets, key=len)

            current_order = 0
            for cs in sorted_cutsets:
                order = len(cs)
                if order != current_order:
                    current_order = order
                    f.write(f"\n--- Minimal Cutsets of Order {order} ---\n")

                # Build human-readable component name string
                cs_components = []
                for be_id in sorted(list(cs)):  # Also sort inside the cut set for consistency
                    name = D_ac.get(be_id, {}).get('Name', 'UnknownName')
                    cs_components.append(f"{name} (ID: {be_id})")

                f.write(f" - {{ {', '.join(cs_components)} }}\n")

        print(f"Minimal cut sets successfully written.")
    except Exception as e:
        print(f"Error: Failed to write minimal cut sets to file '{filepath}': {e}")

def calculate_minimal_cut_sets(fta_root_node):
    """Calculate the minimal cut sets of the fault tree (Basic MOCUS Implementation)."""
    global memo_cutsets_cache
    memo_cutsets_cache = {}  # Clear cache before each call

    print("Calculating minimal cut sets (Basic MOCUS Version)...")
    if not fta_root_node:
        print("Error: Fault tree root node is empty, calculation not possible.")
        return set()

    final_cutsets = get_cutsets_recursive(fta_root_node)

    print(f"Calculation completed. Found {len(final_cutsets)} minimal cut sets in total.")
    # Print a few sample cut sets for verification
    if final_cutsets:
        print("Minimal Cut Set Samples (up to 5 shown):")
        for i, cs in enumerate(final_cutsets):
            if i >= 5:
                print("...")
                break
            print(f"-{cs}")

    return final_cutsets


def calculate_top_event_probability(cutsets, D_ac):
    """Calculate top event probability based on minimal cut sets (rare event approximation)."""
    print("Calculating top event probability...")
    top_prob = 0.0

    if not cutsets:
        print("No minimal cut sets found, top event probability is 0.0.")
        return 0.0

    for cs in cutsets:
        mcs_prob = 1.0
        valid_mcs = True

        for be_id in cs:
            # Retrieve complete component information from D_ac
            comp_info = D_ac.get(be_id, {})
            rate = comp_info.get('Rate')
            time = comp_info.get('Time')  # (Defaults to 1.0)

            # Check if both Rate and Time are valid
            if rate is None or time is None:
                print(f"Warning: Cannot calculate probability for cut set {cs} because the failure rate or time of component {be_id} is unknown. Skipping this cut set.")
                valid_mcs = False
                break

            # Calculate basic event probability P_be = Rate * Time
            P_be = rate * time

            # Assume failure rate ≈ probability P (applies to low-probability events and unit time)
            mcs_prob *= P_be

        if valid_mcs:
            top_prob += mcs_prob  # Rare event approximation: direct summation

    print(f"Calculated top event probability (P ≈ Σ(Π(λt))) ≈ {top_prob:.6e}")
    return top_prob

def export_to_raaml_file_content(fta_root_node, D_ac):
    """Export the in-memory FTA tree to an XML string."""
    print("Exporting to XML...")
    xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
    # Example namespace; you may need to adjust based on the actual RAAML version
    xml_parts.append('<raaml:FTAModel xmlns:raaml="http://www.omg.org/spec/RAAML/20230101/RAAML">')

    def build_xml_recursive(node, indent_level=1):
        indent = "  " * indent_level
        parts = []
        node_id_attr = f'id="{node.id}"' if node.id else ""
        node_name_attr = f'name="{node.name}"' if node.name else ""

        if node.type == 'OR' or node.type == 'AND':
            parts.append(f'{indent}<Gate type="{node.type}" {node_id_attr} {node_name_attr}>')
            for child in node.children:
                parts.append(build_xml_recursive(child, indent_level + 1))
            parts.append(f'{indent}</Gate>')
        elif node.type == 'BE':
            rate_attr = f'probability="{node.rate}"' if node.rate is not None else ""
            parts.append(f'{indent}<BasicEvent {node_id_attr} {node_name_attr} {rate_attr}/>')
        # Add handling for other node types if needed (e.g., UndevelopedEvent, HouseEvent)
        return "\n".join(parts)

    if fta_root_node:
        xml_parts.append(build_xml_recursive(fta_root_node, 1))

    xml_parts.append('</raaml:FTAModel>')
    print("XML content generated successfully.")
    return "\n".join(xml_parts)


def analyze_and_export(fta_root_node, D_ac, cutsets_filepath):
    """Count the number of first-order and second-order cut sets."""
    print("\n--- Step 5: Starting quantitative analysis and result export ---")
    if not fta_root_node:
        print("  Error: The input fault tree is empty, analysis cannot be performed.")
        return {'Cutsets': [], 'Probability': 0.0, 'FirstOrderCutsetCount': 0, 'SecondOrderCutsetCount': 0}, ""

    cutsets = calculate_minimal_cut_sets(fta_root_node)
    top_prob = calculate_top_event_probability(cutsets, D_ac)

    # Count the number of first-order and second-order cut sets
    first_order_cutset_count = 0
    second_order_cutset_count = 0
    for cs in cutsets:
        if len(cs) == 1:
            first_order_cutset_count += 1
        elif len(cs) == 2:
            second_order_cutset_count += 1

    print(f"Statistics: Found a total of {first_order_cutset_count} first-order cut sets (single-point failures).")
    print(f"Statistics: Found a total of {second_order_cutset_count} second-order cut sets.")

    R_analysis = {
        'Cutsets': cutsets,
        'Probability': top_prob,
        'FirstOrderCutsetCount': first_order_cutset_count,
        'SecondOrderCutsetCount': second_order_cutset_count
    }

    F_raaml_content = export_to_raaml_file_content(fta_root_node, D_ac)
    _write_cutsets_to_file(cutsets, D_ac, cutsets_filepath)

    print("--- Step 5: Completed. ---")
    return R_analysis, F_raaml_content

def load_model_data(filepath):
    """Load JSON model data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded model data: {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: Input file '{filepath}' not found. Please ensure the file path and name are correct.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{filepath}': {e}. Please check the JSON format.")
        return None
    except Exception as e:
        print(f"Unknown error occurred while loading the file: {e}")
        return None


def print_fta_tree(node, indent=""):
    """Recursively print the structure of the FtaNode tree for debugging purposes."""
    if not node:
        print(indent + "Tree is empty (None)")
        return

    rate_info = f" (Rate: {node.rate})" if node.type == 'BE' and node.rate is not None else ""
    print(f"{indent}+ [{node.type}] {node.name} (ID: {node.id}){rate_info}")

    # Recursively print child nodes
    for child in node.children:
        print_fta_tree(child, indent + "  |")

class FtaNode:
    _be_memo = {}   # Class level memoization for Basic Events

    def __init__(self, node_id, node_type, name="Unknown", rate=None):
        """ Initialize the node """
        self.id = node_id
        self.type = node_type.upper()  # Ensure type: 'OR', 'AND', 'BE'
        self.name = name
        self.rate = None   # Only valid for BE
        self.children = []

        if self.type == 'BE':
            # --- Handling of common cause events ---
            if node_id in FtaNode._be_memo:
                # If this BE ID already exists, the current object reuses the cached object
                existing_node = FtaNode._be_memo[node_id]
                self.__dict__.update(existing_node.__dict__)  # Key: Make the current instance point to the same memory object
                # Check and update rate (theoretically should be consistent)
                if rate is not None and self.rate is None:
                    self.rate = rate
                elif rate is not None and self.rate is not None and self.rate != rate:
                    print(f"  Warning (FtaNode): Component '{node_id}' has different failure rates defined in different places ({self.rate} vs {rate}). Using the first value.")
            else:
                # First time encountering this BE ID, store in cache
                try:  # Ensure rate is a float or None
                    self.rate = float(rate) if rate is not None else None
                except (ValueError, TypeError):
                    print(f"  Warning (FtaNode): The failure rate '{rate}' for component '{node_id}' is invalid, setting to None.")
                    self.rate = None
                FtaNode._be_memo[self.id] = self
        elif self.type not in ['OR', 'AND']:
            print(f"  Warning (FtaNode): Created a node of unknown type '{self.type}' (ID: {self.id})")

    def add_child(self, child_node):
        """Add child node to gate node (if not None)."""
        if child_node and self.type in ['OR', 'AND']:
            self.children.append(child_node)
        elif self.type not in ['OR', 'AND']:
            print(f"  Warning (FtaNode): Attempted to add child node to non-gate node '{self.id}' (type {self.type}).")

    def __repr__(self):
        """Provide a simple string representation for debugging."""
        child_count = len(self.children)
        rate_info = f", rate={self.rate}" if self.type == 'BE' else ""
        return f"FtaNode(id='{self.id}', type='{self.type}', name='{self.name}'{rate_info}, children={child_count})"

    @classmethod
    def clear_be_memo(cls):
        """Clear the basic event cache (called before each run of assemble_fta)."""
        cls._be_memo = {}

# ===========================================================================
# Mian Function
# ===========================================================================
if __name__ == "__main__":
    json_filepath = 'case_test.json'
    output_xml_filepath = 'ads_fta_output.xml'
    output_cutsets_filepath = 'ads_cutsets_report.txt'

    # Initialize
    model_data = None
    component_dict_D_ac = None
    component_set_U = None
    logic_list_L_logic = None
    pattern_instance_ids_Pi = None
    in_pattern_component_ids_S = None
    dependency_graph_G_dep = None
    fta_tree_root = None
    analysis_results = None
    raaml_output = None
    # (analysis_target variable has been removed)

    try:
        model_data = load_model_data(json_filepath)
        if not model_data:
            raise ValueError("Failed to load model data.")

        print(f"\nAnalysis target: Based on all final actuators (unlinked components) in the (U-S) set.")
        if "analysisTopEventName" not in model_data:
            print("  Warning: 'analysisTopEventName' is missing in the JSON file; default top event name will be used.")

        # --- Execute the five-step process ---
        # Step 1
        component_dict_D_ac, component_set_U = index_global_elements(model_data)
        if not component_dict_D_ac: raise ValueError("Step 1 failed to index components successfully.")

        # Step 2
        logic_list_L_logic, pattern_instance_ids_Pi, in_pattern_component_ids_S = \
            instantiate_patterns_and_categorize(model_data)
        if pattern_instance_ids_Pi is None or in_pattern_component_ids_S is None:
            raise ValueError("Step 2 failed to process patterns effectively.")

        # Step 3
        dependency_graph_G_dep = analyze_dependency_topology(
            model_data,
            pattern_instance_ids_Pi,
            component_set_U,
            in_pattern_component_ids_S
        )
        if dependency_graph_G_dep is None: raise ValueError("Step 3 failed to generate dependency graph.")

        # Step 4
        logic_map = dict(logic_list_L_logic)
        fta_tree_root = assemble_fta_hierarchically(
            model_data,
            dependency_graph_G_dep,
            logic_map,
            component_dict_D_ac,
            pattern_instance_ids_Pi,
            component_set_U,
            in_pattern_component_ids_S
        )
        if not fta_tree_root: raise ValueError("Step 4 failed to generate a valid fault tree.")

        # Output preview
        # print("\n--- Step 4 Output (FTA Tree Structure Preview) ---")
        # print_fta_tree(fta_tree_root)  # (Ensure print_fta_tree function is defined)

        # Step 5
        analysis_results, raaml_output = analyze_and_export(
            fta_tree_root,
            component_dict_D_ac,
            output_cutsets_filepath
        )
        if analysis_results is None or raaml_output is None:
            raise ValueError("Step 5 failed to complete analysis or export.")

        # --- Output and save results ---
        print("\n=== Final Analysis Results ===")
        print(f"Top Event Probability: {analysis_results.get('Probability', 'Calculation failed'):.6e}")
        print(f"Number of Minimal Cutsets: {len(analysis_results.get('Cutsets', []))}")
        print(f"First-Order Cutsets: {analysis_results.get('FirstOrderCutsetCount', 'Calculation failed')}")
        print(f"Second-Order Cutsets: {analysis_results.get('SecondOrderCutsetCount', 'Calculation failed')}")

        with open(output_xml_filepath, 'w', encoding='utf-8') as f:
            f.write(raaml_output)
        print(f"\nRAAML model successfully exported to: {output_xml_filepath}")
        # Printing of cut sets report is handled internally in analyze_and_export

    except FileNotFoundError:
        pass
    except ValueError as ve:
        print(f"\n!!! Execution interrupted: {ve} !!!")
    except Exception as e:
        print(f"\n!!! Unexpected critical error occurred during execution: {e} !!!")
        traceback.print_exc()