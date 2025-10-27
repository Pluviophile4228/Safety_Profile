# Safety_Profile

## 1. Input: case_test.json
- This is the controlled input for the prototype validation. It is a manually distilled JSON file that accurately and formally describes a complex SysML architecture model (based on the eCRM-001 system).
- It includes all logical information required to drive automated analysis:
  - Components(U): All atomic components along with their ID, failureRate and operationtime.
  - Patterns(Π): All safety pattern instances (e.g., StandbyRedundancy, RuntimeAssurance), along with their contained components and failure logic templates (ftaTemplate).
  - Connections(C): A list of connections defining functional dependency relationships between high-level components (patterns and free-standing components).
  - Analysis Target: Defines the top event of the analysis (analysisTopEventName).
## 2. Processing: Analysis-by-Construction.py
- This is the core algorithm engine that implements methodology.
- The script reads the case_test.json file and then executes the process strictly in accordance with the five-step automated workflow defined in paper:
  1. Global Element Indexing: Reads components and constructs D_ac and U.
  2. Pattern Instantiation and Component Classification: Parses patterns to generate L_logic, Π, and S.
  3. Functional Dependency Topology Sorting: Parses connections (C) to construct G_dep (dependency graph).
  4. Hierarchical Fault Tree Construction: Recursively "assembles" L_logic and G_dep into an in-memory FtaNode tree (N_root).
  5. Quantitative Analysis and Export: Runs the MOCUS algorithm on N_root to calculate cut sets and computes the top event probability.
## 3. Outputs: ads_fta_output.xml and ads_cutsets_report.txt
- ads_fta_output.xml (Machine-Readable Model)
  - This is a standardized analysis model exported in Step 5.
  - It is an auditable and verifiable file that strictly complies with OMG’s RAAML Standard. It does not contain calculation "results" but includes the complete fault tree structure and input parameters (failure rates). This ensures transparency of the analysis process and allows other third-party professional tools (e.g., Cameo, Isograph) to import the model for independent validation.
- ads_cutsets_report.txt (Human-Readable Results)
  - This is the final analysis result exported in Step 5.
  - It provides engineers with immediate and decision-usable safety insights. It includes two key metrics:
    1. Sorted by order, clearly exposing all single-point faults (1st-order cut sets) and high-risk failure combinations in the system.
    2. The final quantitative risk value, which can be used for comparison with safety targets (φ).
