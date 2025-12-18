#!/bin/bash
#
# JAX-CTSM Translation Workflow Script
#
# This script runs the complete translation workflow:
# 1. Translate Fortran modules to JAX (translate_with_json.py)
# 2. Generate tests for translated modules (generate_tests.py)
# 3. Run tests (optional repair with --repair flag)
#
# Usage:
#   ./run_translation_workflow.sh [OPTIONS]
#   ./run_translation_workflow.sh --all
#   ./run_translation_workflow.sh --translate
#   ./run_translation_workflow.sh --test
#   ./run_translation_workflow.sh --repair
#   ./run_translation_workflow.sh --help

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration - Auto-detect project root from script location
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Base directories for different output types
SRC_OUTPUT_DIR="$PROJECT_ROOT/src"
TESTS_OUTPUT_DIR="$PROJECT_ROOT/tests"
DOCS_OUTPUT_DIR="$PROJECT_ROOT/docs"
# Legacy output directory (for backward compatibility)
OUTPUT_DIR="$SCRIPT_DIR/translated_modules"
REPAIR_DIR="$SCRIPT_DIR/repair_outputs"

# Modules to process (can be overridden with --modules flag)
DEFAULT_MODULES=("clm_varctl" "SoilStateType" "SoilTemperatureMod")

# Function to print colored output
print_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
${CYAN}JAX-CTSM Translation Workflow Script${NC}

${YELLOW}Usage:${NC}
  $(basename $0) [OPTIONS]

${YELLOW}Options:${NC}
  --all                Run complete workflow (translate → test)
  --translate          Run translation only
  --test               Run test generation only (requires translated modules)
  --repair             Run repair agent only (requires test failures)
  --interactive        Interactive mode - prompts for each step
  
  --modules MODULES    Comma-separated list of modules (default: clm_varctl,SoilStateType,SoilTemperatureMod)
  --output DIR         Output directory (default: ./translated_modules)
  --skip-tests         Skip test generation
  
  -h, --help           Show this help message

${YELLOW}Examples:${NC}
  # Run complete workflow with default modules
  $(basename $0) --all

  # Translate specific modules
  $(basename $0) --translate --module "clm_varctl,WaterFluxType"

  # Generate tests for already-translated modules
  $(basename $0) --test

  # Interactive mode (prompts for each step)
  $(basename $0) --interactive

  # Run repair agent for failed tests
  $(basename $0) --repair

${YELLOW}Workflow Steps:${NC}
  1. ${GREEN}TRANSLATE${NC} - Convert Fortran modules to JAX Python
  2. ${GREEN}TEST${NC}      - Generate comprehensive test suites and run them
  3. ${GREEN}REPAIR${NC}    - Fix any failing tests automatically (run separately)

${YELLOW}Requirements:${NC}
  - Anthropic API key set: export ANTHROPIC_API_KEY="your-key"
  - Python environment with jax-agents installed
  - JSON analysis files in static_analysis_output/

EOF
}

# Function to check requirements
check_requirements() {
    print_info "Checking requirements..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi
    
    # Check API key
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        print_warning "ANTHROPIC_API_KEY not set. Some operations may fail."
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check JSON files
    if [ ! -f "./static_analysis_output/analysis_results.json" ]; then
        print_error "analysis_results.json not found in static_analysis_output/"
        print_info "Please ensure JSON analysis files are available"
        exit 1
    fi
    
    print_success "Requirements check passed"
}

# Function to run translation
run_translation() {
    local modules=("$@")
    
    print_header "STEP 1: Translation (Fortran → JAX Python)"
    
    # Check which modules already have translations
    local modules_needing_translation=()
    local modules_already_translated=()
    
    for module in "${modules[@]}"; do
        translation_found=false
        # Check all possible source directories for existing translation
        for src_dir in "clm_src_main" "clm_src_biogeophys" "clm_src_utils" "multilayer_canopy" "offline_driver" "cime_src_share_util" "clm_src_cpl"; do
            python_file="$SRC_OUTPUT_DIR/$src_dir/$module.py"
            if [ -f "$python_file" ]; then
                translation_found=true
                modules_already_translated+=("$module")
                break
            fi
        done
        
        if [ "$translation_found" = false ]; then
            modules_needing_translation+=("$module")
        fi
    done
    
    # Show status of requested modules
    if [ ${#modules_already_translated[@]} -gt 0 ]; then
        print_info "Translations already exist for: ${modules_already_translated[*]}"
        echo
        print_info "Existing Python files:"
        for module in "${modules_already_translated[@]}"; do
            for src_dir in "clm_src_main" "clm_src_biogeophys" "clm_src_utils" "multilayer_canopy" "offline_driver" "cime_src_share_util" "clm_src_cpl"; do
                python_file="$SRC_OUTPUT_DIR/$src_dir/$module.py"
                if [ -f "$python_file" ]; then
                    print_success "src/$src_dir/$module.py"
                    break
                fi
            done
        done
        echo
    fi
    
    # If all modules are already translated, skip translation
    if [ ${#modules_needing_translation[@]} -eq 0 ]; then
        print_success "All specified modules already translated - skipping translation step"
        return 0
    fi
    
    # Only translate modules that need it
    print_info "Translating modules: ${modules_needing_translation[*]}"
    print_info "Output directory: $OUTPUT_DIR"
    echo
    
    # Run translate_with_json.py with module arguments and structured output
    if python examples/translate_with_json.py "${modules_needing_translation[@]}" --structured-output; then
        print_success "Translation completed successfully"
        
        # List newly translated modules (check both structured and legacy locations)
        echo
        print_info "Newly translated modules:"
        for module in "${modules_needing_translation[@]}"; do
            # Check for structured output in src/
            structured_found=false
            for src_dir in "clm_src_main" "clm_src_biogeophys" "clm_src_utils" "multilayer_canopy" "offline_driver" "cime_src_share_util" "clm_src_cpl"; do
                if [ -f "$SRC_OUTPUT_DIR/$src_dir/$module.py" ]; then
                    print_success "$module → $SRC_OUTPUT_DIR/$src_dir/$module.py"
                    structured_found=true
                    break
                fi
            done
            
            # Check legacy location if not found in structured output
            if [ "$structured_found" = false ]; then
                if [ -d "$OUTPUT_DIR/$module" ]; then
                    print_success "$module → $OUTPUT_DIR/$module/ (legacy)"
                else
                    print_warning "$module - translation may have failed"
                fi
            fi
        done
        return 0
    else
        print_error "Translation failed"
        return 1
    fi
}

# Function to generate tests
run_test_generation() {
    local modules=("$@")
    
    print_header "STEP 2: Test Generation"
    
    # If no modules specified, check if all modules have tests and generate for any missing
    if [ ${#modules[@]} -eq 0 ]; then
        # Check all existing structured tests
        structured_tests_exist=false
        for test_dir in "$TESTS_OUTPUT_DIR/clm_src_main" "$TESTS_OUTPUT_DIR/clm_src_biogeophys" "$TESTS_OUTPUT_DIR/clm_src_utils" "$TESTS_OUTPUT_DIR/multilayer_canopy" "$TESTS_OUTPUT_DIR/offline_driver"; do
            if [ -d "$test_dir" ] && [ "$(find "$test_dir" -name "test_*.py" 2>/dev/null | wc -l)" -gt 0 ]; then
                structured_tests_exist=true
                break
            fi
        done
        
        if [ "$structured_tests_exist" = true ]; then
            print_info "Tests already generated during structured translation phase"
            echo
            print_info "Generated test files in structured layout:"
            find "$TESTS_OUTPUT_DIR" -name "test_*.py" -type f 2>/dev/null | while read test_file; do
                relative_path=$(realpath --relative-to="$TESTS_OUTPUT_DIR" "$test_file")
                print_success "tests/$relative_path"
            done
            return 0
        fi
    else
        # Check if tests exist for the specific modules being processed
        modules_needing_tests=()
        modules_with_tests=()
        
        for module in "${modules[@]}"; do
            test_found=false
            # Check all possible source directories for this module's test
            for src_dir in "clm_src_main" "clm_src_biogeophys" "clm_src_utils" "multilayer_canopy" "offline_driver" "cime_src_share_util" "clm_src_cpl"; do
                test_file="$TESTS_OUTPUT_DIR/$src_dir/test_$module.py"
                if [ -f "$test_file" ]; then
                    test_found=true
                    modules_with_tests+=("$module")
                    break
                fi
            done
            
            if [ "$test_found" = false ]; then
                modules_needing_tests+=("$module")
            fi
        done
        
        # Show status of requested modules
        if [ ${#modules_with_tests[@]} -gt 0 ]; then
            print_info "Tests already exist for: ${modules_with_tests[*]}"
        fi
        
        if [ ${#modules_needing_tests[@]} -eq 0 ]; then
            print_info "All specified modules already have tests"
            echo
            print_info "Existing test files:"
            for module in "${modules_with_tests[@]}"; do
                for src_dir in "clm_src_main" "clm_src_biogeophys" "clm_src_utils" "multilayer_canopy" "offline_driver" "cime_src_share_util" "clm_src_cpl"; do
                    test_file="$TESTS_OUTPUT_DIR/$src_dir/test_$module.py"
                    if [ -f "$test_file" ]; then
                        print_success "tests/$src_dir/test_$module.py"
                        break
                    fi
                done
            done
            return 0
        fi
        
        # Update modules array to only include those needing tests
        modules=("${modules_needing_tests[@]}")
        print_info "Generating tests for modules without existing tests: ${modules[*]}"
    fi
    
    # Generate tests for specific modules (either all modules or those needing tests)
    print_info "Generating tests for modules: ${modules[*]}"
    echo
    
    local all_success=true
    for module in "${modules[@]}"; do
        print_info "Generating tests for $module..."
        
        # Find the module's Python file to determine source directory and generate structured tests
        python_file=""
        for src_dir in "clm_src_main" "clm_src_biogeophys" "clm_src_utils" "multilayer_canopy" "offline_driver" "cime_src_share_util" "clm_src_cpl"; do
            candidate_file="$SRC_OUTPUT_DIR/$src_dir/$module.py"
            if [ -f "$candidate_file" ]; then
                python_file="$candidate_file"
                break
            fi
        done
        
        if [ -z "$python_file" ]; then
            print_error "$module - Python file not found in src/"
            all_success=false
            continue
        fi
        
        # Generate tests with structured output
        if python examples/generate_tests.py --module "$module" --python "$python_file" --output /tmp/dummy --structured-output; then
            print_success "$module - test generation completed with structured output"
        else
            print_error "$module - test generation failed"
            all_success=false
        fi
        echo
    done
    
    if [ "$all_success" = true ]; then
        print_success "All test generation completed successfully"
        return 0
    else
        print_error "Some test generations failed"
        return 1
    fi
}

# Function to run tests and check for failures
run_tests() {
    local modules_to_test=("$@")
    
    print_header "Running Tests"
    
    local has_failures=false
    local failed_modules=()
    
    # Find all test files (check structured location first, then legacy)
    test_files=""
    if [ -d "$TESTS_OUTPUT_DIR" ]; then
        test_files=$(find "$TESTS_OUTPUT_DIR" -name "test_*.py" -type f 2>/dev/null)
    fi
    
    # If no structured tests found, check legacy location
    if [ -z "$test_files" ]; then
        test_files=$(find "$OUTPUT_DIR" -name "test_*.py" -type f 2>/dev/null)
    fi
    
    if [ -z "$test_files" ]; then
        print_warning "No test files found in structured ($TESTS_OUTPUT_DIR) or legacy ($OUTPUT_DIR) locations"
        return 1
    fi
    
    # Filter test files if specific modules are requested
    if [ ${#modules_to_test[@]} -gt 0 ]; then
        print_info "Testing only specified modules: ${modules_to_test[*]}"
        echo
        filtered_test_files=""
        while IFS= read -r test_file; do
            # Extract module name from test filename (test_ModuleName.py -> ModuleName)
            test_filename=$(basename "$test_file")
            module="${test_filename#test_}"
            module="${module%.py}"
            
            for target_module in "${modules_to_test[@]}"; do
                if [ "$module" == "$target_module" ]; then
                    filtered_test_files+="$test_file"$'\n'
                    break
                fi
            done
        done <<< "$test_files"
        test_files="$filtered_test_files"
        
        if [ -z "$test_files" ]; then
            print_warning "No test files found for specified modules: ${modules_to_test[*]}"
            return 1
        fi
    fi
    
    # Run each test file
    while IFS= read -r test_file; do
        [ -z "$test_file" ] && continue  # Skip empty lines
        # Extract module name from test filename (test_ModuleName.py -> ModuleName)
        test_filename=$(basename "$test_file")
        module="${test_filename#test_}"
        module="${module%.py}"
        print_info "Testing $module..."
        
        # Run pytest and capture output (with PYTHONPATH set to find src modules)
        test_output_file="${test_file%.py}_output.txt"
        if PYTHONPATH="$SRC_OUTPUT_DIR:$PYTHONPATH" pytest "$test_file" -v --tb=short > "$test_output_file" 2>&1; then
            print_success "$module - all tests passed"
        else
            print_warning "$module - tests failed"
            has_failures=true
            failed_modules+=("$module:$test_file:$test_output_file")
        fi
    done <<< "$test_files"
    
    # Return failure info
    if [ "$has_failures" = true ]; then
        echo
        print_warning "Some tests failed:"
        for item in "${failed_modules[@]}"; do
            module="${item%%:*}"
            print_error "  - $module"
        done
        
        # Export for use by repair function
        export FAILED_MODULES="${failed_modules[*]}"
        return 1
    else
        print_success "All tests passed!"
        return 0
    fi
}

# Function to run repair agent
run_repair() {
    local modules_to_repair=("$@")
    
    print_header "STEP 3: Repair Agent"
    
    # Check if we have failed modules from previous run
    if [ -z "$FAILED_MODULES" ]; then
        print_info "Running tests to identify failures..."
        if run_tests "${modules_to_repair[@]}"; then
            print_success "All tests pass - no repair needed"
            return 0
        fi
    fi
    
    # Parse failed modules
    IFS=' ' read -ra failed_array <<< "$FAILED_MODULES"
    
    if [ ${#failed_array[@]} -eq 0 ]; then
        print_info "No failed tests found - nothing to repair"
        return 0
    fi
    
    # Filter failed modules based on modules_to_repair if specified
    local modules_to_process=()
    if [ ${#modules_to_repair[@]} -gt 0 ]; then
        print_info "Filtering to repair only specified modules: ${modules_to_repair[*]}"
        for item in "${failed_array[@]}"; do
            IFS=':' read -r module test_file test_output <<< "$item"
            for target_module in "${modules_to_repair[@]}"; do
                if [ "$module" == "$target_module" ]; then
                    modules_to_process+=("$item")
                    break
                fi
            done
        done
        
        if [ ${#modules_to_process[@]} -eq 0 ]; then
            print_warning "None of the specified modules have test failures"
            print_info "Specified modules: ${modules_to_repair[*]}"
            print_info "Failed modules: "
            for item in "${failed_array[@]}"; do
                module="${item%%:*}"
                print_info "  - $module"
            done
            return 0
        fi
    else
        modules_to_process=("${failed_array[@]}")
    fi
    
    print_info "Repairing ${#modules_to_process[@]} module(s) with failures..."
    echo
    
    # Repair each failed module
    for item in "${modules_to_process[@]}"; do
        IFS=':' read -r module test_file test_output <<< "$item"
        
        print_info "Repairing $module..."
        
        # Find Python file in structured layout (src/) or legacy (translated_modules/)
        python_file=""
        for src_dir in "clm_src_main" "clm_src_biogeophys" "multilayer_canopy" "cime_src_share_util" "clm_src_cpl" "clm_src_utils" "offline_driver"; do
            candidate="$SRC_OUTPUT_DIR/$src_dir/${module}.py"
            if [ -f "$candidate" ]; then
                python_file="$candidate"
                break
            fi
        done
        
        # Fall back to legacy location if not found in structured layout
        if [ -z "$python_file" ]; then
        module_dir="$OUTPUT_DIR/$module"
        python_file="$module_dir/${module}.py"
        fi
        
        # Look for Fortran reference in multiple locations
        fortran_ref=""
        
        # Check common Fortran source locations
        fortran_search_paths=(
            "$SCRIPT_DIR/../CLM-ml_v1/clm_src_main/${module}.F90"
            "$SCRIPT_DIR/../CLM-ml_v1/clm_src_biogeophys/${module}.F90"
            "$SCRIPT_DIR/../CLM-ml_v1/clm_src_cpl/${module}.F90"
            "$SCRIPT_DIR/../CLM-ml_v1/clm_src_utils/${module}.F90"
            "$SCRIPT_DIR/../CLM-ml_v1/cime_src_share_util/${module}.F90"
            "$SCRIPT_DIR/../CLM-ml_v1/multilayer_canopy/${module}.F90"
            "$SCRIPT_DIR/../CLM-ml_v1/offline_driver/${module}.F90"
        )
        
        for path in "${fortran_search_paths[@]}"; do
            if [ -f "$path" ]; then
                fortran_ref="$path"
                print_info "Found Fortran source: $fortran_ref"
                break
            fi
        done
        
        # Check if files exist
        if [ ! -f "$python_file" ]; then
            print_warning "Python file not found: $python_file"
            print_info "Searched in structured layout (src/*/) and legacy (translated_modules/)"
            continue
        fi
        
        print_info "Found Python file: $python_file"
        
        if [ ! -f "$test_output" ]; then
            print_warning "Test output not found: $test_output"
            continue
        fi
        
        # For repair, we always repair the main Python file
            python_file_to_repair="$python_file"
        
        # Run repair agent
        repair_output="$REPAIR_DIR/${module}"
        mkdir -p "$repair_output"
        
        if [ -n "$fortran_ref" ] && [ -f "$fortran_ref" ]; then
            # With Fortran reference
            python examples/repair_agent_example.py \
                --module "$module" \
                --fortran "$fortran_file" \
                --python "$python_file" \
                --test-report "$test_output" \
                --test-file "$test_file" \
                --max-iterations 1 \
                -o "$repair_output"
        else
            # Without Fortran reference (less effective)
            print_warning "No Fortran reference found for $module"
            print_info "Searched locations:"
            for path in "${fortran_search_paths[@]}"; do
                print_info "  - $path"
            done
            python examples/repair_agent_example.py \
                --module "$module" \
                --fortran "$python_file" \
                --python "$python_file_to_repair" \
                --test-report "$test_output" \
                --max-iterations 1 \
                -o "$repair_output"
        fi
        
        if [ $? -eq 0 ]; then
            print_success "$module repaired - check $repair_output/"
            
            # Copy corrected code back to original translation file
            corrected_file="$repair_output/${module}_corrected.py"
            if [ -f "$corrected_file" ]; then
                print_info "Copying corrected code to original file..."
                
                # Backup original file in the same directory
                python_dir=$(dirname "$python_file")
                backup_file="$python_dir/${module}_backup_$(date +%Y%m%d_%H%M%S).py"
                cp "$python_file" "$backup_file"
                print_info "Backed up original to: $backup_file"
                
                # Copy corrected code to original location
                cp "$corrected_file" "$python_file"
                print_success "Updated $python_file with corrected code"
                
                # Also save repair notes if they exist (save to docs directory)
                if [ -f "$repair_output/${module}_root_cause_analysis.md" ]; then
                    repair_notes_dir="$DOCS_OUTPUT_DIR/repair_notes"
                    mkdir -p "$repair_notes_dir"
                    cp "$repair_output/${module}_root_cause_analysis.md" "$repair_notes_dir/${module}_repair_notes.md"
                    print_info "Saved repair notes to: $repair_notes_dir/${module}_repair_notes.md"
                fi
            else
                print_warning "Corrected file not found: $corrected_file"
            fi
        else
            print_error "$module repair failed"
        fi
        
        echo
    done
    
    print_success "Repair process completed"
    print_info "Review results in: $REPAIR_DIR/"
}

# Function for interactive mode
interactive_mode() {
    print_header "Interactive Mode"
    
    echo -e "${YELLOW}What would you like to do?${NC}"
    echo "1) Run complete workflow (translate → test → repair)"
    echo "2) Translate modules only"
    echo "3) Generate tests only"
    echo "4) Run tests only"
    echo "5) Repair failed tests"
    echo "6) Exit"
    echo
    read -p "Enter choice [1-6]: " choice
    
    case $choice in
        1)
            print_info "Running complete workflow..."
            check_requirements
            run_translation "${DEFAULT_MODULES[@]}"
            run_test_generation "${DEFAULT_MODULES[@]}"
            run_tests "${DEFAULT_MODULES[@]}"
            read -p "Repair failed tests? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                run_repair "${DEFAULT_MODULES[@]}"
            fi
            ;;
        2)
            check_requirements
            run_translation "${DEFAULT_MODULES[@]}"
            ;;
        3)
            run_test_generation "${DEFAULT_MODULES[@]}"
            ;;
        4)
            run_tests "${DEFAULT_MODULES[@]}"
            ;;
        5)
            run_repair "${DEFAULT_MODULES[@]}"
            ;;
        6)
            print_info "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Main script logic
main() {
    # Parse command line arguments
    RUN_ALL=false
    RUN_TRANSLATE=false
    RUN_TEST=false
    RUN_REPAIR=false
    INTERACTIVE=false
    AUTO_REPAIR=false
    SKIP_TESTS=false
    MODULES=("${DEFAULT_MODULES[@]}")
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                RUN_ALL=true
                shift
                ;;
            --translate)
                RUN_TRANSLATE=true
                shift
                ;;
            --test)
                RUN_TEST=true
                shift
                ;;
            --repair)
                RUN_REPAIR=true
                shift
                ;;
            --interactive)
                INTERACTIVE=true
                shift
                ;;
            --auto-repair)
                AUTO_REPAIR=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --modules)
                IFS=',' read -ra MODULES <<< "$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Show banner
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════╗"
    echo "║  JAX-CTSM Translation Workflow Script ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Interactive mode
    if [ "$INTERACTIVE" = true ]; then
        interactive_mode
        exit 0
    fi
    
    # Run all
    if [ "$RUN_ALL" = true ]; then
        check_requirements
        run_translation "${MODULES[@]}" || exit 1
        
        if [ "$SKIP_TESTS" = false ]; then
            run_test_generation "${MODULES[@]}" || exit 1
            
            if run_tests "${MODULES[@]}"; then
                print_success "All tests passed!"
            else
                print_info "Tests completed with some failures."
                print_info "To repair failed tests, run: $(basename $0) --repair"
            fi
        fi
        exit 0
    fi
    
    # Run individual steps
    if [ "$RUN_TRANSLATE" = true ]; then
        check_requirements
        run_translation "${MODULES[@]}"
        exit 0
    fi
    
    if [ "$RUN_TEST" = true ]; then
        run_test_generation "${MODULES[@]}"
        exit 0
    fi
    
    if [ "$RUN_REPAIR" = true ]; then
        run_repair "${MODULES[@]}"
        exit 0
    fi
    
    # No options provided - show usage
    show_usage
}

# Run main function
main "$@"

