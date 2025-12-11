"""
Example usage of the Repair Agent.

This example demonstrates how to use the RepairAgent to fix a failed
Python/JAX translation by:
1. Analyzing test failures
2. Generating corrected code
3. Running tests iteratively
4. Producing a root cause analysis report
"""

import argparse
from pathlib import Path
from jax_agents.repair_agent import RepairAgent
from rich.console import Console

console = Console()


def repair_failed_translation(
    module_name: str,
    fortran_code: str,
    failed_python_code: str,
    test_report: str,
    test_file_path: Path = None,
    output_dir: Path = None,
    max_iterations: int = 5,
):
    """
    Repair a failed translation.
    
    Args:
        module_name: Name of the module
        fortran_code: Original Fortran code
        failed_python_code: Failed Python translation
        test_report: Test failure report
        test_file_path: Optional pytest file for automatic testing
        output_dir: Output directory for results
        max_iterations: Maximum repair iterations
    """
    # Initialize the repair agent
    repair_agent = RepairAgent(
        max_repair_iterations=max_iterations,
    )
    
    if output_dir is None:
        output_dir = Path("repair_outputs")
    
    console.print("Starting repair process...")
    console.print("=" * 80)
    
    # Run the repair process
    result = repair_agent.repair_translation(
        module_name=module_name,
        fortran_code=fortran_code,
        failed_python_code=failed_python_code,
        test_report=test_report,
        test_file_path=test_file_path,
        output_dir=output_dir,
    )
    
    # Display results
    console.print("\n" + "=" * 80)
    console.print("REPAIR RESULTS")
    console.print("=" * 80)
    console.print(f"Module: {result.module_name}")
    console.print(f"Iterations: {result.iterations}")
    console.print(f"All tests passed: {result.all_tests_passed}")
    console.print(f"\nRoot causes identified: {len(result.failure_analysis.get('root_causes', []))}")
    
    # Show corrected code snippet
    console.print("\n" + "=" * 80)
    console.print("CORRECTED CODE (first 20 lines):")
    console.print("=" * 80)
    lines = result.corrected_python_code.split('\n')[:20]
    console.print('\n'.join(lines))
    
    # Show root cause analysis summary
    console.print("\n" + "=" * 80)
    console.print("ROOT CAUSE ANALYSIS (first 500 chars):")
    console.print("=" * 80)
    console.print(result.root_cause_analysis[:500] + "...")
    
    # Print cost estimate
    console.print("\n" + "=" * 80)
    console.print("COST ESTIMATE")
    console.print("=" * 80)
    cost_info = repair_agent.get_cost_estimate()
    console.print(f"Input tokens: {cost_info['input_tokens']:,}")
    console.print(f"Output tokens: {cost_info['output_tokens']:,}")
    console.print(f"Total cost: ${cost_info['total_cost_usd']:.4f}")
    
    return result


def run_example_repair():
    """Run the built-in example with sample bug."""
    
    module_name = "SoilTemperatureMod"
    
    # Original Fortran code
    fortran_code = """
subroutine calculate_temperature(temp_in, temp_out, n)
    implicit none
    integer, intent(in) :: n
    real(r8), intent(in) :: temp_in(n)
    real(r8), intent(out) :: temp_out(n)
    integer :: i
    
    do i = 1, n
        temp_out(i) = temp_in(i) + 273.15
    end do
end subroutine calculate_temperature
"""
    
    # Failed Python translation (has indexing bug)
    failed_python_code = """
import jax.numpy as jnp
from jax import jit

@jit
def calculate_temperature(temp_in):
    '''Convert temperature from Celsius to Kelvin.
    
    Args:
        temp_in: Input temperatures in Celsius
        
    Returns:
        Temperatures in Kelvin
    '''
    # BUG: Using wrong indexing (0-based vs 1-based)
    n = temp_in.shape[0]
    temp_out = jnp.zeros(n)
    
    for i in range(1, n):  # BUG: Should start from 0
        temp_out = temp_out.at[i].set(temp_in[i] + 273.15)
    
    return temp_out
"""
    
    # Test report showing failures
    test_report = """
============================= test session starts ==============================
collected 5 items

test_SoilTemperatureMod.py::test_basic_conversion FAILED             [ 20%]
test_SoilTemperatureMod.py::test_array_conversion FAILED             [ 40%]
test_SoilTemperatureMod.py::test_edge_cases PASSED                   [ 60%]
test_SoilTemperatureMod.py::test_negative_temps FAILED               [ 80%]
test_SoilTemperatureMod.py::test_zero_temp FAILED                    [100%]

=================================== FAILURES ===================================
______________________________ test_basic_conversion ___________________________
    
    def test_basic_conversion():
        temp_in = jnp.array([0.0, 25.0, 100.0])
        expected = jnp.array([273.15, 298.15, 373.15])
        result = calculate_temperature(temp_in)
>       assert jnp.allclose(result, expected)
E       AssertionError: Arrays not equal
E       Expected: [273.15, 298.15, 373.15]
E       Got:      [0.0, 298.15, 373.15]

Error: First element is not converted (remains 0.0 instead of 273.15)

______________________________ test_array_conversion ___________________________
Similar error - first element not being processed

============================== SUMMARY =====================================
4 failed, 1 passed in 2.34s
"""
    
    return repair_failed_translation(
        module_name=module_name,
        fortran_code=fortran_code,
        failed_python_code=failed_python_code,
        test_report=test_report,
        test_file_path=None,
        output_dir=Path("repair_outputs"),
        max_iterations=5,
    )


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Repair failed Python/JAX translations automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with built-in example (demonstrates repair on sample bug)
  python repair_agent_example.py --example
  
  # Repair with your own code files
  python repair_agent_example.py \\
    --module SoilTemperatureMod \\
    --fortran /path/to/original.F90 \\
    --python /path/to/failed.py \\
    --test-report /path/to/pytest_output.txt
  
  # With automatic test execution
  python repair_agent_example.py \\
    --module MyModule \\
    --fortran /path/to/original.F90 \\
    --python /path/to/failed.py \\
    --test-report /path/to/pytest_output.txt \\
    --test-file /path/to/test_MyModule.py
  
  # Interactive mode
  python repair_agent_example.py --interactive
  
  # Increase max iterations for complex issues
  python repair_agent_example.py --example --max-iterations 10
        """
    )
    
    parser.add_argument(
        "--module",
        help="Module name"
    )
    parser.add_argument(
        "--fortran",
        help="Path to original Fortran code file"
    )
    parser.add_argument(
        "--python",
        help="Path to failed Python code file"
    )
    parser.add_argument(
        "--test-report",
        help="Path to test report file (pytest output)"
    )
    parser.add_argument(
        "--test-file",
        help="Path to pytest file (for automatic re-testing)"
    )
    parser.add_argument(
        "-o", "--output",
        default="repair_outputs",
        help="Output directory (default: repair_outputs)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum repair iterations (default: 5)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - prompts for inputs"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run with built-in example (sample bug)"
    )
    
    args = parser.parse_args()
    
    # Example mode
    if args.example:
        console.print("[cyan]Running built-in example with sample bug[/cyan]\n")
        result = run_example_repair()
        console.print("\n✓ Repair agent example completed!")
        console.print(f"  Outputs saved to: repair_outputs/")
        return
    
    # Interactive mode
    if args.interactive:
        console.print("\n[bold cyan]Interactive Repair Agent[/bold cyan]\n")
        
        console.print("[cyan]Module name:[/cyan]")
        module_name = input("> ").strip()
        
        console.print("\n[cyan]Path to original Fortran code file:[/cyan]")
        fortran_path = input("> ").strip()
        with open(fortran_path, 'r') as f:
            fortran_code = f.read()
        
        console.print("\n[cyan]Path to failed Python code file:[/cyan]")
        python_path = input("> ").strip()
        with open(python_path, 'r') as f:
            failed_python_code = f.read()
        
        console.print("\n[cyan]Path to test report file (pytest output):[/cyan]")
        test_report_path = input("> ").strip()
        with open(test_report_path, 'r') as f:
            test_report = f.read()
        
        console.print("\n[cyan]Path to pytest file (optional, press Enter to skip):[/cyan]")
        test_file_input = input("> ").strip()
        test_file_path = Path(test_file_input) if test_file_input else None
        
        console.print("\n[cyan]Output directory (default: repair_outputs):[/cyan]")
        output_input = input("> ").strip() or "repair_outputs"
        output_dir = Path(output_input)
        
        console.print("\n[cyan]Max iterations (default: 5):[/cyan]")
        max_iter_input = input("> ").strip()
        max_iterations = int(max_iter_input) if max_iter_input else 5
        
        result = repair_failed_translation(
            module_name=module_name,
            fortran_code=fortran_code,
            failed_python_code=failed_python_code,
            test_report=test_report,
            test_file_path=test_file_path,
            output_dir=output_dir,
            max_iterations=max_iterations,
        )
        
        console.print("\n✓ Repair completed!")
        console.print(f"  Outputs saved to: {output_dir}/")
        return
    
    # CLI mode
    if args.module and args.fortran and args.python and args.test_report:
        # Read files
        with open(args.fortran, 'r') as f:
            fortran_code = f.read()
        
        with open(args.python, 'r') as f:
            failed_python_code = f.read()
        
        with open(args.test_report, 'r') as f:
            test_report = f.read()
        
        test_file_path = Path(args.test_file) if args.test_file else None
        
        result = repair_failed_translation(
            module_name=args.module,
            fortran_code=fortran_code,
            failed_python_code=failed_python_code,
            test_report=test_report,
            test_file_path=test_file_path,
            output_dir=Path(args.output),
            max_iterations=args.max_iterations,
        )
        
        console.print("\n✓ Repair completed!")
        console.print(f"  Outputs saved to: {args.output}/")
    else:
        parser.print_help()
        console.print("\n[yellow]Please provide all required arguments, use --interactive, or --example[/yellow]")


if __name__ == "__main__":
    main()
