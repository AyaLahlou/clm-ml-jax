#!/usr/bin/env python3
"""
Generate tests for translated Python/JAX modules.

This script uses the Test Agent to create comprehensive pytest files
and test data for JAX implementations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_agents import TestAgent
from rich.console import Console

console = Console()


def generate_tests_for_module(
    module_name: str,
    python_file: Path,
    output_dir: Path,
    num_cases: int = 10,
    structured_output: bool = False,
    project_root: Path = None,
):
    """
    Generate test suite for a Python/JAX module.
    
    Args:
        module_name: Name of the module
        python_file: Path to Python/JAX implementation
        output_dir: Directory to save test outputs
        num_cases: Number of test cases to generate
    """
    console.print(f"\n[bold cyan]Generating tests for {module_name}[/bold cyan]")
    
    # Read Python code
    if not python_file.exists():
        console.print(f"[red]Error: File not found: {python_file}[/red]")
        return
    
    with open(python_file, 'r') as f:
        python_code = f.read()
    
    # Initialize test agent
    test_agent = TestAgent()
    
    # Determine source directory from python_file path
    source_directory = "clm_src_main"  # default
    parts = python_file.parts
    for part in parts:
        if part in ['clm_src_main', 'clm_src_biogeophys', 'clm_src_utils', 'multilayer_canopy', 'offline_driver', 'cime_src_share_util', 'clm_src_cpl']:
            source_directory = part
            break
    
    console.print(f"[dim]Detected source directory: {source_directory}[/dim]")

    # Generate tests
    try:
        # Use structured output approach if requested
        if structured_output and project_root:
            # Don't pass output_dir to avoid duplicate saving
            result = test_agent.generate_tests(
                module_name=module_name,
                python_code=python_code,
                num_test_cases=num_cases,
                include_edge_cases=True,
                include_performance_tests=False,
                source_directory=source_directory,
            )
            
            # Save with structured output
            saved_files = result.save_structured(project_root, source_directory)
            
            # Print summary for structured output
            console.print("\n[bold green]✓ Test generation complete with structured output![/bold green]")
            console.print(f"\n[cyan]Generated files:[/cyan]")
            for file_type, file_path in saved_files.items():
                relative_path = file_path.relative_to(project_root)
                console.print(f"  • {file_type}: {relative_path}")
        else:
            # Traditional output method
            result = test_agent.generate_tests(
                module_name=module_name,
                python_code=python_code,
                output_dir=output_dir,
                num_test_cases=num_cases,
                include_edge_cases=True,
                include_performance_tests=False,
                source_directory=source_directory,
            )
            
            # Print summary for traditional output
            console.print("\n[bold green]✓ Test generation complete![/bold green]")
            console.print(f"\n[cyan]Generated files:[/cyan]")
            console.print(f"  • pytest file: test_{module_name}.py")
            console.print(f"  • test data: test_data_{module_name}.json")
            console.print(f"  • documentation: test_documentation_{module_name}.md")
            
            console.print(f"\n[cyan]To run tests:[/cyan]")
            console.print(f"  cd {output_dir}")
            console.print(f"  pytest test_{module_name}.py -v")
        
        # Show cost
        cost = test_agent.get_cost_estimate()
        console.print(f"\n[dim]Cost: ${cost['total_cost_usd']:.4f}[/dim]")
        
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


def generate_tests_for_all_modules():
    """
    Generate tests for all translated modules.
    """
    project_root = Path(__file__).parent.parent
    translated_dir = project_root / "translated_modules"
    
    if not translated_dir.exists():
        console.print(f"[red]Directory not found: {translated_dir}[/red]")
        return
    
    # Find all translated modules
    modules_found = []
    for module_dir in translated_dir.iterdir():
        if not module_dir.is_dir():
            continue
        
        module_name = module_dir.name
        python_file = module_dir / f"{module_name}.py"
        
        if python_file.exists():
            modules_found.append((module_name, python_file, module_dir))
    
    if not modules_found:
        console.print("[yellow]No translated modules found[/yellow]")
        return
    
    console.print(f"\n[bold cyan]Found {len(modules_found)} modules[/bold cyan]")
    
    # Generate tests for each
    for module_name, python_file, module_dir in modules_found:
        output_dir = module_dir / "tests"
        try:
            generate_tests_for_module(
                module_name=module_name,
                python_file=python_file,
                output_dir=output_dir,
                num_cases=10,
            )
        except Exception as e:
            console.print(f"[red]Failed for {module_name}: {e}[/red]")
            continue


def interactive_mode():
    """Interactive test generation."""
    console.print("\n[bold cyan]Test Generator - Interactive Mode[/bold cyan]\n")
    
    # Get module name
    console.print("[cyan]Enter module name:[/cyan]")
    module_name = input("> ").strip()
    
    # Get Python file path
    console.print("\n[cyan]Enter path to Python/JAX file:[/cyan]")
    python_path = input("> ").strip()
    python_file = Path(python_path)
    
    # Get output directory
    console.print("\n[cyan]Enter output directory:[/cyan]")
    output_path = input("> ").strip()
    output_dir = Path(output_path)
    
    # Get number of test cases
    console.print("\n[cyan]Number of test cases (default: 10):[/cyan]")
    num_str = input("> ").strip()
    num_cases = int(num_str) if num_str else 10
    
    # Generate
    generate_tests_for_module(module_name, python_file, output_dir, num_cases)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate tests for Python/JAX modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tests for a specific module (auto-locates in translated_modules/)
  python generate_tests.py SoilTemperatureMod
  
  # Generate with custom number of test cases
  python generate_tests.py SoilTemperatureMod --num-cases 20
  
  # Generate for all translated modules
  python generate_tests.py --all
  
  # Interactive mode
  python generate_tests.py --interactive
  
  # Manual mode (specify all paths)
  python generate_tests.py --module SoilTemperatureMod \\
    --python ./translated_modules/SoilTemperatureMod/SoilTemperatureMod.py \\
    --output ./translated_modules/SoilTemperatureMod/tests
        """
    )
    
    parser.add_argument(
        "module_name",
        nargs="?",
        help="Module name (will auto-locate in translated_modules/)"
    )
    parser.add_argument(
        "--module",
        help="Module name (manual mode)"
    )
    parser.add_argument(
        "--python",
        help="Path to Python/JAX file (manual mode)"
    )
    parser.add_argument(
        "--output",
        help="Output directory (manual mode)"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of test cases (default: 10)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all modules in translated_modules/"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Use structured output (save to tests/, tests/test_data/, docs/)"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.all:
        generate_tests_for_all_modules()
    elif args.module_name:
        # Auto-locate mode: find module in translated_modules/
        project_root = Path(__file__).parent.parent
        translated_dir = project_root / "translated_modules"
        module_dir = translated_dir / args.module_name
        python_file = module_dir / f"{args.module_name}.py"
        output_dir = module_dir / "tests"
        
        if not module_dir.exists():
            console.print(f"[red]Error: Module directory not found: {module_dir}[/red]")
            console.print(f"[yellow]Available modules in {translated_dir}:[/yellow]")
            if translated_dir.exists():
                for d in sorted(translated_dir.iterdir()):
                    if d.is_dir():
                        console.print(f"  • {d.name}")
            return
        
        if not python_file.exists():
            console.print(f"[red]Error: Python file not found: {python_file}[/red]")
            return
        
        generate_tests_for_module(
            module_name=args.module_name,
            python_file=python_file,
            output_dir=output_dir,
            num_cases=args.num_cases,
            structured_output=args.structured_output,
            project_root=project_root if args.structured_output else None,
        )
    elif args.module and args.python and args.output:
        # Manual mode: use specified paths
        project_root = Path(__file__).parent.parent.parent  # Go up to clm-ml-jax root
        generate_tests_for_module(
            module_name=args.module,
            python_file=Path(args.python),
            output_dir=Path(args.output),
            num_cases=args.num_cases,
            structured_output=args.structured_output,
            project_root=project_root if args.structured_output else None,
        )
    else:
        parser.print_help()
        console.print("\n[yellow]Please provide a module name, use --all, --interactive, or specify all manual mode arguments[/yellow]")


if __name__ == "__main__":
    main()

