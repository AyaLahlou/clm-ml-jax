#!/usr/bin/env python3
"""
Example: Translate Fortran modules using static analysis JSON files.

This demonstrates how to use the updated TranslatorAgent with 
analysis_results.json and translation_units.json as inputs.
"""

import sys
import argparse
from pathlib import Path
from jax_agents.translator import TranslatorAgent
from rich.console import Console

console = Console()


def main():
    """Example translation workflow with JSON files."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Translate Fortran modules to JAX using static analysis JSON files"
    )
    parser.add_argument(
        "modules",
        nargs="*",
        default=["clm_varctl", "SoilStateType", "SoilTemperatureMod"],
        help="Module names to translate (default: clm_varctl, SoilStateType, SoilTemperatureMod)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: jax-agents/translated_modules)"
    )
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Use structured output (files to src/, tests/, docs/ directories)"
    )
    args = parser.parse_args()
    
    # Setup paths - use current project structure (auto-detect from script location)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent  # jax-agents/examples -> jax-agents -> clm-ml-jax
    
    # JSON files are in jax-agents/static_analysis_output 
    analysis_dir = project_root / "jax-agents/static_analysis_output"
    analysis_results_json = analysis_dir / "analysis_results.json"
    translation_units_json = analysis_dir / "translation_units.json"
    
    # Path to jax-ctsm for reference patterns (adjust if needed)
    # Safely check for jax-ctsm directory without permission errors
    try:
        jax_ctsm_candidate = project_root.parent / "jax-ctsm"
        jax_ctsm_dir = jax_ctsm_candidate if jax_ctsm_candidate.exists() else None
    except (PermissionError, OSError):
        jax_ctsm_dir = None
    
    # Path to Fortran source files
    fortran_root = project_root / "CLM-ml_v1"
    
    # Output directory for translated code
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = project_root / "jax-agents/translated_modules"
    
    # Verify JSON files exist
    if not analysis_results_json.exists():
        console.print(f"[red]Error: {analysis_results_json} not found[/red]")
        return
    
    if not translation_units_json.exists():
        console.print(f"[red]Error: {translation_units_json} not found[/red]")
        return
    
    console.print("[bold green]🚀 Initializing Translator Agent with JSON files...[/bold green]")
    
    # Initialize translator with JSON files
    translator = TranslatorAgent(
        analysis_results_path=analysis_results_json,
        translation_units_path=translation_units_json,
        jax_ctsm_dir=jax_ctsm_dir,
        fortran_root=fortran_root,
        # Optional: override LLM settings
        model="claude-sonnet-4-5",
        temperature=0.0,
        max_tokens=48000,
    )
    
    console.print("[green]✓ Translator initialized successfully![/green]\n")
    
    # Translate requested modules
    modules_to_translate = args.modules
    console.print(f"[bold cyan]Translating {len(modules_to_translate)} module(s): {', '.join(modules_to_translate)}[/bold cyan]\n")
    
    success_count = 0
    fail_count = 0
    
    for i, module_name in enumerate(modules_to_translate, 1):
        console.print(f"[bold cyan]Module {i}/{len(modules_to_translate)}: Translating '{module_name}'[/bold cyan]")
        try:
            result = translator.translate_module(module_name=module_name)
            
            # Use structured output if requested
            if args.structured_output:
                try:
                    saved_files = result.save_structured(project_root)
                    console.print(f"[green]✓ Translated {module_name} successfully with structured output![/green]")
                    console.print(f"[dim]Saved {len(saved_files)} files across project structure[/dim]")
                except Exception as save_error:
                    console.print(f"[red]✗ Structured output failed: {save_error}[/red]")
                    # Fallback to traditional save method
                    result.save(output_dir / module_name)
                    console.print(f"[yellow]⚠ Used legacy output method instead[/yellow]")
            else:
                # Traditional output method
                result.save(output_dir / module_name)
                console.print(f"[green]✓ Translated {module_name} successfully![/green]")
            
            console.print(f"[dim]Generated {len(result.physics_code)} chars of physics code[/dim]\n")
            success_count += 1
        except Exception as e:
            console.print(f"[red]✗ Error translating {module_name}: {e}[/red]\n")
            fail_count += 1
            import traceback
            traceback.print_exc()
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Translation Summary[/bold green]")
    console.print("=" * 60)
    console.print(f"[green]✓ Success: {success_count}[/green]")
    if fail_count > 0:
        console.print(f"[red]✗ Failed: {fail_count}[/red]")
    console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
    console.print("=" * 60)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
