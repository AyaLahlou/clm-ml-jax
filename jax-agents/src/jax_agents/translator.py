"""
Translator Agent for converting Fortran to JAX.

This agent translates Fortran CTSM code to JAX following established patterns.
Uses comprehensive static analysis results and translation unit breakdowns.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

from jax_agents.base_agent import BaseAgent
# from jax_agents.prompts.translation_prompts import TRANSLATION_PROMPTS
from jax_agents.prompts.translation_prompts_v2 import TRANSLATION_PROMPTS
from jax_agents.utils.config_loader import get_llm_config
from rich.console import Console

console = Console()


@dataclass
class TranslationResult:
    """Result of translating a Fortran module to JAX."""
    module_name: str
    physics_code: str
    source_directory: Optional[str] = None  # Track source directory for smart output routing
    params_code: Optional[str] = None
    test_code: Optional[str] = None
    translation_notes: str = ""
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """
        Save translated code to files.
        
        Args:
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping file type to path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save main physics module
        physics_file = output_dir / f"{self.module_name}.py"
        with open(physics_file, 'w') as f:
            f.write(self.physics_code)
        saved_files["physics"] = physics_file
        console.print(f"[green]✓ Saved physics module to {physics_file}[/green]")
        
        # Save parameters if present
        if self.params_code:
            params_file = output_dir / f"{self.module_name}_params.py"
            with open(params_file, 'w') as f:
                f.write(self.params_code)
            saved_files["params"] = params_file
            console.print(f"[green]✓ Saved parameters to {params_file}[/green]")
        
        # Save tests if generated
        if self.test_code:
            test_file = output_dir / f"test_{self.module_name}.py"
            with open(test_file, 'w') as f:
                f.write(self.test_code)
            saved_files["test"] = test_file
            console.print(f"[green]✓ Saved tests to {test_file}[/green]")
        
        # Save translation notes
        if self.translation_notes:
            notes_file = output_dir / f"{self.module_name}_translation_notes.md"
            with open(notes_file, 'w') as f:
                f.write(self.translation_notes)
            saved_files["notes"] = notes_file
            console.print(f"[green]✓ Saved translation notes to {notes_file}[/green]")
        
        return saved_files

    def save_structured(self, project_root: Path) -> Dict[str, Path]:
        """
        Save translated code to structured directories based on content type.
        
        Args:
            project_root: Root directory of the CLM-ml-jax project
            
        Returns:
            Dictionary mapping file type to path
        """
        saved_files = {}
        
        # Determine source directory mapping
        if self.source_directory:
            source_subdir = self.source_directory
        else:
            # Fallback to clm_src_main for unknown sources
            source_subdir = "clm_src_main"
        
        # Save main physics module to src/
        src_target_dir = project_root / "src" / source_subdir
        src_target_dir.mkdir(parents=True, exist_ok=True)
        physics_file = src_target_dir / f"{self.module_name}.py"
        with open(physics_file, 'w') as f:
            f.write(self.physics_code)
        saved_files["physics"] = physics_file
        console.print(f"[green]✓ Saved physics module to {physics_file}[/green]")
        
        # Save parameters if present to same source directory
        if self.params_code:
            params_file = src_target_dir / f"{self.module_name}_params.py"
            with open(params_file, 'w') as f:
                f.write(self.params_code)
            saved_files["params"] = params_file
            console.print(f"[green]✓ Saved parameters to {params_file}[/green]")
        
        # Save tests to tests/ with matching directory structure
        if self.test_code:
            test_target_dir = project_root / "tests" / source_subdir
            test_target_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_target_dir / f"test_{self.module_name}.py"
            with open(test_file, 'w') as f:
                f.write(self.test_code)
            saved_files["test"] = test_file
            console.print(f"[green]✓ Saved tests to {test_file}[/green]")
        
        # Save translation notes to docs/
        if self.translation_notes:
            docs_target_dir = project_root / "docs" / "translation_notes"
            docs_target_dir.mkdir(parents=True, exist_ok=True)
            notes_file = docs_target_dir / f"{self.module_name}_translation_notes.md"
            with open(notes_file, 'w') as f:
                f.write(self.translation_notes)
            saved_files["notes"] = notes_file
            console.print(f"[green]✓ Saved translation notes to {notes_file}[/green]")
        
        return saved_files


class TranslatorAgent(BaseAgent):
    """
    Agent for translating Fortran code to JAX.
    
    Responsibilities:
    - Convert Fortran syntax to JAX/Python
    - Apply JAX best practices (pure functions, immutable state)
    - Follow established jax-ctsm patterns
    - Generate type hints and documentation
    - Create parameter classes
    - Convert loops to vectorized operations
    """
    
    def __init__(
        self,
        analysis_results_path: Optional[Path] = None,
        translation_units_path: Optional[Path] = None,
        jax_ctsm_dir: Optional[Path] = None,
        fortran_root: Optional[Path] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize Translator Agent.
        
        Args:
            analysis_results_path: Path to analysis_results.json from static analyzer
            translation_units_path: Path to translation_units.json from static analyzer
            jax_ctsm_dir: Path to jax-ctsm directory (for reference patterns)
            fortran_root: Path to Fortran source root (overrides paths in JSON)
            model: Claude model to use (defaults to config.yaml)
            temperature: Sampling temperature (defaults to config.yaml)
            max_tokens: Maximum tokens in response (defaults to config.yaml)
        """
        # Load config if not provided
        llm_config = get_llm_config()
        
        super().__init__(
            name="Translator",
            role="Fortran to JAX code translator",
            model=model or llm_config.get("model", "claude-sonnet-4-5"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 48000),
        )
        
        self.jax_ctsm_dir = jax_ctsm_dir
        self.fortran_root = fortran_root
        self.reference_patterns = self._load_reference_patterns()
        
        # Load static analysis results
        self.analysis_results = self._load_json(analysis_results_path) if analysis_results_path else None
        self.translation_units = self._load_json(translation_units_path) if translation_units_path else None
    
    def translate_module(
        self,
        module_name: str,
        fortran_file: Optional[Path] = None,
        analysis: Optional[Any] = None,
        output_dir: Optional[Path] = None,
    ) -> TranslationResult:
        """
        Translate a complete Fortran module to JAX by processing each translation unit.
        
        Args:
            module_name: Name of the module to translate (must match name in JSON files)
            fortran_file: Optional path to Fortran source file (will extract from JSON if not provided)
            analysis: Optional static analysis result (legacy, use JSON files instead)
            output_dir: Optional directory to save output
            
        Returns:
            TranslationResult with generated code
        """
        console.print(f"\n[bold cyan]🔄 Translating {module_name} to JAX[/bold cyan]")
        
        # Extract module-specific information from JSON files
        module_info = self._extract_module_info(module_name)
        
        if not module_info:
            raise ValueError(f"Module '{module_name}' not found in analysis results")
        
        # Read Fortran source
        if fortran_file:
            fortran_path = fortran_file
        else:
            fortran_path = self._remap_fortran_path(module_info['file_path'])
        
        console.print(f"[dim]Reading from: {fortran_path}[/dim]")
        
        with open(fortran_path, 'r') as f:
            fortran_code = f.read()
        
        fortran_lines = fortran_code.split('\n')
        
        # Get reference pattern
        reference_pattern = self._get_reference_pattern()
        
        # Get translation units for this module
        module_units = self._get_module_units(module_name)
        
        if not module_units:
            console.print("[yellow]⚠ No translation units found, falling back to full module translation[/yellow]")
            return self._translate_module_legacy(module_name, fortran_code, module_info, reference_pattern, output_dir)
        
        console.print(f"[cyan]Found {len(module_units)} translation units[/cyan]")
        
        # Translate each unit iteratively
        translated_units = []
        for i, unit in enumerate(module_units, 1):
            console.print(f"[cyan]Translating unit {i}/{len(module_units)}: {unit.get('id', 'unknown')} ({unit.get('unit_type', 'unknown')})[/cyan]")
            
            translated_code = self._translate_unit(
                module_name=module_name,
                unit=unit,
                fortran_lines=fortran_lines,
                module_info=module_info,
                reference_pattern=reference_pattern,
                previously_translated=translated_units,
            )
            
            translated_units.append({
                "unit_id": unit.get("id", "unknown"),
                "unit_type": unit.get("unit_type", "unknown"),
                "translated_code": translated_code,
                "original_lines": f"{unit.get('line_start', 0)}-{unit.get('line_end', 0)}",
            })
        
        # Assemble all units into final module
        console.print("[cyan]Assembling complete module...[/cyan]")
        result = self._assemble_module(
            module_name=module_name,
            translated_units=translated_units,
            module_info=module_info,
            reference_pattern=reference_pattern,
        )
        
        # Extract source directory from file path for smart output routing
        result.source_directory = self._extract_source_directory(module_info.get('file_path', ''))
        
        console.print(f"[green]✓ Translation complete![/green]")
        
        # Save if output directory specified
        if output_dir:
            result.save(output_dir)
        
        return result
    
    def translate_function(
        self,
        fortran_code: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Translate a single Fortran subroutine to JAX function.
        
        Args:
            fortran_code: Fortran subroutine code
            context: Context from static analysis
            
        Returns:
            Translated JAX function code
        """
        prompt = TRANSLATION_PROMPTS["translate_function"].format(
            fortran_code=fortran_code,
            context=context,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATION_PROMPTS["system"],
        )
        
        return self._extract_code(response)
    
    def convert_data_structure(
        self,
        fortran_type: str,
    ) -> str:
        """
        Convert Fortran derived type to JAX NamedTuple.
        
        Args:
            fortran_type: Fortran type definition
            
        Returns:
            Python NamedTuple code
        """
        prompt = TRANSLATION_PROMPTS["convert_data_structure"].format(
            fortran_type=fortran_type,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATION_PROMPTS["system"],
        )
        
        return self._extract_code(response)
    
    def vectorize_loop(
        self,
        loop_code: str,
        loop_analysis: Dict[str, Any],
    ) -> str:
        """
        Convert Fortran loop to vectorized JAX operations.
        
        Args:
            loop_code: Fortran loop code
            loop_analysis: Analysis of loop structure
            
        Returns:
            Vectorized JAX code
        """
        prompt = TRANSLATION_PROMPTS["vectorize_loop"].format(
            loop_code=loop_code,
            loop_analysis=loop_analysis,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATION_PROMPTS["system"],
        )
        
        return self._extract_code(response)
    
    def handle_conditional(
        self,
        conditional_code: str,
    ) -> str:
        """
        Convert Fortran conditional to JIT-compatible JAX.
        
        Args:
            conditional_code: Fortran if/else code
            
        Returns:
            JAX jnp.where code
        """
        prompt = TRANSLATION_PROMPTS["handle_conditional"].format(
            conditional_code=conditional_code,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATION_PROMPTS["system"],
        )
        
        return self._extract_code(response)
    
    def create_parameters(
        self,
        parameters: str,
    ) -> str:
        """
        Create JAX parameter class from Fortran parameters.
        
        Args:
            parameters: Fortran parameter definitions
            
        Returns:
            Python NamedTuple parameter class
        """
        prompt = TRANSLATION_PROMPTS["create_parameters"].format(
            parameters=parameters,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATION_PROMPTS["system"],
        )
        
        return self._extract_code(response)
    
    def _get_module_units(self, module_name: str) -> List[Dict[str, Any]]:
        """
        Get translation units for a specific module.
        
        Args:
            module_name: Module name (case-insensitive)
            
        Returns:
            List of translation units for the module
        """
        if not self.translation_units:
            return []
        
        units = self.translation_units.get("translation_units", [])
        module_units = [
            unit for unit in units 
            if unit.get("module_name", "").lower() == module_name.lower()
        ]
        
        # Sort by line_start to process in order
        module_units.sort(key=lambda u: u.get("line_start", 0))
        
        return module_units
    
    def _translate_unit(
        self,
        module_name: str,
        unit: Dict[str, Any],
        fortran_lines: List[str],
        module_info: Dict[str, Any],
        reference_pattern: str,
        previously_translated: List[Dict[str, Any]],
    ) -> str:
        """
        Translate a single translation unit.
        
        Args:
            module_name: Module name
            unit: Translation unit metadata
            fortran_lines: Full Fortran source as list of lines
            module_info: Module information from analysis
            reference_pattern: Reference JAX pattern
            previously_translated: Previously translated units for context
            
        Returns:
            Translated code for this unit
        """
        # Extract Fortran code for this unit
        line_start = unit.get("line_start", 1) - 1  # Convert to 0-indexed
        line_end = unit.get("line_end", len(fortran_lines))
        unit_fortran = '\n'.join(fortran_lines[line_start:line_end])
        
        # Build context
        context = {
            "module_dependencies": self._get_module_dependencies(module_name),
            "previously_translated": [
                {
                    "unit_id": u["unit_id"],
                    "unit_type": u["unit_type"],
                    "code_snippet": u["translated_code"][:200] + "..." if len(u["translated_code"]) > 200 else u["translated_code"],
                }
                for u in previously_translated
            ],
        }
        
        # Build prompt
        prompt = TRANSLATION_PROMPTS["translate_unit"].format(
            module_name=module_name,
            unit_id=unit.get("id", "unknown"),
            unit_type=unit.get("unit_type", "unknown"),
            line_start=unit.get("line_start", 1),
            line_end=unit.get("line_end", len(fortran_lines)),
            fortran_code=unit_fortran,
            unit_info=json.dumps(unit, indent=2),
            context=json.dumps(context, indent=2),
            reference_pattern=reference_pattern,
            parent_id=unit.get("parent_id", "N/A"),
        )
        
        # Query LLM
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATION_PROMPTS["system"],
            max_tokens=self.max_tokens,
        )
        
        return self._extract_code(response)
    
    def _assemble_module(
        self,
        module_name: str,
        translated_units: List[Dict[str, Any]],
        module_info: Dict[str, Any],
        reference_pattern: str,
    ) -> TranslationResult:
        """
        Assemble translated units into complete module.
        
        Args:
            module_name: Module name
            translated_units: List of translated units
            module_info: Module information
            reference_pattern: Reference pattern
            
        Returns:
            TranslationResult with assembled code
        """
        # Build assembly prompt
        prompt = TRANSLATION_PROMPTS["assemble_module"].format(
            module_name=module_name,
            translated_units=json.dumps(translated_units, indent=2),
            module_info=json.dumps(module_info, indent=2),
            reference_pattern=reference_pattern,
        )
        
        # Query LLM for assembly
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATION_PROMPTS["system"],
            max_tokens=self.max_tokens,
        )
        
        # Parse response
        return self._parse_translation_response(response, module_name, module_info)
    
    def _translate_module_legacy(
        self,
        module_name: str,
        fortran_code: str,
        module_info: Dict[str, Any],
        reference_pattern: str,
        output_dir: Optional[Path] = None,
    ) -> TranslationResult:
        """
        Legacy translation method (translate full module at once).
        Used as fallback when no translation units available.
        
        Args:
            module_name: Module name
            fortran_code: Full Fortran source
            module_info: Module information
            reference_pattern: Reference pattern
            output_dir: Optional output directory
            
        Returns:
            TranslationResult
        """
        # Build enhanced context
        enhanced_context = self._build_enhanced_context(module_name, module_info)
        
        # Build translation prompt
        prompt = TRANSLATION_PROMPTS["translate_module"].format(
            module_name=module_name,
            fortran_code=fortran_code,
            module_info=json.dumps(module_info, indent=2),
            enhanced_context=json.dumps(enhanced_context, indent=2),
            reference_pattern=reference_pattern,
        )
        
        console.print("[cyan]Generating JAX translation (legacy mode)...[/cyan]")
        
        # Query Claude for translation
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATION_PROMPTS["system"],
            max_tokens=self.max_tokens,
        )
        
        # Parse response into code files
        return self._parse_translation_response(response, module_name)
    
    def _get_module_dependencies(self, module_name: str) -> Dict[str, Any]:
        """
        Get dependencies for a module.
        
        Args:
            module_name: Module name
            
        Returns:
            Dictionary with uses and used_by lists
        """
        deps = {"uses": [], "used_by": []}
        
        if self.analysis_results:
            all_deps = self.analysis_results.get("parsing", {}).get("dependencies", {})
            if module_name in all_deps:
                deps["uses"] = all_deps[module_name]
            
            # Find who uses this module
            deps["used_by"] = [
                mod for mod, mod_deps in all_deps.items() 
                if module_name in mod_deps
            ]
        
        return deps
    
    def _extract_source_directory(self, file_path: str) -> str:
        """
        Extract the source directory from a Fortran file path.
        
        Args:
            file_path: Path to the Fortran file
            
        Returns:
            Source directory name (e.g., 'clm_src_main', 'multilayer_canopy')
        """
        path_obj = Path(file_path)
        parts = path_obj.parts
        
        # Look for known source directories
        known_dirs = [
            'clm_src_main', 'clm_src_biogeophys', 'clm_src_utils', 
            'multilayer_canopy', 'offline_driver', 'cime_src_share_util', 'clm_src_cpl'
        ]
        
        for part in parts:
            if part in known_dirs:
                return part
        
        # Default fallback
        return 'clm_src_main'

    def _remap_fortran_path(self, original_path: str) -> Path:
        """
        Remap Fortran file path from JSON to current project structure.
        
        Args:
            original_path: Original path from JSON
            
        Returns:
            Remapped path
        """
        path_obj = Path(original_path)
        
        # Try to check if path exists, but handle permission errors
        try:
            if path_obj.exists():
                return path_obj
        except (PermissionError, OSError):
            # Path exists but we don't have permission - need to remap
            pass
            
        # If we have a fortran_root override, try to remap relative to it
        if self.fortran_root:
            # Extract relative path from CLM-ml_v1 onwards
            parts = path_obj.parts
            for i, part in enumerate(parts):
                if 'CLM' in part or 'clm' in part:
                    relative_parts = parts[i+1:]  # Skip the CLM-ml_v1 part  
                    remapped = self.fortran_root / Path(*relative_parts)
                    try:
                        if remapped.exists():
                            return remapped
                    except (PermissionError, OSError):
                        pass
            
            # Try just the filename
            filename_path = self.fortran_root / path_obj.name
            if filename_path.exists():
                return filename_path
        
        # Fall back to original path
        return path_obj
    
    def _load_json(self, json_path: Path) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def _extract_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract module-specific information from analysis results.
        
        Args:
            module_name: Name of module to extract (case-insensitive)
            
        Returns:
            Dictionary with module information or None if not found
        """
        if not self.analysis_results:
            return None
        
        # Search in parsing.modules (case-insensitive)
        modules = self.analysis_results.get("parsing", {}).get("modules", {})
        for mod_name, mod_data in modules.items():
            if mod_name.lower() == module_name.lower():
                return mod_data
        
        return None
    
    def _build_enhanced_context(self, module_name: str, module_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build enhanced context for translation from JSON files.
        
        Args:
            module_name: Name of module being translated
            module_info: Module information from analysis_results.json
            
        Returns:
            Dictionary with enhanced context including dependencies, translation units, etc.
        """
        context = {
            "module_name": module_name,
            "dependencies": {},
            "translation_units": [],
            "complexity_info": {},
            "recommendations": [],
        }
        
        # Extract dependencies
        if self.analysis_results:
            deps = self.analysis_results.get("parsing", {}).get("dependencies", {})
            if module_name in deps:
                context["dependencies"]["uses"] = deps[module_name]
            
            # Find who uses this module
            context["dependencies"]["used_by"] = [
                mod for mod, mod_deps in deps.items() 
                if module_name in mod_deps
            ]
        
        # Extract translation units for this module
        if self.translation_units:
            units = self.translation_units.get("translation_units", [])
            module_units = [
                unit for unit in units 
                if unit.get("module_name", "").lower() == module_name.lower()
            ]
            context["translation_units"] = module_units
            
            # Calculate complexity summary
            if module_units:
                complexities = [u.get("complexity_score", 0) for u in module_units]
                efforts = [u.get("estimated_effort", "unknown") for u in module_units]
                context["complexity_info"] = {
                    "total_units": len(module_units),
                    "avg_complexity": sum(complexities) / len(complexities) if complexities else 0,
                    "max_complexity": max(complexities) if complexities else 0,
                    "effort_breakdown": {
                        "low": efforts.count("low"),
                        "medium": efforts.count("medium"),
                        "high": efforts.count("high"),
                    },
                    "has_split_functions": any(u.get("unit_type") == "inner" for u in module_units),
                }
        
        # Extract any module-specific recommendations
        if self.analysis_results:
            recs = self.analysis_results.get("recommendations", {})
            context["recommendations"] = [
                rec for rec in recs.get("translation_strategy", [])
            ]
        
        return context
    
    def _load_reference_patterns(self) -> Dict[str, str]:
        """
        Load reference patterns from existing jax-ctsm code.
        
        Returns:
            Dictionary of reference patterns
        """
        patterns = {}
        
        if self.jax_ctsm_dir and self.jax_ctsm_dir.exists():
            # Load maintenance respiration as main reference
            mr_file = self.jax_ctsm_dir / "src/jax_ctsm/physics/maintenance_respiration.py"
            if mr_file.exists():
                with open(mr_file, 'r') as f:
                    patterns["maintenance_respiration"] = f.read()
            
            # Load parameter example
            params_file = self.jax_ctsm_dir / "src/jax_ctsm/params/respiration.py"
            if params_file.exists():
                with open(params_file, 'r') as f:
                    patterns["params_example"] = f.read()
            
            # Load hierarchy example
            hierarchy_file = self.jax_ctsm_dir / "src/jax_ctsm/core/hierarchy.py"
            if hierarchy_file.exists():
                with open(hierarchy_file, 'r') as f:
                    patterns["hierarchy_example"] = f.read()
        
        return patterns
    
    def _get_reference_pattern(self) -> str:
        """
        Get reference pattern for translation.
        
        Returns:
            Reference pattern code
        """
        if "maintenance_respiration" in self.reference_patterns:
            # Return abbreviated version (first 100 lines as example)
            full_code = self.reference_patterns["maintenance_respiration"]
            lines = full_code.split('\n')[:100]
            return '\n'.join(lines) + "\n\n# ... (abbreviated for context) ..."
        else:
            return """# Reference pattern not available
# The translator will use general JAX best practices:
# - Pure functions with type hints
# - NamedTuples for state
# - Vectorized operations
# - Google-style docstrings
"""
    
    def _parse_translation_response(
        self,
        response: str,
        module_name: str,
        module_info: Optional[Dict[str, Any]] = None,
    ) -> TranslationResult:
        """
        Parse Claude's translation response into code files.
        
        Args:
            response: Claude's response
            module_name: Name of module being translated
            
        Returns:
            TranslationResult with parsed code
        """
        # Extract code blocks from response
        physics_code = ""
        params_code = None  # No longer generating separate params files
        test_code = None
        notes = ""
        
        # Split response into sections
        sections = response.split("```python")
        
        if len(sections) > 1:
            # First (and expected only) Python block contains the complete module
            physics_end = sections[1].find("```")
            physics_code = sections[1][:physics_end].strip()
            
            # Log warning if unexpected additional code blocks are found
            if len(sections) > 2:
                console.print("[yellow]⚠ Warning: Found multiple Python code blocks. Using first block only.[/yellow]")
        
        # Extract any markdown notes (text before first code block)
        if "```" in response:
            notes = response[:response.find("```")].strip()
        
        return TranslationResult(
            module_name=module_name,
            physics_code=physics_code if physics_code else response,
            source_directory=self._extract_source_directory(module_info.get('file_path', '')) if module_info else 'clm_src_main',
            params_code=params_code,
            test_code=test_code,
            translation_notes=notes,
        )
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from response (handles markdown code blocks).
        
        Args:
            response: Claude's response
            
        Returns:
            Extracted code
        """
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        else:
            return response.strip()

