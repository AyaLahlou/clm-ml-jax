"""
Test Agent for generating tests for JAX translations.

This agent focuses on Python/JAX test generation:
1. Analyzes Python function signatures
2. Generates comprehensive synthetic test data
3. Creates pytest files with multiple test scenarios
4. Provides test templates and documentation
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from jax_agents.base_agent import BaseAgent
from jax_agents.utils.config_loader import get_llm_config
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class TestGenerationResult:
    """Complete result of test generation."""
    module_name: str
    pytest_file: str
    test_data_file: str
    test_documentation: str
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Save all test artifacts to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save pytest file
        pytest_path = output_dir / f"test_{self.module_name}.py"
        with open(pytest_path, 'w') as f:
            f.write(self.pytest_file)
        saved_files["pytest"] = pytest_path
        console.print(f"[green]✓ Saved pytest file to {pytest_path}[/green]")
        
        # Save test data
        test_data_path = output_dir / f"test_data_{self.module_name}.json"
        with open(test_data_path, 'w') as f:
            f.write(self.test_data_file)
        saved_files["test_data"] = test_data_path
        console.print(f"[green]✓ Saved test data to {test_data_path}[/green]")
        
        # Save documentation
        docs_path = output_dir / f"test_documentation_{self.module_name}.md"
        with open(docs_path, 'w') as f:
            f.write(self.test_documentation)
        saved_files["documentation"] = docs_path
        console.print(f"[green]✓ Saved test documentation to {docs_path}[/green]")
        
        return saved_files
    
    def save_structured(self, project_root: Path, source_directory: str) -> Dict[str, Path]:
        """
        Save test artifacts to structured project layout.
        
        Args:
            project_root: Root directory of the project
            source_directory: Source directory name (e.g., 'clm_src_main')
        
        Returns:
            Dictionary mapping artifact type to saved file path
        """
        saved_files = {}
        
        # Save pytest file to tests/<source_directory>/
        tests_dir = project_root / "tests" / source_directory
        tests_dir.mkdir(parents=True, exist_ok=True)
        pytest_path = tests_dir / f"test_{self.module_name}.py"
        with open(pytest_path, 'w') as f:
            f.write(self.pytest_file)
        saved_files["pytest"] = pytest_path
        console.print(f"[green]✓ Saved pytest file to {pytest_path}[/green]")
        
        # Save test data to central tests/test_data/
        test_data_dir = project_root / "tests" / "test_data"
        test_data_dir.mkdir(parents=True, exist_ok=True)
        test_data_path = test_data_dir / f"test_data_{self.module_name}.json"
        with open(test_data_path, 'w') as f:
            f.write(self.test_data_file)
        saved_files["test_data"] = test_data_path
        console.print(f"[green]✓ Saved test data to {test_data_path}[/green]")
        
        # Save test documentation to docs/
        docs_dir = project_root / "CLM-ml_v1" / "docs" / "test_documentation"
        docs_dir.mkdir(parents=True, exist_ok=True)
        docs_path = docs_dir / f"test_documentation_{self.module_name}.md"
        with open(docs_path, 'w') as f:
            f.write(self.test_documentation)
        saved_files["documentation"] = docs_path
        console.print(f"[green]✓ Saved test documentation to {docs_path}[/green]")
        
        return saved_files


class TestAgent(BaseAgent):
    """
    Agent for generating comprehensive tests for JAX translations.
    
    Responsibilities:
    - Parse Python/JAX function signatures
    - Generate synthetic test data covering edge cases
    - Create pytest files with fixtures and parametrized tests
    - Generate test documentation and examples
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize Test Agent.
        
        Args:
            model: Claude model to use (defaults to config.yaml)
            temperature: Sampling temperature (defaults to config.yaml)
            max_tokens: Maximum tokens in response (defaults to config.yaml)
        """
        llm_config = get_llm_config()
        
        super().__init__(
            name="TestAgent",
            role="JAX test generator",
            model=model or llm_config.get("model", "claude-sonnet-4-5"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 32000),
        )
    
    def generate_tests(
        self,
        module_name: str,
        python_code: str,
        output_dir: Optional[Path] = None,
        num_test_cases: int = 10,
        include_edge_cases: bool = True,
        include_performance_tests: bool = False,
        source_directory: str = None,
    ) -> TestGenerationResult:
        """
        Generate comprehensive test suite for a Python/JAX module.
        
        Args:
            module_name: Name of the module being tested
            python_code: Translated Python function code
            output_dir: Optional directory to save outputs
            num_test_cases: Number of synthetic test cases to generate
            include_edge_cases: Include edge case tests (zeros, negatives, etc.)
            include_performance_tests: Include performance/benchmark tests
            source_directory: Source directory for imports (e.g., 'clm_src_biogeophys')
            
        Returns:
            TestGenerationResult with pytest file, test data, and documentation
        """
        console.print(f"\n[bold cyan]🧪 Generating tests for {module_name}[/bold cyan]")
        
        # Step 1: Analyze Python function signature
        console.print("[cyan]Step 1/3: Analyzing Python function...[/cyan]")
        python_sig = self._analyze_python_signature(python_code, module_name)
        
        # Step 2: Generate comprehensive test data
        console.print("[cyan]Step 2/3: Generating test data...[/cyan]")
        test_data = self._generate_test_data(
            python_sig, num_test_cases, include_edge_cases
        )
        
        # Step 3: Generate pytest file
        console.print("[cyan]Step 3/3: Generating pytest file...[/cyan]")
        pytest_file = self._generate_pytest(
            module_name, python_sig, test_data, include_performance_tests, source_directory
        )
        
        # Generate documentation
        test_docs = self._generate_documentation(
            module_name, python_sig, test_data, num_test_cases
        )
        
        result = TestGenerationResult(
            module_name=module_name,
            pytest_file=pytest_file,
            test_data_file=json.dumps(test_data, indent=2),
            test_documentation=test_docs,
        )
        
        console.print(f"[green]✓ Test generation complete![/green]")
        
        if output_dir:
            result.save(output_dir)
        
        return result
    
    
    def _analyze_python_signature(
        self, python_code: str, module_name: str
    ) -> Dict[str, Any]:
        """Analyze Python function signature and parameters."""
        from jax_agents.prompts.test_prompts_simplified import TEST_PROMPTS
        
        prompt = TEST_PROMPTS["analyze_python_signature"].format(
            python_code=python_code,
            module_name=module_name,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TEST_PROMPTS["system"],
        )
        
        # Parse JSON response
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Python signature: {e}")
            logger.error(f"Response: {response}")
            raise
    
    def _generate_test_data(
        self,
        python_sig: Dict[str, Any],
        num_cases: int,
        include_edge_cases: bool,
    ) -> Dict[str, Any]:
        """Generate comprehensive synthetic test data."""
        from jax_agents.prompts.test_prompts_simplified import TEST_PROMPTS
        
        prompt = TEST_PROMPTS["generate_test_data"].format(
            python_signature=json.dumps(python_sig, indent=2),
            num_cases=num_cases,
            include_edge_cases=include_edge_cases,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TEST_PROMPTS["system"],
        )
        
        # Parse JSON response
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse test data: {e}")
            logger.error(f"Response: {response}")
            raise
    
    
    def _generate_pytest(
        self,
        module_name: str,
        python_sig: Dict[str, Any],
        test_data: Dict[str, Any],
        include_performance: bool,
        source_directory: str = None,
    ) -> str:
        """Generate comprehensive pytest file."""
        from jax_agents.prompts.test_prompts_simplified import TEST_PROMPTS
        
        # Default source directory if not provided
        if not source_directory:
            source_directory = "clm_src_main"  # Default fallback
        
        prompt = TEST_PROMPTS["generate_pytest"].format(
            module_name=module_name,
            source_directory=source_directory,
            python_signature=json.dumps(python_sig, indent=2),
            test_data=json.dumps(test_data, indent=2),
            include_performance=include_performance,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TEST_PROMPTS["system"],
        )
        
        # Extract Python code
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        else:
            return response
    
    def _generate_documentation(
        self,
        module_name: str,
        python_sig: Dict[str, Any],
        test_data: Dict[str, Any],
        num_cases: int,
    ) -> str:
        """Generate test documentation."""
        from jax_agents.prompts.test_prompts_simplified import TEST_PROMPTS
        
        prompt = TEST_PROMPTS["generate_documentation"].format(
            module_name=module_name,
            python_signature=json.dumps(python_sig, indent=2),
            test_data_summary=json.dumps({
                "num_cases": num_cases,
                "test_types": list(set(tc.get("metadata", {}).get("type", "nominal") 
                                      for tc in test_data.get("test_cases", []))),
            }, indent=2),
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=TEST_PROMPTS["system"],
        )
        
        return response

