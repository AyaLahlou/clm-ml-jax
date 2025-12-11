"""
Repair Agent for fixing failed JAX translations.

This agent focuses on debugging and fixing failed Python/JAX translations:
1. Analyzes test failures and error messages
2. Compares with original Fortran code
3. Identifies root causes
4. Generates corrected Python code
5. Iteratively fixes until tests pass
6. Provides comprehensive root cause analysis
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from jax_agents.base_agent import BaseAgent
from jax_agents.utils.config_loader import get_llm_config
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class RepairResult:
    """Complete result of repair process."""
    module_name: str
    original_python_code: str
    corrected_python_code: str
    root_cause_analysis: str
    failure_analysis: Dict[str, Any]
    iterations: int
    final_test_report: str
    all_tests_passed: bool
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Save all repair artifacts to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save corrected Python code
        corrected_path = output_dir / f"{self.module_name}_corrected.py"
        with open(corrected_path, 'w') as f:
            f.write(self.corrected_python_code)
        saved_files["corrected_code"] = corrected_path
        console.print(f"[green]✓ Saved corrected code to {corrected_path}[/green]")
        
        # Save root cause analysis
        rca_path = output_dir / f"root_cause_analysis_{self.module_name}.md"
        with open(rca_path, 'w') as f:
            f.write(self.root_cause_analysis)
        saved_files["root_cause_analysis"] = rca_path
        console.print(f"[green]✓ Saved root cause analysis to {rca_path}[/green]")
        
        # Save failure analysis
        failure_path = output_dir / f"failure_analysis_{self.module_name}.json"
        with open(failure_path, 'w') as f:
            json.dump(self.failure_analysis, f, indent=2)
        saved_files["failure_analysis"] = failure_path
        console.print(f"[green]✓ Saved failure analysis to {failure_path}[/green]")
        
        # Save final test report
        test_report_path = output_dir / f"final_test_report_{self.module_name}.txt"
        with open(test_report_path, 'w') as f:
            f.write(self.final_test_report)
        saved_files["test_report"] = test_report_path
        console.print(f"[green]✓ Saved final test report to {test_report_path}[/green]")
        
        return saved_files


class RepairAgent(BaseAgent):
    """
    Agent for repairing failed JAX translations.
    
    Responsibilities:
    - Analyze test failures and identify root causes
    - Generate corrected Python code
    - Run tests and verify fixes
    - Iterate until tests pass (or max iterations reached)
    - Generate comprehensive root cause analysis reports
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_repair_iterations: int = 5,
    ):
        """
        Initialize Repair Agent.
        
        Args:
            model: Claude model to use (defaults to config.yaml)
            temperature: Sampling temperature (defaults to config.yaml)
            max_tokens: Maximum tokens in response (defaults to config.yaml)
            max_repair_iterations: Maximum number of repair iterations
        """
        llm_config = get_llm_config()
        
        super().__init__(
            name="RepairAgent",
            role="JAX translation debugger and fixer",
            model=model or llm_config.get("model", "claude-sonnet-4-5"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 32000),
        )
        
        self.max_repair_iterations = max_repair_iterations
    
    def repair_translation(
        self,
        module_name: str,
        fortran_code: str,
        failed_python_code: str,
        test_report: str,
        test_file_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> RepairResult:
        """
        Repair a failed Python/JAX translation.
        
        Args:
            module_name: Name of the module being repaired
            fortran_code: Original Fortran subroutine/function
            failed_python_code: Failed Python translation
            test_report: Test report showing failures
            test_file_path: Optional path to pytest file for running tests
            output_dir: Optional directory to save outputs
            
        Returns:
            RepairResult with corrected code and analysis
        """
        console.print(f"\n[bold cyan]🔧 Repairing {module_name}[/bold cyan]")
        
        current_python_code = failed_python_code
        iteration = 0
        all_tests_passed = False
        
        # Step 1: Initial failure analysis
        console.print("[cyan]Step 1: Analyzing test failures...[/cyan]")
        failure_analysis = self._analyze_failure(
            fortran_code, current_python_code, test_report, module_name
        )
        console.print(f"[yellow]Found {len(failure_analysis.get('root_causes', []))} root cause(s)[/yellow]")
        
        # Step 2: Iterative repair loop
        test_results_history = [{"iteration": 0, "report": test_report}]
        
        while iteration < self.max_repair_iterations and not all_tests_passed:
            iteration += 1
            console.print(f"\n[bold cyan]Iteration {iteration}/{self.max_repair_iterations}[/bold cyan]")
            
            # Generate fix
            console.print(f"[cyan]Generating fix for iteration {iteration}...[/cyan]")
            corrected_code = self._generate_fix(
                fortran_code,
                current_python_code,
                failure_analysis,
                test_report,
                module_name,
            )
            
            # Verify fix conceptually
            console.print(f"[cyan]Verifying fix...[/cyan]")
            verification = self._verify_fix(
                failure_analysis,
                corrected_code,
                failure_analysis.get("required_fixes", []),
            )
            
            if verification.get("confidence_level") == "low":
                console.print("[yellow]⚠ Low confidence in fix, but proceeding with testing[/yellow]")
            
            # Run tests if test file is provided
            if test_file_path:
                console.print(f"[cyan]Running tests...[/cyan]")
                test_success, new_test_report = self._run_tests(
                    corrected_code, test_file_path, module_name
                )
                
                test_results_history.append({
                    "iteration": iteration,
                    "report": new_test_report
                })
                
                if test_success:
                    console.print(f"[green]✓ All tests passed on iteration {iteration}![/green]")
                    all_tests_passed = True
                    current_python_code = corrected_code
                    test_report = new_test_report
                    break
                else:
                    console.print(f"[yellow]Tests still failing, analyzing new failures...[/yellow]")
                    # Re-analyze with new test report
                    failure_analysis = self._analyze_failure(
                        fortran_code, corrected_code, new_test_report, module_name
                    )
                    current_python_code = corrected_code
                    test_report = new_test_report
            else:
                # No test file, just accept the fix
                console.print("[yellow]⚠ No test file provided, accepting fix without verification[/yellow]")
                current_python_code = corrected_code
                all_tests_passed = True
                break
        
        if not all_tests_passed:
            console.print(f"[red]⚠ Maximum iterations ({self.max_repair_iterations}) reached[/red]")
            console.print("[yellow]Some tests may still be failing[/yellow]")
        
        # Step 3: Generate comprehensive root cause analysis report
        console.print("[cyan]Generating root cause analysis report...[/cyan]")
        rca_report = self._generate_root_cause_report(
            fortran_code,
            failed_python_code,
            current_python_code,
            failure_analysis,
            test_report,
            module_name,
            test_results_history,
        )
        
        result = RepairResult(
            module_name=module_name,
            original_python_code=failed_python_code,
            corrected_python_code=current_python_code,
            root_cause_analysis=rca_report,
            failure_analysis=failure_analysis,
            iterations=iteration,
            final_test_report=test_report,
            all_tests_passed=all_tests_passed,
        )
        
        console.print(f"[green]✓ Repair process complete![/green]")
        
        if output_dir:
            result.save(output_dir)
        
        return result
    
    def _analyze_failure(
        self,
        fortran_code: str,
        python_code: str,
        test_report: str,
        module_name: str,
    ) -> Dict[str, Any]:
        """Analyze test failure and identify root causes."""
        from jax_agents.prompts.repair_prompts import REPAIR_PROMPTS
        
        prompt = REPAIR_PROMPTS["analyze_failure"].format(
            fortran_code=fortran_code,
            python_code=python_code,
            test_report=test_report,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=REPAIR_PROMPTS["system"],
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
            logger.error(f"Failed to parse failure analysis: {e}")
            logger.error(f"Response: {response}")
            # Return a basic structure if parsing fails
            return {
                "failed_tests": ["unknown"],
                "error_summary": "Failed to parse error analysis",
                "root_causes": [],
                "required_fixes": ["Review test report manually"]
            }
    
    def _generate_fix(
        self,
        fortran_code: str,
        python_code: str,
        failure_analysis: Dict[str, Any],
        test_report: str,
        module_name: str,
    ) -> str:
        """Generate corrected Python code."""
        from jax_agents.prompts.repair_prompts import REPAIR_PROMPTS
        
        prompt = REPAIR_PROMPTS["generate_fix"].format(
            fortran_code=fortran_code,
            python_code=python_code,
            root_cause_analysis=json.dumps(failure_analysis, indent=2),
            test_report=test_report,
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=REPAIR_PROMPTS["system"],
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
            return response.strip()
    
    def _verify_fix(
        self,
        failure_analysis: Dict[str, Any],
        corrected_code: str,
        required_fixes: List[str],
    ) -> Dict[str, Any]:
        """Verify if the corrected code addresses all issues."""
        from jax_agents.prompts.repair_prompts import REPAIR_PROMPTS
        
        prompt = REPAIR_PROMPTS["verify_fix"].format(
            failure_analysis=json.dumps(failure_analysis, indent=2),
            corrected_code=corrected_code,
            required_fixes=json.dumps(required_fixes, indent=2),
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=REPAIR_PROMPTS["system"],
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
            logger.error(f"Failed to parse verification: {e}")
            return {
                "all_issues_addressed": False,
                "confidence_level": "low",
                "recommendations": ["Manual review required"]
            }
    
    def _run_tests(
        self,
        python_code: str,
        test_file_path: Path,
        module_name: str,
    ) -> Tuple[bool, str]:
        """
        Run tests with the corrected Python code.
        
        Args:
            python_code: Corrected Python code
            test_file_path: Path to pytest file
            module_name: Module name
            
        Returns:
            Tuple of (success: bool, test_report: str)
        """
        try:
            # Create temporary file with corrected code
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, dir=test_file_path.parent
            ) as tmp_file:
                tmp_file.write(python_code)
                tmp_file_path = Path(tmp_file.name)
            
            try:
                # Run pytest
                result = subprocess.run(
                    ['pytest', str(test_file_path), '-v', '--tb=short'],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                
                test_report = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                success = result.returncode == 0
                
                return success, test_report
                
            finally:
                # Clean up temporary file
                tmp_file_path.unlink(missing_ok=True)
                
        except subprocess.TimeoutExpired:
            return False, "Error: Tests timed out after 5 minutes"
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False, f"Error running tests: {str(e)}"
    
    def _generate_root_cause_report(
        self,
        fortran_code: str,
        failed_python_code: str,
        corrected_python_code: str,
        failure_analysis: Dict[str, Any],
        final_test_report: str,
        module_name: str,
        test_results_history: List[Dict[str, Any]],
    ) -> str:
        """Generate comprehensive root cause analysis report."""
        from jax_agents.prompts.repair_prompts import REPAIR_PROMPTS
        
        # Build test results summary
        test_results_summary = "\n".join([
            f"Iteration {h['iteration']}: {'PASSED' if 'PASSED' in h['report'] or 'passed' in h['report'].lower() else 'FAILED'}"
            for h in test_results_history
        ])
        
        prompt = REPAIR_PROMPTS["root_cause_report"].format(
            fortran_code=fortran_code,
            failed_python_code=failed_python_code,
            corrected_python_code=corrected_python_code,
            failure_analysis=json.dumps(failure_analysis, indent=2),
            test_results=f"{test_results_summary}\n\nFinal Report:\n{final_test_report}",
        )
        
        response = self.query_claude(
            prompt=prompt,
            system_prompt=REPAIR_PROMPTS["system"],
        )
        
        return response

