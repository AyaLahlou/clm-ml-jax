"""
Prompts for the Repair Agent.

This agent focuses on fixing failed Python/JAX translations by:
1. Analyzing test failures and error messages
2. Comparing with original Fortran code
3. Identifying root causes
4. Generating corrected Python code
"""

REPAIR_PROMPTS = {
    "system": """You are an expert Repair Agent specializing in debugging and fixing Fortran-to-JAX translations.

Your expertise:
- Deep understanding of Fortran semantics and Python/JAX
- Debugging complex numerical computing code
- Root cause analysis of test failures
- Identifying translation errors and logic bugs

Your responsibilities:
1. Analyze test failures and error messages carefully
2. Compare failed Python code with original Fortran implementation
3. Identify root causes (indexing errors, type mismatches, logic bugs, etc.)
4. Generate corrected Python code that passes all tests
5. Provide clear root cause analysis reports

You follow these principles:
- Preserve scientific accuracy - physics must match exactly
- Use JAX best practices (pure functions, immutable state, JIT-compatible code)
- Fix only what's broken - don't refactor unrelated code
- Provide clear explanations of what went wrong and how you fixed it
- Be thorough in your analysis""",

    "analyze_failure": """Analyze the test failure and identify the root cause.

**Fortran Subroutine (Original):**
{fortran_code}

**Failed Python Function:**
{python_code}

**Test Report:**
{test_report}

Please analyze:
1. What specific tests failed?
2. What are the error messages/failure modes?
3. How does the Python code differ from the Fortran semantically?
4. What is the root cause of the failure?

Provide your analysis in JSON format:
{{
    "failed_tests": ["list of failed test names"],
    "error_summary": "brief summary of errors",
    "root_causes": [
        {{
            "issue": "description of the issue",
            "location": "where in the code",
            "severity": "critical|major|minor",
            "explanation": "detailed explanation"
        }}
    ],
    "required_fixes": [
        "list of required fixes in priority order"
    ]
}}
""",

    "generate_fix": """Generate a corrected version of the Python function based on the root cause analysis.

**Fortran Subroutine (Reference):**
{fortran_code}

**Failed Python Function:**
{python_code}

**Root Cause Analysis:**
{root_cause_analysis}

**Test Report:**
{test_report}

Please generate the corrected Python function that addresses all identified issues.

Requirements:
1. Fix all identified root causes
2. Preserve the original function structure where possible
3. Maintain JAX compatibility (pure functions, no in-place operations)
4. Add comments explaining critical fixes
5. Ensure type hints are correct
6. Follow Python best practices

Provide ONLY the corrected Python function code (no JSON, no explanations).
The code should be complete and ready to save to a .py file.
""",

    "root_cause_report": """Generate a comprehensive root cause analysis report.

**Original Fortran Code:**
{fortran_code}

**Failed Python Code:**
{failed_python_code}

**Corrected Python Code:**
{corrected_python_code}

**Failure Analysis:**
{failure_analysis}

**Test Results (After Fix):**
{test_results}

Please generate a comprehensive root cause analysis report in markdown format covering:

1. **Executive Summary**
   - Brief overview of the issue
   - Impact and severity

2. **Failure Analysis**
   - What tests failed
   - Error messages and symptoms
   - When/where the failure occurred

3. **Root Cause Identification**
   - Detailed analysis of each root cause
   - Why the original translation was incorrect
   - Comparison with Fortran semantics

4. **Fix Implementation**
   - What changes were made
   - Why these changes fix the issue
   - Code snippets showing before/after

5. **Test Results**
   - Test results after fix
   - Verification that issue is resolved

6. **Lessons Learned**
   - Key takeaways for future translations
   - Common pitfalls to avoid

Format the report in clear, readable markdown.
""",

    "verify_fix": """Verify if the corrected code likely resolves the issues.

**Original Failure Analysis:**
{failure_analysis}

**Corrected Python Code:**
{corrected_code}

**Required Fixes:**
{required_fixes}

Please verify:
1. Does the corrected code address all root causes?
2. Are there any remaining issues?
3. Is the code JAX-compatible?
4. Does it follow best practices?

Provide verification results in JSON format:
{{
    "all_issues_addressed": true/false,
    "addressed_issues": ["list of fixed issues"],
    "remaining_concerns": ["any remaining issues"],
    "confidence_level": "high|medium|low",
    "recommendations": ["any additional recommendations"]
}}
""",
}

