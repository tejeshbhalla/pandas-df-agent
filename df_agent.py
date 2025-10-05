from pydantic_ai import Agent, RunContext, ModelRetry
from openai_model import model
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
import numpy as np
import asyncio
from typing import List, Dict, Optional
import io
import sys

@dataclass
class DataFrameDeps:
    model_config = ConfigDict(arbitrary_types_allowed=True)
    df: pd.DataFrame
    sheet_name: str = "Sheet1"
    persistent_namespace: Dict = field(default_factory=dict)  # Persistent namespace for variables

@dataclass
class SheetPreviewDeps:
    sheet_previews: Dict[str, str]
    original_question: str

@dataclass
class MergerDeps:
    sheet_answers: Dict[str, str]
    original_question: str

# Pydantic models for structured outputs
class SheetRelevance(BaseModel):
    """Structured output for sheet relevance determination."""
    sheet_name: str = Field(description="The name of the sheet being evaluated")
    is_relevant: bool = Field(description="Whether this sheet is relevant to answering the question")
    reason: str = Field(description="Brief explanation of why this sheet is or isn't relevant")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)

class MultiSheetRelevance(BaseModel):
    """Structured output for analyzing multiple sheets at once."""
    relevant_sheets: List[str] = Field(description="List of sheet names that are relevant to the question")
    irrelevant_sheets: List[str] = Field(description="List of sheet names that are not relevant")
    analysis_summary: str = Field(description="Brief summary of which sheets to analyze and why")

# Sheet relevance agent with structured output
relevance_agent = Agent(
    model=model,
    output_type=MultiSheetRelevance,
    system_prompt="""
    You are an expert at quickly determining which Excel sheets are relevant for answering a data analysis question.
    
    You will be given:
    1. A question to answer
    2. Preview data (first 10 rows) from multiple sheets
    
    Your task is to identify which sheets contain data relevant to answering the question.
    
    ## Analysis Strategy:
    
    1. **Examine column names** - Do they relate to the question?
    2. **Check data content** - Does the preview show relevant information?
    3. **Look for key metrics** - Are the metrics mentioned in the question present?
    4. **Identify data types** - Financial data? Sales data? Time series?
    
    ## Decision Guidelines:
    
    - Mark as **RELEVANT** if:
      - Column names match key terms in the question
      - Data types align with what's needed
      - The sheet appears to contain the required metrics
    
    - Mark as **IRRELEVANT** if:
      - No overlap with question terms
      - Wrong type of data entirely
      - Metadata or summary sheet with no raw data
    
    ## Be Decisive:
    
    - It's better to include a potentially relevant sheet than to miss it
    - If uncertain, mark as relevant (we can filter later)
    - Provide clear reasoning for each decision
    
    Return your analysis in the structured format with relevant_sheets, irrelevant_sheets, and a summary.
    """,
    retries=3
)

# Main data analysis agent
data_agent = Agent(
    model=model,
    system_prompt="""
You are a simple data analysis expert who specializes in thoroughly analyzing pandas DataFrames to provide accurate, detailed answers.

## Available Tools and Packages

You have access to the following Python packages and tools (already imported, no need to import them):
- **pandas (pd)**: For DataFrame operations, data manipulation, and analysis
- **numpy (np)**: For numerical operations, mathematical functions, and array handling
- **Python built-ins**: All standard Python functions (len, sum, min, max, round, etc.)
- **df_query tool**: Execute Python code on the DataFrame with persistent state across queries
- **DataFrame persistence**: Any modifications you make to the DataFrame will persist across all queries

You can use any of these packages in your analysis. For example:
- `df.shape`, `df.columns`, `df.dtypes` for basic info
- `df.head()`, `df.tail()`, `df.describe()` for data exploration
- `pd.to_numeric()`, `df.str.replace()` for data cleaning
- `df.groupby()`, `df.agg()` for aggregations
- `np.sum()`, `np.mean()`, `np.std()` for statistical calculations

## Your Core Approach

**Be Thorough**: Always start by deeply exploring the data to understand its structure, identify issues, and ensure accuracy.
**Be Automatic**: Use your expertise to automatically detect and handle common data problems like totals, subtotals, formatting issues, and missing values.
**Be Detailed**: Provide comprehensive answers with exact information and clear traces of what you discovered and how you handled it.

## Key Principles

1. **Data Modifications Persist**: Any changes you make to the DataFrame will persist across all future queries in this conversation. Use this to your advantage - clean once, analyze many times.

2. **Start with Deep Exploration**: Before answering any question, thoroughly explore the data to understand:
   - Overall structure and shape
   - Column names and data types
   - Data quality issues (missing values, formatting problems)
   - Hidden totals, subtotals, or aggregated rows
   - Any anomalies or patterns

3. **Automatically Detect Issues**: Use your expertise to identify common problems:
   - Rows containing totals, subtotals, or summary data that should be excluded
   - String-formatted numbers with currency symbols, commas, percentages
   - Missing values represented as placeholders (--, N/A, blanks)
   - Inconsistent data types or formatting

4. **Clean Systematically**: Once you identify issues, clean the data systematically:
   - Remove total/subtotal rows that would skew analysis
   - Convert string-formatted numbers to proper numeric types
   - Handle missing values appropriately
   - Ensure data consistency

5. **Verify Before Analysis**: Always verify your cleaned data before performing final calculations to ensure accuracy.

6. **Cross-Verify Results**: When possible, verify your final answer using 2 or more different calculation methods to ensure consistency and accuracy. For example:
   - Calculate totals by summing individual rows vs. using built-in functions
   - Verify averages by manual calculation vs. using pandas methods
   - Cross-check percentages using different approaches
   - Use alternative data sources or columns when available

7. **Provide Detailed Traces**: In your final answer, include:
   - What issues you discovered in the data
   - How you cleaned or handled each issue
   - What data was removed or modified
   - The exact methodology used for calculations
   - Multiple verification methods used (if applicable)
   - Confidence level in your results

## Response Structure

Start your response by stating which sheet you're analyzing, then:

1. **Data Exploration Summary**: Brief overview of what you found
2. **Issues Identified**: List of problems discovered (totals, formatting, missing values, etc.)
3. **Data Cleaning Actions**: What you did to address each issue
4. **Final Analysis**: Your answer to the question with exact numbers
5. **Verification Methods**: How you cross-verified the results using multiple approaches
6. **Trace of Actions**: Detailed summary of what was removed, modified, or calculated

## Best Practices

- Combine multiple exploration steps in single queries for efficiency
- Use pattern matching and conditional logic to detect anomalies
- Be systematic in your approach - explore, clean, verify, analyze
- Always state the sheet name at the beginning of your response
- Provide exact numbers and clear methodology
- Cross-verify results using multiple calculation methods when possible
- Explain any assumptions or limitations in your analysis

Remember: Your goal is to provide accurate, detailed answers by being a thorough and systematic data analysis expert. Take the time needed to properly understand and clean the data before providing your final answer.
    """,
    deps_type=DataFrameDeps,
    retries=10
)

# Merger agent to combine answers from all sheets
merger_agent = Agent(
    model=model,
    system_prompt="""
    You are an expert at synthesizing and merging analysis results from multiple data sources.
    
    Your task is to take individual analysis results from multiple Excel sheets and create a 
    comprehensive, unified answer that:
    
    1. **Synthesizes information** across all sheets
    2. **Identifies patterns** and commonalities
    3. **Highlights differences** or notable variations between sheets
    4. **Provides a clear summary** that directly answers the original question
    5. **Organizes information logically** (by sheet, by metric, by finding - whatever makes most sense)
    
    ## Key Notes:
     - You will be given df.head() for each sheet along with its answer from it you can analyze if all sheets are continuous or not and then accumalate answers accordingly for vagueness cite all possible answer you think are valid
     - You need to be smart while synthesizing answers if results are from multiple sheets and it seems like the final sheet is the accumalation of all the other sheets it has to take that number 
     - In case all sheets have answers and you cannot see any accumalation sheet try to accumalate answer from all sheets and see which value was the worst accross all sheets and take that number as the final answer
     
    ## Response Structure:
    
    ### Executive Summary
    - Direct answer to the question with key numbers/findings
    - Overall insights across all sheets
    
    ### Sheet-by-Sheet Breakdown (if relevant)
    - Key findings from each sheet
    - Important metrics or data points
    
    ### Comparative Analysis (if applicable)
    - How sheets compare to each other
    - Trends or patterns across sheets
    
    ### Conclusion
    - Final takeaways
    - Any caveats or notes about the data
    
    Keep your response clear, well-organized, and focused on answering the original question.
    """,
    deps_type=MergerDeps,
    retries=3
)

@data_agent.tool
async def df_query(ctx: RunContext[DataFrameDeps], query: str) -> str:
    """A tool for running queries on the pandas.DataFrame.
    
    **IMPORTANT: All DataFrame modifications and variables persist across tool calls.**
    
    Examples:
    - import pandas as pd
    - import numpy as np  
    - df.shape
    - df.columns.tolist()
    - df.head(10)
    - df = df.iloc[:-1]  # Remove last row (persists!)
    - suspicious_rows = df['col'].isna()  # Variable persists!
    - df['col'] = pd.to_numeric(df['col'].str.replace(',', '', regex=False), errors='coerce')
    - df['col'].mean()
    """
    print(f'\n[{ctx.deps.sheet_name}] Running: `{query}`')
    
    # Capture print output
    output_capture = io.StringIO()
    
    try:
        df = ctx.deps.df
        
        # Custom print function that captures output
        def custom_print(*args, **kwargs):
            print(*args, **kwargs, file=output_capture)
        
        # Build namespace with all basic Python built-ins and libraries
        namespace = {
            # DataFrame and libraries (always available)
            'df': df,
            'pd': pd,
            'np': np,
            'print': custom_print,  # Redirect print to capture output
            
            # Add all Python built-ins that might be needed
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'int': int,
            'isinstance': isinstance,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'range': range,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Merge with persistent namespace (persistent values take precedence)
        namespace.update(ctx.deps.persistent_namespace)
        
        # Check if this is an assignment or expression
        is_assignment = '=' in query and not any(op in query for op in ['==', '!=', '<=', '>=', '=>'])
        
        # Create a restricted builtins that still allows imports
        restricted_builtins = {
            '__import__': __import__,  # Allow imports
            '__build_class__': __build_class__,
            '__name__': __name__,
        }
        
        if is_assignment or '\n' in query or 'import' in query:
            # Execute the code with import capability
            exec(compile(query, '<string>', 'exec'), 
                 {'__builtins__': restricted_builtins}, 
                 namespace)
            
            # Update the DataFrame if it was modified
            ctx.deps.df = namespace['df']
            
            # Update persistent namespace with any new or modified variables
            # Exclude the base items we always provide
            exclude_keys = {'df', 'pd', 'np', 'print', 'abs', 'all', 'any', 'bool', 'dict', 
                          'enumerate', 'filter', 'float', 'int', 'isinstance', 'len', 'list',
                          'map', 'max', 'min', 'range', 'round', 'set', 'sorted', 'str',
                          'sum', 'tuple', 'type', 'zip', 'True', 'False', 'None'}
            
            for key, value in namespace.items():
                if key not in exclude_keys and not key.startswith('__'):
                    ctx.deps.persistent_namespace[key] = value
            
            # Get captured print output
            printed_output = output_capture.getvalue()
            
            if printed_output:
                result_str = printed_output
            else:
                result_str = f"Executed successfully. DataFrame shape: {namespace['df'].shape}"
        else:
            # Evaluate the expression with import capability
            result = eval(compile(query, '<string>', 'eval'), 
                         {'__builtins__': restricted_builtins}, 
                         namespace)
            
            # Update persistent namespace with any changes made during eval
            ctx.deps.df = namespace['df']
            
            exclude_keys = {'df', 'pd', 'np', 'print', 'abs', 'all', 'any', 'bool', 'dict', 
                          'enumerate', 'filter', 'float', 'int', 'isinstance', 'len', 'list',
                          'map', 'max', 'min', 'range', 'round', 'set', 'sorted', 'str',
                          'sum', 'tuple', 'type', 'zip', 'True', 'False', 'None'}
            
            for key, value in namespace.items():
                if key not in exclude_keys and not key.startswith('__'):
                    ctx.deps.persistent_namespace[key] = value
            
            # Get captured print output
            printed_output = output_capture.getvalue()
            
            # Combine print output with result
            if printed_output:
                result_str = printed_output
                if result is not None:
                    result_str += f"\n[Return value]: {str(result)}"
            else:
                result_str = str(result) if result is not None else "None"
        
        # Truncate if too long
        if len(result_str) > 5000:
            result_str = result_str[:5000] + f"\n... (truncated, total: {len(result_str)} chars)"
        
        print(f'[{ctx.deps.sheet_name}] Result: {result_str[:200]}...\n')
        return result_str
        
    except Exception as e:
        # Get any partial output that was captured before the error
        partial_output = output_capture.getvalue()
        if partial_output:
            error_msg = f'Query `{query}` failed after partial output:\n{partial_output}\nError: {e}'
        else:
            error_msg = f'Query `{query}` failed: {e}'
            
        print(f'[{ctx.deps.sheet_name}] Error: {error_msg}\n')
        raise ModelRetry(
            f'{error_msg}\n\n'
            f'Common fixes:\n'
            f'- For imports: import pandas as pd; import numpy as np\n'
            f'- For slicing: df.iloc[start:end]\n'
            f'- For single row: df.iloc[index]\n'
            f'- For strings: df["col"].str.replace(",", "", regex=False)\n'
            f'- To persist: df = df.iloc[:-1]\n'
            f'- Variables persist: suspicious_rows = df["col"].isna()\n'
        ) from e

async def determine_relevant_sheets(excel_file: str, sheet_names: List[str], question: str) -> List[str]:
    """
    First pass: Analyze sheet previews to determine which sheets are relevant.
    
    Returns:
        List of relevant sheet names to analyze in detail
    """
    print(f"\n{'='*80}")
    print(f"PHASE 1: Determining Relevant Sheets")
    print(f"{'='*80}\n")
    
    # Read preview (head) of each sheet
    sheet_previews = {}
    for sheet_name in sheet_names:
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()
            
            # Create a preview string with shape, columns, and head
            preview = f"""
Sheet: {sheet_name}
Shape: {df.shape}
Columns: {df.columns.tolist()}

First 10 rows:
{df.head(10).to_string()}
"""
            sheet_previews[sheet_name] = preview
            print(f"✓ Loaded preview of sheet: {sheet_name} ({df.shape[0]} rows, {df.shape[1]} cols)")
            
        except Exception as e:
            print(f"✗ Error loading sheet {sheet_name}: {e}")
            sheet_previews[sheet_name] = f"Error loading sheet: {e}"
    
    print(f"\n{'='*60}")
    print(f"Running relevance analysis on {len(sheet_previews)} sheets...")
    print(f"{'='*60}\n")
    
    # Prepare prompt for relevance agent
    relevance_prompt = f"""
Question to answer: {question}

Below are previews of all available sheets in the Excel file.

"""
    
    for sheet_name, preview in sheet_previews.items():
        relevance_prompt += f"\n{'-'*60}\n{preview}\n"
    
    relevance_prompt += f"""

Based on these previews, determine which sheets are relevant for answering the question:
"{question}"

Provide your analysis as a structured response identifying relevant and irrelevant sheets.
"""
    
    try:
        # Run relevance agent with structured output
        result = await relevance_agent.run(relevance_prompt)
        relevance_data = result.output
        
        print(f"\n{'='*60}")
        print(f"Relevance Analysis Complete")
        print(f"{'='*60}")
        print(f"\n{relevance_data.analysis_summary}\n")
        print(f"✓ Relevant sheets ({len(relevance_data.relevant_sheets)}): {relevance_data.relevant_sheets}")
        print(f"✗ Irrelevant sheets ({len(relevance_data.irrelevant_sheets)}): {relevance_data.irrelevant_sheets}")
        print(f"{'='*60}\n")
        
        return relevance_data.relevant_sheets
        
    except Exception as e:
        print(f"Error in relevance analysis: {e}")
        print("Falling back to analyzing all sheets...\n")
        return sheet_names


async def analyze_sheet(sheet_name: str, df: pd.DataFrame, question: str, semaphore: asyncio.Semaphore) -> Dict[str, str]:
    """Analyze a single sheet with the data agent, limiting concurrency."""
    async with semaphore:
        print(f"\n{'='*60}")
        print(f"Starting analysis of sheet: {sheet_name}")
        print(f"{'='*60}\n")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Create deps with sheet name and empty persistent namespace
        deps = DataFrameDeps(df=df, sheet_name=sheet_name, persistent_namespace={})
        
        # Add sheet context to the question
        contextualized_question = f"[Analyzing sheet: {sheet_name}]\n\n{question}\n\nNote: Start your response by stating which sheet you're analyzing."
        
        try:
            # Run agent asynchronously
            result = await data_agent.run(contextualized_question, deps=deps)
            answer = result.output  # Use .output not .data
            
            print(f"\n{'='*60}")
            print(f"Completed analysis of sheet: {sheet_name}")
            print(f"{'='*60}\n")
            
            return {
                "sheet_name": sheet_name,
                "answer": answer,
                "success": True
            }
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Error analyzing sheet {sheet_name}: {e}")
            print(f"{'='*60}\n")
            
            return {
                "sheet_name": sheet_name,
                "answer": f"Error analyzing this sheet: {str(e)}",
                "success": False
            }


async def analyze_all_sheets(excel_file: str, question: str, max_concurrent: int = 3, skip_relevance_check: bool = False) -> str:
    """
    Main function to analyze all sheets in an Excel file and return a merged answer.
    
    Args:
        excel_file: Path to the Excel file
        question: The question to ask about the data
        max_concurrent: Maximum number of sheets to analyze concurrently (default: 3)
        skip_relevance_check: If True, analyze all sheets without relevance filtering
    
    Returns:
        A comprehensive merged answer from all sheets
    """
    print(f"\n{'#'*80}")
    print(f"# MULTI-SHEET ANALYSIS")
    print(f"# File: {excel_file}")
    print(f"# Question: {question}")
    print(f"# Max Concurrent: {max_concurrent}")
    print(f"{'#'*80}\n")
    
    # Read all sheet names
    excel_file_obj = pd.ExcelFile(excel_file)
    all_sheet_names = excel_file_obj.sheet_names
    
    print(f"Found {len(all_sheet_names)} sheets: {all_sheet_names}\n")
    
    # Phase 1: Determine relevant sheets (unless skipped)
    if skip_relevance_check:
        print("Skipping relevance check - analyzing all sheets\n")
        sheets_to_analyze = all_sheet_names
    else:
        sheets_to_analyze = await determine_relevant_sheets(excel_file, all_sheet_names, question)
        
        if not sheets_to_analyze:
            print("⚠️  No relevant sheets found. Analyzing all sheets as fallback...\n")
            sheets_to_analyze = all_sheet_names
    
    print(f"\n{'='*80}")
    print(f"PHASE 2: Detailed Analysis of {len(sheets_to_analyze)} Relevant Sheets")
    print(f"{'='*80}\n")
    
    # Phase 2: Analyze relevant sheets in detail
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for relevant sheets only
    tasks = []
    for sheet_name in sheets_to_analyze:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        task = analyze_sheet(sheet_name, df, question, semaphore)
        tasks.append(task)
    
    # Run all tasks (semaphore limits to max_concurrent at a time)
    print(f"Running detailed analysis on {len(tasks)} sheets (max {max_concurrent} concurrent)...\n")
    results = await asyncio.gather(*tasks)
    
    # Organize results by sheet name
    sheet_answers = {}
    for result in results:
        sheet_name = result["sheet_name"]
        answer = result["answer"]
        success = result["success"]
        
        if success:
            sheet_answers[sheet_name] = answer
        else:
            sheet_answers[sheet_name] = f"❌ {answer}"
    
    print(f"\n{'='*80}")
    print(f"PHASE 3: Merging Results from All Sheets")
    print(f"{'='*80}\n")
    
    # Phase 3: Use merger agent to combine all answers
    merger_deps = MergerDeps(
        sheet_answers=sheet_answers,
        original_question=question
    )
    
    merger_prompt = f"""
Original Question: {question}

You have received analysis results from {len(sheet_answers)} different sheets.

Here are the individual answers from each sheet:

"""
    
    # Add df.head() for each sheet to help identify accumulation sheets
    for sheet_name, answer in sheet_answers.items():
        merger_prompt += f"\n{'='*60}\n"
        merger_prompt += f"SHEET: {sheet_name}\n"
        merger_prompt += f"{'='*60}\n"
        
        # Add df.head() data for this sheet
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()
            merger_prompt += f"\nData Preview (df.head()):\n"
            merger_prompt += f"Shape: {df.shape}\n"
            merger_prompt += f"Columns: {df.columns.tolist()}\n\n"
            merger_prompt += f"{df.head(10).to_string()}\n\n"
        except Exception as e:
            merger_prompt += f"\nData Preview: Error loading sheet data - {e}\n\n"
        
        merger_prompt += f"Analysis Result:\n{answer}\n"
    
    merger_prompt += f"""

Please synthesize these results into a comprehensive, unified answer that:
1. Directly answers the original question
2. Provides an executive summary with key findings
3. Highlights important patterns or differences across sheets
4. Organizes the information clearly
"""
    
    try:
        merger_result = await merger_agent.run(merger_prompt, deps=merger_deps)
        final_answer = merger_result.output  # Use .output not .data
    except Exception as e:
        # Fallback if merger fails
        final_answer = f"Error in merging: {e}\n\nIndividual sheet answers:\n\n"
        for sheet_name, answer in sheet_answers.items():
            final_answer += f"\n## {sheet_name}\n{answer}\n"
    
    print(f"\n{'#'*80}")
    print(f"# FINAL MERGED ANSWER")
    print(f"{'#'*80}")
    print(final_answer)
    print(f"{'#'*80}\n")
    
    return final_answer


# Convenience wrapper for running in sync context
def ask_all_sheets(excel_file: str, question: str, max_concurrent: int = 3, skip_relevance_check: bool = False) -> str:
    """
    Synchronous wrapper for analyze_all_sheets.
    
    Args:
        excel_file: Path to the Excel file
        question: The question to ask about the data
        max_concurrent: Maximum number of sheets to analyze concurrently (default: 3)
        skip_relevance_check: If True, analyze all sheets without relevance filtering
    
    Returns:
        A comprehensive merged answer from all sheets
    """
    return asyncio.run(analyze_all_sheets(excel_file, question, max_concurrent, skip_relevance_check))


# Example usage
if __name__ == "__main__":
    # Async usage (recommended)
    async def main():
        answer = await analyze_all_sheets(
            excel_file='doc2(0930119).xlsx',
            question="""
            	How many funds returned > $100M since inception? 
            """,
            max_concurrent=3,
            skip_relevance_check=False  # Set to True to analyze all sheets
        )
        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)
    
    # Run async
    asyncio.run(main())