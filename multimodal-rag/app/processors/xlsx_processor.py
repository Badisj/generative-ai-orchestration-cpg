# FILE: app/processors/xlsx_processor.py
"""
Excel and CSV processor using openpyxl and pandas.
Converts spreadsheet data to markdown tables.
"""

from pathlib import Path
from typing import List, Optional
import logging

from app.processors.base import BaseProcessor, Document

logger = logging.getLogger(__name__)


class ExcelProcessor(BaseProcessor):
    """
    Process Excel and CSV files.
    
    Features:
    - Handles .xlsx, .xls, .csv, .tsv files
    - Converts sheets to markdown tables
    - Preserves sheet names as section headers
    - Handles multiple sheets
    """
    
    supported_extensions = [".xlsx", ".xls", ".csv", ".tsv"]
    
    def __init__(self, max_rows: int = 1000, max_cols: int = 50):
        """
        Initialize Excel processor.
        
        Parameters
        ----------
        max_rows : int
            Maximum rows to process per sheet.
        max_cols : int
            Maximum columns to process.
        """
        self.max_rows = max_rows
        self.max_cols = max_cols
    
    def can_process(self, file_path: Path) -> bool:
        """Check if file is a spreadsheet."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def process(self, file_path: Path) -> List[Document]:
        """
        Extract data from spreadsheet as markdown tables.
        
        Parameters
        ----------
        file_path : Path
            Path to spreadsheet file.
            
        Returns
        -------
        List[Document]
            List of documents, one per sheet.
        """
        import pandas as pd
        
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        documents = []
        
        try:
            if suffix == ".csv":
                # CSV file - single sheet
                df = pd.read_csv(str(file_path), nrows=self.max_rows)
                df = df.iloc[:, :self.max_cols]  # Limit columns
                
                content = self._dataframe_to_markdown(df)
                if content:
                    doc = self._create_document(
                        content=content,
                        source_file=file_path.name,
                        file_type="csv",
                        sheet_name="Sheet1",
                        row_count=len(df),
                        column_count=len(df.columns),
                    )
                    documents.append(doc)
                    
            elif suffix == ".tsv":
                # TSV file
                df = pd.read_csv(str(file_path), sep="\t", nrows=self.max_rows)
                df = df.iloc[:, :self.max_cols]
                
                content = self._dataframe_to_markdown(df)
                if content:
                    doc = self._create_document(
                        content=content,
                        source_file=file_path.name,
                        file_type="tsv",
                        sheet_name="Sheet1",
                        row_count=len(df),
                        column_count=len(df.columns),
                    )
                    documents.append(doc)
                    
            else:
                # Excel file - multiple sheets possible
                excel_file = pd.ExcelFile(str(file_path))
                
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_name,
                        nrows=self.max_rows
                    )
                    df = df.iloc[:, :self.max_cols]
                    
                    # Skip empty sheets
                    if df.empty:
                        continue
                    
                    content = f"## Sheet: {sheet_name}\n\n"
                    content += self._dataframe_to_markdown(df)
                    
                    doc = self._create_document(
                        content=content,
                        source_file=file_path.name,
                        file_type="xlsx",
                        sheet_name=sheet_name,
                        row_count=len(df),
                        column_count=len(df.columns),
                    )
                    documents.append(doc)
                    
        except Exception as e:
            logger.error(f"Error processing spreadsheet {file_path}: {e}")
        
        logger.info(f"Extracted {len(documents)} sheets from {file_path.name}")
        return documents
    
    def _dataframe_to_markdown(self, df) -> str:
        """
        Convert pandas DataFrame to markdown table.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to convert.
            
        Returns
        -------
        str
            Markdown formatted table.
        """
        import pandas as pd
        
        if df.empty:
            return ""
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Convert to markdown
        lines = []
        
        # Header row
        headers = df.columns.tolist()
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Data rows
        for _, row in df.iterrows():
            cells = [str(val).replace("|", "\\|").replace("\n", " ") for val in row]
            lines.append("| " + " | ".join(cells) + " |")
        
        return "\n".join(lines)
    
    def get_summary(self, file_path: Path) -> Optional[str]:
        """
        Get a brief summary of spreadsheet contents.
        
        Parameters
        ----------
        file_path : Path
            Path to spreadsheet.
            
        Returns
        -------
        Optional[str]
            Summary string or None.
        """
        import pandas as pd
        
        try:
            suffix = Path(file_path).suffix.lower()
            
            if suffix in [".csv", ".tsv"]:
                sep = "\t" if suffix == ".tsv" else ","
                df = pd.read_csv(str(file_path), sep=sep, nrows=5)
                return f"CSV file with {len(df.columns)} columns: {', '.join(df.columns.tolist())}"
            else:
                excel_file = pd.ExcelFile(str(file_path))
                sheets = excel_file.sheet_names
                return f"Excel file with {len(sheets)} sheets: {', '.join(sheets)}"
                
        except Exception as e:
            logger.warning(f"Could not get summary: {e}")
            return None
