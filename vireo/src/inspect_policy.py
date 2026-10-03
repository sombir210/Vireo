from pathlib import Path
from PyPDF2 import PdfReader

pdf_path = Path('support-policy.pdf')
reader = PdfReader(str(pdf_path))
text = '\n'.join(page.extract_text() or '' for page in reader.pages)
print(text[:8000])
