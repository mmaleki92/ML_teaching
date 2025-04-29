import torch
import torch.nn as nn
import torch.optim as optim
import re
import hazm  # Persian NLP library

```bash
pip install hazm
```
class WordGRU(nn.Module):
    # [Same as before]

def tokenize_paragraph(paragraph):
    """
    Tokenize a Persian paragraph into words using the Hazm library.
    """
    # For Persian text tokenization
    try:
        # Try to use Hazm library if available
        normalizer = hazm.Normalizer()
        normalized_text = normalizer.normalize(paragraph)
        tokenizer = hazm.WordTokenizer()
        words = tokenizer.tokenize(normalized_text)
    except (ImportError, NameError):
        # Fallback to a more Persian-friendly regex if Hazm is not available
        # This won't be perfect but better than the English word boundary regex
        words = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+', paragraph)
    return words


if __name__ == "__main__":
    # Define a paragraph of Persian text
    paragraph = """
    پردازش زبان طبیعی یک زیرشاخه از زبان‌شناسی، علوم کامپیوتر و هوش مصنوعی است
    که با تعاملات بین کامپیوترها و زبان انسانی سروکار دارد، به ویژه اینکه چگونه
    کامپیوترها را برنامه‌ریزی کنیم تا مقادیر زیادی از داده‌های زبان طبیعی را
    پردازش و تحلیل کنند. هدف، یک کامپیوتر است که قادر به درک محتوای اسناد،
    از جمله ظرافت‌های زمینه‌ای زبان موجود در آنها باشد.
    """
