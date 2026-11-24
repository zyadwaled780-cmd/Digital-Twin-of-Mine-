"""Explicit offline UI test harness for verifying dark theme, bidi lists, tables, and iframes.

This module is strictly for local UI verification when provider API keys are unavailable.
Production traffic uses app.py directly, which never serves mock replies.
"""

import os
import app
from input_guard import validate_and_classify_input


MOCK_ENGLISH_LONG = """Here is an overview of Zyad Waled Talat's active CV projects and technical skills:

### Projects
- **Digital Twin of Me**: Personal AI Digital Twin portfolio assistant.
  - Repository: https://github.com/zyadwaled780-cmd/Digital-Twin-of-Mine-
  - Focus: Grounded Q&A, workflow automation showcase, and visitor follow-up.
- **AI Customer Support Agent**: AI customer support project (DEPI team track).
  - Repository: https://github.com/samirhanna11/ai-customer-support-agent
  - Focus: Retrieval-augmented support responses with Python and Ollama.
- **My Portfolio**: https://github.com/zyadwaled780-cmd/portfolio
- **AI-learning-app**: https://github.com/zyadwaled780-cmd/AI-Learning-app

| Project | Key Focus | Repository |
| :--- | :--- | :--- |
| `Digital Twin of Me` | AI Digital Twin assistant | https://github.com/zyadwaled780-cmd/Digital-Twin-of-Mine- |
| `AI Customer Support Agent` | RAG customer support agent | https://github.com/samirhanna11/ai-customer-support-agent |
| `My Portfolio` | Personal portfolio site | https://github.com/zyadwaled780-cmd/portfolio |
| `AI-learning-app` | AI learning application | https://github.com/zyadwaled780-cmd/AI-Learning-app |

Technical skills listed: `n8n`, `SQL`, `Supabase`, `Ollama`, `Hugging Face`, `Git / GitHub`, `Docker`, `RAG`, `Python`, `JavaScript`, `Dart`, `Flutter`.
"""

MOCK_ARABIC_LONG = """أهلاً بيك! دي نظرة شاملة على مشاريع ومهارات زياد في السيرة الذاتية الحالية:

### المشاريع الحقيقية
- مشروع **Digital Twin of Me**: المساعد الرقمي الشخصي لزياد على البورتفوليو.
  - رابط المشروع على GitHub: https://github.com/zyadwaled780-cmd/Digital-Twin-of-Mine-
  - قائمة فرعية بالإنجليزية للاختبار:
    - Built with `Python`, `n8n`-inspired dark theme, and `RAG`.
    - Clean modular codebase (`app.py`, `context.py`, `providers.py`, `knowledge_base.py`).
- مشروع **AI Customer Support Agent**: مساعد خدمة عملاء بالذكاء الاصطناعي.
  - رابط المشروع على GitHub: https://github.com/samirhanna11/ai-customer-support-agent
  - نقاط القوة الأساسية:
    - استرجاع سياقي للمعلومات باستخدام `RAG` و`Ollama`.
    - تعاون جماعي ضمن مبادرة DEPI في مسار AI Automation.
- مشروع **My Portfolio**: https://github.com/zyadwaled780-cmd/portfolio
- مشروع **AI-learning-app**: https://github.com/zyadwaled780-cmd/AI-Learning-app

### جدول الأدوات والمهارات
| الأداة | المجال | الحالة في السيرة الذاتية |
| :--- | :--- | :--- |
| أداة `n8n` | أتمتة سير العمل | مدرجة ضمن المهارات التقنية |
| قاعدة بيانات `SQL` و`Supabase` | إدارة البيانات والخلفيات | مدرجة ضمن المهارات التقنية |
| لغة `Python` و`JavaScript` و`Flutter` | تطوير البرمجيات والتطبيقات | مدرجة ضمن المهارات التقنية |

ملاحظة ختامية: تم استبعاد المشاريع القديمة غير المدرجة في السيرة الذاتية الجديدة، ولغة C# ولغة C++ مش مذكورتين ضمن مهارات زياد وليد طلعت في فترة 2025/2026!
"""


def mock_chat(message, history):
    """Return deterministic replies for offline browser/UI verification."""
    text = app._extract_text(message).strip()
    clean_history = app._normalize_history(history)
    is_valid, rejection_msg = validate_and_classify_input(text, history=clean_history)
    if not is_valid:
        return rejection_msg

    has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in text)
    if has_arabic:
        return MOCK_ARABIC_LONG
    return MOCK_ENGLISH_LONG


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("MOCK_UI_PORT", "7861"))
    demo = app.create_chat_interface(mock_chat)
    demo.launch(
        server_name=host,
        server_port=port,
        theme=app.N8N_DARK_THEME,
        css=app.BIDI_DARK_CSS,
        js=app.BIDI_OBSERVER_JS,
    )


if __name__ == "__main__":
    main()
