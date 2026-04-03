"""
AI Agents
---------
Agent 1: Medical Summary Agent
Agent 2: Symptom Checker Agent

Both use Claude via the Anthropic SDK.
"""

import json
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Agent 1: Medical Summary Agent ───────────────────────────────────────────

SUMMARY_SYSTEM = """You are a senior clinical documentation specialist and medical writer.
Your task is to generate precise, structured medical summaries from clinical content.
Use proper medical terminology. Highlight clinically significant findings.
Never fabricate or infer information not present in the source text."""

DOC_TYPE_LABELS = {
    "patient_record": "patient medical record",
    "lab_report": "laboratory/pathology report",
    "research_paper": "medical research paper",
    "discharge_summary": "hospital discharge summary",
    "drug_pharma": "pharmaceutical/drug documentation",
    "case_notes": "clinical case notes",
}

FORMAT_INSTRUCTIONS = {
    "structured": "Use clearly labeled sections with headers (##). Include: Chief Complaint, History, Findings, Assessment, Plan/Recommendations.",
    "narrative": "Write as a flowing clinical narrative. Professional tone, paragraph format.",
    "bullet_points": "Use concise bullet points under labeled sections. Maximum 2 lines per bullet.",
    "soap": "Format strictly as a SOAP note:\n## Subjective\n## Objective\n## Assessment\n## Plan",
    "discharge": "Format as a discharge summary with: Admission Details, Diagnosis, Hospital Course, Discharge Medications, Follow-up Instructions.",
}


def run_summary_agent(
    context: str,
    doc_type: str = "patient_record",
    output_format: str = "structured",
    focus_areas: str = "",
    api_key: str = "",
) -> dict:
    """
    Generate a medical summary from context text.
    Returns {summary: str, tokens_used: int, doc_type: str, format: str}
    """
    _client = Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    type_label = DOC_TYPE_LABELS.get(doc_type, "medical document")
    fmt_instruction = FORMAT_INSTRUCTIONS.get(output_format, FORMAT_INSTRUCTIONS["structured"])

    focus_str = f"\nPay special attention to: {focus_areas}" if focus_areas else ""

    system = f"""{SUMMARY_SYSTEM}

Document type: {type_label}
Output format: {fmt_instruction}{focus_str}"""

    prompt = f"""Please generate a comprehensive {output_format} summary of the following {type_label}:

---
{context[:8000]}
---

Generate the summary now:"""

    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "summary": response.content[0].text,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        "doc_type": doc_type,
        "format": output_format,
    }


# ── Agent 2: Symptom Checker Agent ───────────────────────────────────────────

SYMPTOM_SYSTEM = """You are a clinical decision support AI assistant.
Analyze the patient presentation and provide structured clinical analysis.
You must respond ONLY with valid JSON — no preamble, no markdown fences.

Your JSON must match this exact schema:
{
  "urgency": "Emergency|Urgent|Routine",
  "urgency_reason": "one sentence explanation",
  "differentials": [
    {
      "name": "Condition name",
      "probability": 75,
      "severity": "high|medium|low",
      "rationale": "brief clinical reasoning",
      "icd10": "ICD-10 code if known"
    }
  ],
  "red_flags": ["flag1", "flag2"],
  "investigations": ["investigation1", "investigation2"],
  "management": ["step1", "step2"],
  "when_to_seek_care": "specific guidance",
  "disclaimer": "Always consult a qualified healthcare professional for diagnosis and treatment."
}

Rules:
- Include 3-5 differentials ordered by probability
- Be specific with investigations (e.g., 'CBC with differential' not just 'blood test')
- Red flags should be actionable warning signs
- Never fabricate specific patient data not provided
"""


def run_symptom_agent(
    symptoms: list[str],
    age: int | None,
    sex: str,
    vitals: dict,
    description: str,
    medical_history: str,
    doc_context: str,
    api_key: str = "",
) -> dict:
    """
    Run the symptom checker agent.
    Returns structured JSON with differentials, urgency, investigations etc.
    """
    _client = Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    # Build patient presentation string
    parts = []
    if age:
        parts.append(f"Age: {age} years")
    sex_label = {"M": "Male", "F": "Female", "O": "Other"}.get(sex, sex)
    parts.append(f"Sex: {sex_label}")

    if symptoms:
        parts.append(f"Presenting symptoms: {', '.join(symptoms)}")

    if vitals:
        vital_strs = []
        if vitals.get("temp"):
            vital_strs.append(f"Temperature {vitals['temp']}°C")
        if vitals.get("hr"):
            vital_strs.append(f"HR {vitals['hr']} bpm")
        if vitals.get("bp"):
            vital_strs.append(f"BP {vitals['bp']} mmHg")
        if vitals.get("spo2"):
            vital_strs.append(f"SpO₂ {vitals['spo2']}%")
        if vital_strs:
            parts.append(f"Vitals: {', '.join(vital_strs)}")

    if description:
        parts.append(f"Clinical description: {description}")

    if medical_history:
        parts.append(f"Medical history: {medical_history}")

    if doc_context:
        parts.append(f"\nRelevant patient document context:\n{doc_context[:3000]}")

    presentation = "\n".join(parts)

    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYMPTOM_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Analyze this patient presentation:\n\n{presentation}"
        }],
    )

    raw = response.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return raw text wrapped
        result = {
            "urgency": "Unknown",
            "urgency_reason": "Could not parse structured response",
            "differentials": [],
            "red_flags": [],
            "investigations": [],
            "management": [],
            "when_to_seek_care": "Please consult a doctor",
            "disclaimer": "Always consult a qualified healthcare professional.",
            "raw_response": raw,
        }

    result["tokens_used"] = response.usage.input_tokens + response.usage.output_tokens
    return result


# ── RAG Chat ──────────────────────────────────────────────────────────────────

def run_rag_chat(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    api_key: str = "",
) -> str:
    """
    Run a RAG-grounded chat turn.
    context_chunks: list of {text, doc_name, score}
    chat_history: list of {role, content} (last N turns)
    """
    _client = Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    if context_chunks:
        context_str = "\n\n---\n\n".join([
            f"[Source {i+1} — {c['doc_name']} (relevance: {c['score']:.2f})]:\n{c['text']}"
            for i, c in enumerate(context_chunks)
        ])
        system = f"""You are MedRAG, a precise clinical knowledge assistant.
Answer questions based strictly on the provided medical context.
Always cite source numbers like [Source 1] when referencing specific content.
If the context is insufficient, say so clearly — never fabricate clinical information.

MEDICAL CONTEXT:
{context_str}"""
    else:
        system = """You are MedRAG, a clinical knowledge assistant.
No relevant documents were found in the knowledge base for this query.
Inform the user and suggest they upload relevant documents, or answer from general medical knowledge with appropriate caveats."""

    messages = chat_history[-8:] + [{"role": "user", "content": query}]

    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        system=system,
        messages=messages,
    )

    return response.content[0].text
