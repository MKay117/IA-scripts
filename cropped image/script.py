import os
import time
import json
import base64
import cv2
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from openai import AzureOpenAI

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

date_str = datetime.now().strftime("%Y-%m-%d")
INPUT_DIR = Path("input")
OUTPUT_DIR = Path(f"output/architecture-agent/{date_str}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

oai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION,
)


def encode_image_base64(img_array):
    _, buffer = cv2.imencode(".jpg", img_array)
    return base64.b64encode(buffer).decode("utf-8")


def get_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def call_vlm(prompt, img_array, schema=None, json_mode=True):
    """Wrapper for GPT-4o calls ensuring detail: high."""
    b64_img = encode_image_base64(img_array)
    sys_msg = "You are an Expert Enterprise Infrastructure Architect."
    if json_mode:
        sys_msg += " Output strictly valid JSON."
        prompt += f"\n\nStrict JSON Schema:\n{schema}"

    messages = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    kwargs = {"model": DEPLOYMENT_NAME, "messages": messages, "temperature": 0.0}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = oai_client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return json.loads(content) if json_mode else content


# ==========================================
# STAGE 0: OCR & ANCHORING
# ==========================================
def run_stage0_extraction(image_path, img_array):
    print(f"\n[Stage 0] Azure OCR & Visual Anchoring...")
    client = DocumentIntelligenceClient(
        DOC_INTEL_ENDPOINT, AzureKeyCredential(DOC_INTEL_KEY)
    )
    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
    raw_ocr = poller.result().as_dict()

    elements, counter, annotated = [], 1, img_array.copy()
    for para in raw_ocr.get("paragraphs", []):
        content = para.get("content", "").strip()
        if len(content) < 2 or "Page" in content:
            continue

        if para.get("boundingRegions"):
            bbox = get_bbox(para["boundingRegions"][0]["polygon"])
            el_id = str(counter)
            elements.append({"id": el_id, "text": content, "bbox": bbox})

            x, y, x_max, y_max = bbox
            cv2.rectangle(annotated, (x, y), (x_max, y_max), (255, 0, 0), 2)
            label = f"[{el_id}]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - lh - 4), (x + lw, y), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                label,
                (x, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            counter += 1
    return elements, annotated


# ==========================================
# STAGE 1: HIERARCHY
# ==========================================
def run_stage1_hierarchy(annotated_img, elements):
    print("[Stage 1] VLM: Extracting Semantic Context & Hierarchy...")
    lookup = {el["id"]: el["text"] for el in elements}
    prompt = f"""
    Lookup Table: {json.dumps(lookup)}
    1. Define the hosting environment (Cloud, On-Prem, Hybrid).
    2. Establish Parent-Child hierarchy. CRITICAL RULE: A parent MUST physically enclose the child with a drawn boundary box. Do not assign a parent if they are just stacked next to each other.
    """
    schema = """{
      "hosting_context": "Domain description",
      "hierarchy": [{ "child_id": "ID", "reasoning": "Visible inside box X", "parent_id": "ID or null" }]
    }"""
    result = call_vlm(prompt, annotated_img, schema)

    hierarchy_map = {
        str(item["child_id"]): str(item.get("parent_id"))
        for item in result.get("hierarchy", [])
    }
    for el in elements:
        el["parent_id"] = hierarchy_map.get(el["id"])
    return elements, result.get("hosting_context", "Unknown")


# ==========================================
# STAGE 2: MACRO ROUTING
# ==========================================
def run_stage2_macro_routing(annotated_img, elements, context):
    print("[Stage 2] VLM: Mapping Macro Edges...")
    lookup = {el["id"]: el["text"] for el in elements}
    prompt = f"""
    Context: {context}
    Lookup: {json.dumps(lookup)}
    Trace MACRO connections (thick lines between major containers/zones). Read text directly on the lines for the 'label_or_protocol'.
    """
    schema = """{
      "macro_edges": [{"source_id": "ID", "target_id": "ID", "direction": "one-way | bi-directional", "label_or_protocol": "e.g., 443/HTTPS"}]
    }"""
    return call_vlm(prompt, annotated_img, schema).get("macro_edges", [])


# ==========================================
# STAGE 3: MICRO ROUTING (AUTOCROP)
# ==========================================
def run_stage3_micro_routing(annotated_img, elements, macro_edges, context):
    print("[Stage 3] VLM Autocrop: Tracing Micro/Orphan Edges...")
    parent_ids = {str(el["parent_id"]) for el in elements if el.get("parent_id")}
    leaf_nodes = [el for el in elements if str(el["id"]) not in parent_ids]
    routed_ids = {str(edge.get("source_id")) for edge in macro_edges} | {
        str(edge.get("target_id")) for edge in macro_edges
    }
    orphans = [node for node in leaf_nodes if str(node["id"]) not in routed_ids]

    micro_edges = []
    img_h, img_w = annotated_img.shape[:2]
    lookup = {el["id"]: el["text"] for el in elements}

    for orphan in orphans:
        x_min, y_min, x_max, y_max = orphan["bbox"]
        pad = 150
        c_y1, c_y2 = max(0, y_min - pad), min(img_h, y_max + pad)
        c_x1, c_x2 = max(0, x_min - pad), min(img_w, x_max + pad)

        prompt = f"""Context: {context}\nLookup: {json.dumps(lookup)}\nTrace the physical line originating from ID [{orphan['id']}] ({orphan['text']}). What is the ID at the end of this line? Extract line text for the protocol."""
        schema = """{"target_id": "ID", "direction": "one-way | bi-directional", "label_or_protocol": "text"}"""
        try:
            res = call_vlm(prompt, annotated_img[c_y1:c_y2, c_x1:c_x2], schema)
            if res.get("target_id"):
                micro_edges.append(
                    {
                        "source_id": orphan["id"],
                        "target_id": str(res["target_id"]),
                        "direction": res.get("direction"),
                        "label_or_protocol": res.get("label_or_protocol"),
                    }
                )
        except Exception as e:
            pass
    return micro_edges


# ==========================================
# STAGE 4: CONSOLIDATION (TEXT-BASED GRAPH)
# ==========================================
def run_stage4_consolidation(elements, macro_edges, micro_edges, context):
    print("[Stage 4] Python: Assembling Final Digraph with OCR Text...")
    id_lookup = {el["id"]: el["text"] for el in elements}

    resolved_edges = []
    for edge in macro_edges + micro_edges:
        src_text = id_lookup.get(str(edge.get("source_id")), edge.get("source_id"))
        tgt_text = id_lookup.get(str(edge.get("target_id")), edge.get("target_id"))
        resolved_edges.append(
            {
                "source": src_text,
                "target": tgt_text,
                "direction": edge.get("direction", "Unknown"),
                "label_or_protocol": edge.get("label_or_protocol", "None"),
            }
        )

    return {
        "metadata": {"context": context},
        "nodes": elements,
        "edges": resolved_edges,
    }


# ==========================================
# STAGE 5: THE ARCHITECT (SUMMARY)
# ==========================================
def run_stage5_summary(final_digraph):
    print("[Stage 5] LLM: Generating Architecture Summary...")
    prompt = f"""
    You are an Enterprise Architect. Review this JSON architecture graph:
    {json.dumps(final_digraph)}
    
    Output a comprehensive, highly detailed architectural summary in valid Markdown.
    Use nested bullet points. Cover:
    - Hosting Environment & Primary Boundaries
    - End-to-End User Traffic Flow
    - Core Application & Database Tiers
    - Security & Networking Protocols
    """
    response = oai_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ==========================================
# STAGE 6: THE CRITIC (VALIDATION)
# ==========================================
def run_stage6_critic(annotated_img, final_digraph, summary):
    print("[Stage 6] VLM: Red-Team Validation & Delta Report...")
    prompt = f"""
    You are a strict Architecture Auditor. Compare this generated Graph & Summary against the visual image provided.
    
    Graph: {json.dumps(final_digraph['edges'])}
    Summary: {summary}
    
    Find discrepancies. Did the graph miss a thick line? Did it invent a connection?
    """
    schema = """{
      "good": ["List of accurate topological features captured"],
      "bad": ["List of hallucinated or incorrect connections"],
      "delta": ["List of visually obvious connections that are missing from the graph"],
      "corrections_needed": ["Specific instructions to fix the graph"]
    }"""
    return call_vlm(prompt, annotated_img, schema)


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        ts = time.strftime("%H%M%S")
        base_name = img_path.stem
        print(f"\n{'='*50}\nSTARTING PIPELINE: {base_name}\n{'='*50}")
        raw_img = cv2.imread(str(img_path))

        # Stage 0
        elements, annotated_img = run_stage0_extraction(img_path, raw_img)
        cv2.imwrite(
            str(OUTPUT_DIR / f"{base_name}_stage0_annotated_{ts}.jpg"), annotated_img
        )

        # Stage 1
        elements, context = run_stage1_hierarchy(annotated_img, elements)
        with open(OUTPUT_DIR / f"{base_name}_stage1_nodes_{ts}.json", "w") as f:
            json.dump({"nodes": elements}, f, indent=4)

        # Stage 2 & 3
        macro_edges = run_stage2_macro_routing(annotated_img, elements, context)
        micro_edges = run_stage3_micro_routing(
            annotated_img, elements, macro_edges, context
        )

        # Stage 4 (Text-based Edges)
        final_digraph = run_stage4_consolidation(
            elements, macro_edges, micro_edges, context
        )
        with open(OUTPUT_DIR / f"{base_name}_stage4_DIGRAPH_{ts}.json", "w") as f:
            json.dump(final_digraph, f, indent=4)

        # Stage 5 (Summary)
        summary_md = run_stage5_summary(final_digraph)
        with open(OUTPUT_DIR / f"{base_name}_stage5_SUMMARY_{ts}.md", "w") as f:
            f.write(summary_md)

        # Stage 6 (Critic)
        critic_report = run_stage6_critic(annotated_img, final_digraph, summary_md)
        with open(OUTPUT_DIR / f"{base_name}_stage6_VALIDATION_{ts}.json", "w") as f:
            json.dump(critic_report, f, indent=4)

        print(f"\n[SUCCESS] Pipeline complete. Artifacts saved to {OUTPUT_DIR.name}")


if __name__ == "__main__":
    main()
