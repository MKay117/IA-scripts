import os
import time
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from shapely.geometry import Polygon, Point
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeOutputOption,
    DocumentContentFormat,
)
from openai import AzureOpenAI

load_dotenv()

# ==========================================
# CONFIGURATION & TRACKING
# ==========================================
DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

INPUT_DIR = Path("input")

date_str = datetime.now().strftime("%Y-%m-%d")
# OUTPUT_DIR = Path(f"output/single-shot-tracing-pipeline/v2/{date_str}/run 05")
OUTPUT_DIR = Path(f"output/{date_str}/run-03")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global tracker for LLM calls
llm_call_count = 0


def encode_image_base64(img_array):
    _, buffer = cv2.imencode(".jpg", img_array)
    return base64.b64encode(buffer).decode("utf-8")


def get_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def get_centroid(coords):
    bbox = get_bbox(coords)
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]


# ==========================================
# STAGE 0: DOC INTEL & ANNOTATION
# ==========================================


def run_stage0_extraction(image_path):
    print(f"-> [Stage 0] Azure OCR & Minification on {image_path.name}...")

    client = DocumentIntelligenceClient(
        DOC_INTEL_ENDPOINT, AzureKeyCredential(DOC_INTEL_KEY)
    )

    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            output_content_format=DocumentContentFormat.MARKDOWN,
            features=["ocrHighResolution"],
            output=[AnalyzeOutputOption.FIGURES],
        )

    raw_ocr = poller.result().as_dict()

    elements = []
    counter = 1
    for para in raw_ocr.get("paragraphs", []):
        if para.get("boundingRegions"):
            coords = para["boundingRegions"][0]["polygon"]
            elements.append(
                {
                    "id": str(counter),
                    "type": "text",
                    "content": para.get("content"),
                    "polygon": coords,
                    "bbox": get_bbox(coords),
                    "centroid": get_centroid(coords),
                }
            )
            counter += 1

    return elements


def draw_annotations(img_array, elements):
    annotated = img_array.copy()
    for el in elements:
        x, y, x_max, y_max = map(int, el["bbox"])
        cv2.rectangle(annotated, (x, y), (x_max, y_max), (255, 0, 0), 2)
        label = f"[{el['id']}]"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x, y - label_h - 4), (x + label_w, y), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (x, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return annotated


# ==========================================
# STAGE 1: VLM SPATIAL HIERARCHY
# ==========================================


def run_stage1_vlm_hierarchy(annotated_img_array, elements):
    global llm_call_count
    print("-> [Stage 1] VLM: Extracting Parent-Child Hierarchy...")

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    base64_image = encode_image_base64(annotated_img_array)
    id_lookup = {el["id"]: el["content"] for el in elements if el["type"] == "text"}

    prompt = f"""
You are a Spatial Architecture Analyzer. Look at the visual boundary boxes drawn in the background.
Lookup Table: {json.dumps(id_lookup)}
TASK: Establish the Parent-Child hierarchy. Output JSON format: {{ "hierarchy": [ {{ "id": "child", "parent_id": "parent_or_null" }} ] }}
"""

    print(
        f" [LLM Payload Sent] Prompt Length: {len(prompt)} chars  Image Base64 Attached"
    )

    try:
        llm_call_count += 1
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.0,
        )
        vlm_hierarchy = json.loads(response.choices[0].message.content).get(
            "hierarchy", []
        )
        hierarchy_dict = {
            str(item["id"]): str(item["parent_id"]) if item.get("parent_id") else None
            for item in vlm_hierarchy
        }
        final_hierarchy = [
            {**el, "parent_id": hierarchy_dict.get(el["id"], None)}
            for el in elements
            if el["type"] == "text"
        ]
        return final_hierarchy
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        return elements


# ==========================================
# STAGE 2: MACRO-ROUTING (GLOBAL LLM)
# ==========================================


def run_stage2_macro_routing(annotated_img_array, elements):
    global llm_call_count
    print("-> [Stage 2] VLM: Extracting Macro connections...")

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    base64_image = encode_image_base64(annotated_img_array)
    id_lookup = {el["id"]: el["content"] for el in elements}

    prompt = f"""
You are a Cloud Architecture Router. Lookup Table: {json.dumps(id_lookup)}
TASK: Trace ONLY MACRO connections (major outer boundaries/standalone systems).
CRITICAL INSTRUCTIONS:
1. Look for arrowheads to determine flow direction exactly (One-way vs Bi-directional).
2. Actively scan for Private Endpoints (often represented by network link icons). Do not miss them.
Output JSON format: {{ "macro_connections": [ {{"source_id": "ID", "target_id": "ID", "flow": "One-way | Bi-directional", "style_and_meaning": "color/type"}} ] }}
"""

    print(
        f" [LLM Payload Sent] Prompt Length: {len(prompt)} chars  Image Base64 Attached"
    )

    try:
        llm_call_count += 1
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content).get(
            "macro_connections", []
        )
    except Exception as e:
        print(f"Stage 2 failed: {e}")
        return []


# ==========================================
# STAGE 3: AGENTIC SINGLE-SHOT (MICRO-ROUTING)
# ==========================================


def run_stage3_agentic_loop(annotated_img_array, hierarchy, macro_connections):
    global llm_call_count
    print("-> [Stage 3] Agentic Loop: Tracing micro-connections...")

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    base64_image = encode_image_base64(annotated_img_array)

    parent_ids = {
        item["parent_id"] for item in hierarchy if item["parent_id"] is not None
    }
    leaf_nodes = [item for item in hierarchy if item["id"] not in parent_ids]

    routed_ids = {conn["source_id"] for conn in macro_connections} | {
        conn["target_id"] for conn in macro_connections
    }
    orphans = [node for node in leaf_nodes if node["id"] not in routed_ids]

    micro_connections = []
    print(f" -> Found {len(orphans)} unbound components.")

    for orphan in orphans:
        prompt = f"""
Focus ONLY on box explicitly labeled [{orphan['id']}] ({orphan['content']}).
1. Follow ONLY the line touching this box.
2. Identify any arrowheads to determine flow (One-way or Bi-directional).
3. Double check if this line connects to a Private Endpoint icon.
Output JSON format: {{ "target_id": "ID", "flow": "...", "style_and_meaning": "..." }}
"""
        print(
            f" [LLM Payload Sent] Tracing orphan {orphan['id']}. Prompt Length: {len(prompt)} chars"
        )
        try:
            llm_call_count += 1
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.0,
            )
            result = json.loads(response.choices[0].message.content)
            if "target_id" in result and result["target_id"]:
                micro_connections.append(
                    {
                        "source_id": orphan["id"],
                        "target_id": str(result["target_id"]),
                        "flow": result.get("flow", "One-way"),
                        "style_and_meaning": result.get("style_and_meaning", "Unknown"),
                    }
                )
        except Exception as e:
            print(f" -> Trace failed for [ID: {orphan['id']}]: {e}")

    return micro_connections


# ==========================================
# STAGE 4: RECURSIVE CONSOLIDATION
# ==========================================


def build_tree(node_id, id_to_node, parent_to_children):
    node = id_to_node[node_id]
    children_ids = parent_to_children.get(node_id, [])
    if not children_ids:
        return node["content"]
    children_trees = [
        build_tree(child_id, id_to_node, parent_to_children)
        for child_id in children_ids
    ]
    return {
        "name": node["content"],
        "type": "Container/Boundary",
        "children": children_trees,
    }


def run_stage4_consolidation(hierarchy, macro_conns, micro_conns):
    print("-> [Stage 4] Python: Assembling nested JSON Tree...")

    id_to_node = {item["id"]: item for item in hierarchy}
    parent_to_children = {}
    root_nodes = []

    for item in hierarchy:
        pid = item["parent_id"]
        if pid is None:
            root_nodes.append(item["id"])
        else:
            if pid not in parent_to_children:
                parent_to_children[pid] = []
            parent_to_children[pid].append(item["id"])

    architectural_hierarchy, unbound_entities = [], []
    for root_id in root_nodes:
        tree = build_tree(root_id, id_to_node, parent_to_children)
        if isinstance(tree, str):
            unbound_entities.append(tree)
        else:
            architectural_hierarchy.append(tree)

    all_edges = []
    for edge in macro_conns + micro_conns:
        if edge.get("source_id") in id_to_node and edge.get("target_id") in id_to_node:
            all_edges.append(
                {
                    "source": id_to_node[edge["source_id"]]["content"],
                    "target": id_to_node[edge["target_id"]]["content"],
                    "flow": edge.get("flow", "Unknown"),
                    "style_and_meaning": edge.get("style_and_meaning", "Unknown"),
                }
            )

    return {
        "architectural_hierarchy": architectural_hierarchy,
        "external_and_unbound_entities": unbound_entities,
        "end_to_end_flows": all_edges,
    }


# ==========================================
# STAGE 5: FRONT-END SUMMARY GENERATION
# ==========================================


def run_stage5_summary(raw_img_array, final_graph):
    global llm_call_count
    print("-> [Stage 5] VLM: Generating Front-End Summary...")

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    base64_image = encode_image_base64(raw_img_array)

    prompt = f"""
You are a Cloud Solutions Architect. Analyze the raw image and the provided JSON graph of the architecture.
JSON Graph: {json.dumps(final_graph)}
TASK: Generate a clear, high-level summary of this architecture in bullet points, suitable for integration into a Front-End dashboard.
Output JSON format: {{ "summary": ["bullet point 1", "bullet point 2"] }}
"""

    print(
        f" [LLM Payload Sent] Prompt Length: {len(prompt)} chars  Raw Image Base64 Attached"
    )

    try:
        llm_call_count += 1
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Stage 5 failed: {e}")
        return {"summary": ["Summary generation failed."]}


# ==========================================
# STAGE 6: VALIDATION AGENT
# ==========================================


def run_stage6_validation(raw_img_array, final_graph, summary):
    global llm_call_count
    print("-> [Stage 6] Validation Agent: Cross-referencing results...")

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    base64_image = encode_image_base64(raw_img_array)

    prompt = f"""
You are a strict Architecture Validation QA Agent.
1. Look at the raw input image.
2. Review the extracted JSON Graph: {json.dumps(final_graph)}
3. Review the Summary: {json.dumps(summary)}
TASK: Honestly identify what was mapped correctly and what was missed or hallucinated. Pay special attention to arrow directions and Private Endpoints.
Output JSON format:
{{
  "correct_elements": ["List of accurately mapped features"],
  "errors": [
    {{ "issue": "Description of error", "stage_to_fix": "Stage 1/2/3/4", "suggested_correction": "What needs changing" }}
  ]
}}
"""

    print(
        f" [LLM Payload Sent] Prompt Length: {len(prompt)} chars  Raw Image Base64 Attached"
    )

    try:
        llm_call_count += 1
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Stage 6 failed: {e}")
        return {"correct_elements": [], "errors": [{"issue": "Validation failed"}]}


# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================


def main():
    global llm_call_count

    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        llm_call_count = 0

        print(f"========== STARTING PIPELINE: {img_path.name} ==========")
        timestamp = time.strftime("%H%M%S")
        base_name = img_path.stem

        raw_img_array = cv2.imread(str(img_path))

        # --- Stage 0 ---
        elements = run_stage0_extraction(img_path)
        with open(
            OUTPUT_DIR / f"{base_name}_stage0_elements_{timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(elements, f, indent=4, ensure_ascii=False)

        annotated_img = draw_annotations(raw_img_array, elements)
        cv2.imwrite(
            str(OUTPUT_DIR / f"{base_name}_stage0_annotated_{timestamp}.jpg"),
            annotated_img,
        )

        # --- Stage 1 ---
        hierarchy = run_stage1_vlm_hierarchy(annotated_img, elements)
        with open(
            OUTPUT_DIR / f"{base_name}_stage1_hierarchy_{timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(hierarchy, f, indent=4, ensure_ascii=False)

        # --- Stage 2 ---
        macro_connections = run_stage2_macro_routing(annotated_img, elements)
        with open(
            OUTPUT_DIR / f"{base_name}_stage2_macro_routing_{timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(macro_connections, f, indent=4, ensure_ascii=False)

        # --- Stage 3 ---
        micro_connections = run_stage3_agentic_loop(
            annotated_img, hierarchy, macro_connections
        )
        with open(
            OUTPUT_DIR / f"{base_name}_stage3_micro_routing_{timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(micro_connections, f, indent=4, ensure_ascii=False)

        # --- Stage 4 ---
        final_graph = run_stage4_consolidation(
            hierarchy, macro_connections, micro_connections
        )
        with open(
            OUTPUT_DIR / f"{base_name}_stage4_FINAL_GRAPH_{timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(final_graph, f, indent=4, ensure_ascii=False)

        # --- Stage 5 ---
        summary_json = run_stage5_summary(raw_img_array, final_graph)
        with open(
            OUTPUT_DIR / f"{base_name}_stage5_summary_{timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary_json, f, indent=4, ensure_ascii=False)

        # --- Stage 6 ---
        validation_json = run_stage6_validation(
            raw_img_array, final_graph, summary_json
        )
        with open(
            OUTPUT_DIR / f"{base_name}_stage6_validation_{timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(validation_json, f, indent=4, ensure_ascii=False)

        print(f"-> SUCCESS: Pipeline completed for {base_name}.")
        print(f"-> Total Azure OpenAI API Calls Made: {llm_call_count}")


if __name__ == "__main__":
    main()
