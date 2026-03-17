import os
import time
import json
import base64
import cv2
import networkx as nx
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeOutputOption,
    DocumentContentFormat,
)
from openai import AzureOpenAI

load_dotenv()

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

INPUT_DIR = Path("input")
date_str = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = Path(f"output/{date_str}/run-01")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Azure OpenAI Client
oai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION,
)

# ==========================================
# THE GLOBAL SYSTEM PROMPT (BFSI PERSONA)
# ==========================================
BFSI_EXPERT_SYSTEM_PROMPT = """
You are an Expert BFSI (Banking, Financial Services, and Insurance) Enterprise Infrastructure Architecture Diagram Analyzer. You possess deep, technical expertise in standard cloud practices (Azure, AWS, GCP), on-premise setups, and hybrid networking patterns.

Core Directives & Strict Constraints:
1. ZERO ASSUMPTION RULE: You must never infer, assume, or fabricate components, connections, or boundaries that are not explicitly depicted visually in the diagram. If a detail is missing or ambiguous, omit it rather than guessing.
2. PIXEL-PERFECT ATTENTION: Every line, arrow, and boundary matters. A simple line represents a critical network flow or dependency. Do not miss any physical connections.
3. BFSI & CLOUD STANDARDS: Apply standard enterprise architecture logic to your visual parsing. Recognize that Subnets reside within VNets/VPCs, Private Endpoints map to distinct resources, and network traffic is often governed by Firewalls, WAFs, and gateways.
4. VISUAL FOCUS: Rely strictly on the physical icons, shapes, and drawn boundaries. Always scan for a visual legend first. If present, strictly use it to decode colors, line types (e.g., dotted, solid, pipelines), and icons. If absent, rely strictly on explicit text labels.
5. CONFIDENCE SCORING: Every JSON object or nested property you output MUST include a "confidence_score" (a float between 0.0 and 1.0). 1.0 means it is explicitly labeled and visually unambiguous; lower scores indicate visual ambiguity (e.g., a line that crosses multiple boxes where the exact terminus is slightly obscured).
"""


# ==========================================
# UTILITIES
# ==========================================
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


def save_json(data, filename):
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"      [Saved] {filename}")


def call_vlm_agent(system_prompt, user_prompt, img_array, expected_format_desc=""):
    """Generic wrapper for calling the VLM with strict system prompts and JSON enforcement."""
    base64_image = encode_image_base64(img_array)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{user_prompt}\n\nRespond strictly in JSON format matching this description:\n{expected_format_desc}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ]
    response = oai_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0.0,  # Zero creativity, maximum determinism
    )
    return json.loads(response.choices[0].message.content)


# ==========================================
# STAGES 1-3: OCR & ANNOTATION
# ==========================================
def run_stages_1_to_3_ocr(image_path, img_array):
    print(f"\n-> [Stages 1-3] Azure Document Intelligence OCR on {image_path.name}...")
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
                    "raw_ocr_content": para.get("content"),
                    "bbox": get_bbox(coords),
                    "centroid": get_centroid(coords),
                }
            )
            counter += 1

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

    return elements, annotated


# ==========================================
# STAGE 4: VISUAL ICON EXTRACTION & MERGE
# ==========================================
def run_stage4_icon_extraction_and_merge(annotated_img, ocr_elements):
    print("-> [Stage 4] Visual Agent: Identifying cloud icons and shapes by ID...")

    prompt = """
    TASK: Look strictly at the bounding boxes with ID numbers overlaid on this image. 
    For each ID, identify the specific cloud service icon, generic IT icon, or physical shape.
    DO NOT read the text. Focus entirely on the visual symbol (e.g., is it an Azure SQL Database icon? A generic server rack? A firewall brick wall?).
    If there is no distinct icon and it is just a text box, label the visual_entity as "Text Box".
    """

    expected_format = """
    {
      "visual_icons": [
        {"id": "1", "visual_entity": "Azure SQL Database", "confidence_score": 0.95}
      ],
      "stage_confidence_score": 0.90
    }
    """

    result = call_vlm_agent(
        BFSI_EXPERT_SYSTEM_PROMPT, prompt, annotated_img, expected_format
    )
    icon_data = {item["id"]: item for item in result.get("visual_icons", [])}

    print("   -> Python Engine: Merging OCR text with Visual Icons...")
    verified_elements = []

    for el in ocr_elements:
        element_id = el["id"]
        ocr_text = el.get("raw_ocr_content", "").strip()

        visual_info = icon_data.get(
            element_id, {"visual_entity": "Unknown", "confidence_score": 0.0}
        )
        visual_entity = visual_info.get("visual_entity", "Unknown")
        conf_score = visual_info.get("confidence_score", 0.0)

        verified_elements.append(
            {
                "id": element_id,
                "ocr_text": ocr_text,
                "visual_icon": visual_entity,
                "bbox": el["bbox"],
                "centroid": el["centroid"],
                "confidence_score": conf_score,
            }
        )

    return verified_elements, result


# ==========================================
# STAGE 5: MACRO ARCHITECTURE CONTEXT
# ==========================================
def run_stage5_macro_context(raw_img_array):
    print("-> [Stage 5] Architect Agent: Establishing global environment context...")

    prompt = """
    TASK: Analyze the overall architecture diagram as a macro-system. Do not focus on micro-components.
    Evaluate the following based strictly on visual evidence:
    1. Hosting Type: Is it Cloud, On-Prem, or Hybrid? (Look for Datacenter / DR labels).
    2. Provider: Which cloud provider(s) are used based on icons/terminology?
    3. Access Method: How does the user/traffic enter the system? (e.g., Internet, Intranet, VPN, ExpressRoute, specific Firewalls/WAF).
    4. Application Tiering: Is this a 2-tier, 3-tier, or microservices application?
    5. Primary Boundary: What is the main network boundary encapsulating the workload (e.g., specific VPC, VNet)?
    """

    expected_format = """
    {
      "hosting_type": {"value": "Cloud/On-Prem/Hybrid", "confidence_score": 0.95},
      "provider": {"value": "Azure/AWS/None", "confidence_score": 0.95},
      "access_method": {"value": "Details", "confidence_score": 0.85},
      "application_tiering": {"value": "Details", "confidence_score": 0.80},
      "primary_boundary": {"value": "Details", "confidence_score": 0.90},
      "stage_confidence_score": 0.89
    }
    """

    return call_vlm_agent(
        BFSI_EXPERT_SYSTEM_PROMPT, prompt, raw_img_array, expected_format
    )


# ==========================================
# STAGE 6: COMPONENT & BEST PRACTICES EXTRACTION
# ==========================================
def run_stage6_component_extraction(annotated_img, context, verified_elements):
    print(
        "-> [Stage 6] Detail Extractor: Correlating BFSI components and network features..."
    )

    element_summary = [
        {"id": e["id"], "text": e["ocr_text"], "icon": e["visual_icon"]}
        for e in verified_elements
    ]

    prompt = f"""
    Context established: {json.dumps(context)}
    Verified Elements: {json.dumps(element_summary)}
    
    TASK: Using BFSI cloud networking principles, scan the image for specific structural boundaries and network gateways.
    Look strictly for explicit visual representations of:
    - Distinct Subnets (count them and map to IDs)
    - Private Endpoints / NICs (count them and map to IDs)
    - Open Ports explicitly mentioned in text
    - DC / DR
    DO NOT INFER. If a standard 3-tier architecture usually has a WAF, but it is not drawn, omit it.
    """

    expected_format = """
    {
      "subnets_identified": [{"name": "name", "associated_ids": ["1", "2"], "confidence_score": 0.95}],
      "private_endpoints": [{"name": "name", "associated_ids": ["3"], "confidence_score": 0.90}],
      "ports_explicitly_mentioned": ["443", "80"],
      "stage_confidence_score": 0.95
    }
    """

    return call_vlm_agent(
        BFSI_EXPERT_SYSTEM_PROMPT, prompt, annotated_img, expected_format
    )


# ==========================================
# STAGE 7: ROUTING & HIERARCHY (ORIGINAL FORMAT)
# ==========================================
def run_stage7_hierarchy_and_routing(annotated_img, verified_elements, macro_context):
    print(
        "-> [Stage 7] Router Agent: Mapping Parent-Child hierarchies and Flow connections..."
    )

    lookup_table = [
        {"id": e["id"], "text": e["ocr_text"], "icon": e["visual_icon"]}
        for e in verified_elements
    ]

    prompt = f"""
    Context: {json.dumps(macro_context)}
    Lookup Table: {json.dumps(lookup_table)}
    
    TASK 1 (Hierarchy): Look at the visual boundary boxes (rectangles, dashed borders, colored zones). 
    Establish the Parent-Child hierarchy. If an ID is physically enclosed inside a boundary that belongs to another ID, map it. If an item is not inside any boundary, its parent_id should be null.
    
    TASK 2 (Macro Connections): Trace lines connecting major outer boundaries, large containers, or standalone systems. Do not trace internal micro-lines here.
    
    TASK 3 (Micro Connections): Trace specific lines between distinct child resources (e.g., a line from a Web App to a Database inside a Subnet).
    
    Follow the pixels exactly. Do not invent connections based on BFSI best practices if they are not drawn.
    """

    expected_format = """
    {
      "hierarchy": [
        { "id": "child_id", "parent_id": "parent_id_or_null", "confidence_score": 0.95 }
      ],
      "macro_connections": [
        { "source_id": "ID", "target_id": "ID", "flow": "One-way | Bi-directional", "style_and_meaning": "color/type (meaning)", "confidence_score": 0.90 }
      ],
      "micro_connections": [
        { "source_id": "ID", "target_id": "ID", "flow": "One-way | Bi-directional", "style_and_meaning": "color/type (meaning)", "confidence_score": 0.85 }
      ],
      "stage_confidence_score": 0.90
    }
    """

    return call_vlm_agent(
        BFSI_EXPERT_SYSTEM_PROMPT, prompt, annotated_img, expected_format
    )


# ==========================================
# STAGE 8: GRAPH CONVERSION (NETWORKX)
# ==========================================
def run_stage8_graph_conversion(verified_elements, s7_routing):
    print("-> [Stage 8] Python Engine: Converting JSON routing to NetworkX Graph...")

    G = nx.DiGraph()

    # Map hierarchy parents
    hierarchy_dict = {
        str(item["id"]): str(item.get("parent_id"))
        for item in s7_routing.get("hierarchy", [])
        if item.get("parent_id")
    }

    # Add Nodes
    for el in verified_elements:
        node_id = str(el["id"])
        G.add_node(
            node_id,
            label=el["ocr_text"],
            icon_type=el["visual_icon"],
            parent_id=hierarchy_dict.get(node_id, None),
            confidence=el.get("confidence_score", 0.0),
        )

    # Add Edges (Combine Macro and Micro)
    all_edges = s7_routing.get("macro_connections", []) + s7_routing.get(
        "micro_connections", []
    )
    for edge in all_edges:
        G.add_edge(
            str(edge["source_id"]),
            str(edge["target_id"]),
            flow=edge.get("flow", "Unknown"),
            meaning=edge.get("style_and_meaning", "Unknown"),
            confidence=edge.get("confidence_score", 0.0),
        )

    # Export for Frontend compatibility (Node-link format)
    fe_graph = nx.node_link_data(G)
    return fe_graph


# ==========================================
# STAGE 9: CLEAN SUMMARY FOR FE
# ==========================================
def run_stage9_summary(fe_graph, context_data):
    print("-> [Stage 9] Presenter Agent: Generating human-readable flow summary...")

    prompt = f"""
    Context: {json.dumps(context_data)}
    Graph Data: {json.dumps(fe_graph)}
    
    TASK: Write a clean, professional, bullet-pointed summary of the architecture and network flow. 
    Explain how user traffic enters and traverses the system based purely on the structured graph data provided.
    Ensure the output is directly usable in a frontend UI.
    """

    response = oai_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": BFSI_EXPERT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f'{prompt}\n\nRespond strictly in JSON: {{\n"markdown_summary": "string",\n"confidence_score": 0.95\n}}',
            },
        ],
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)


# ==========================================
# STAGE 10: CRITIC / VALIDATION AGENT
# ==========================================
def run_stage10_critic(raw_img_array, summary_data, graph_data):
    print("-> [Stage 10] Critic Agent: Validating Final Output against Raw Image...")

    prompt = f"""
    TASK: You are an Independent Auditor. Compare the derived Graph Data and Summary against the RAW visual image. 
    
    Graph Data: {json.dumps(graph_data)}
    Summary: {summary_data.get('markdown_summary', '')}
    
    Be brutally honest. 
    - Are there missed lines or missing components? 
    - Did the previous stages hallucinate components that aren't there?
    - At which stage did the error likely occur (e.g., OCR missed it, Routing hallucinated it)?
    """

    expected_format = """
    {
      "is_accurate": true,
      "correct_elements_identified": ["List of things done right"],
      "gaps_identified": ["List of things missed"],
      "errors_and_hallucinations": ["List of missed lines or invented components"],
      "recommended_pipeline_corrections": ["Specific fixes needed and at what stage"],
      "overall_confidence_score": 0.95
    }
    """

    return call_vlm_agent(
        BFSI_EXPERT_SYSTEM_PROMPT, prompt, raw_img_array, expected_format
    )


# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def main():
    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        timestamp = time.strftime("%H%M%S")
        base_name = img_path.stem

        print(
            f"\n{'='*55}\nSTARTING 13-STAGE ARCHITECTURE PIPELINE: {img_path.name}\n{'='*55}"
        )

        raw_img_array = cv2.imread(str(img_path))

        try:
            # Stages 1-3
            ocr_elements, annotated_img = run_stages_1_to_3_ocr(img_path, raw_img_array)
            cv2.imwrite(
                str(OUTPUT_DIR / f"{base_name}_{timestamp}_stage3_annotated.jpg"),
                annotated_img,
            )

            # Stage 4
            verified_elements, s4_raw = run_stage4_icon_extraction_and_merge(
                annotated_img, ocr_elements
            )
            save_json(
                {"raw_vlm": s4_raw, "fused_elements": verified_elements},
                f"{base_name}_{timestamp}_stage4_verified.json",
            )

            # Stage 5
            s5_context = run_stage5_macro_context(raw_img_array)
            save_json(s5_context, f"{base_name}_{timestamp}_stage5_macro_context.json")

            # Stage 6
            s6_details = run_stage6_component_extraction(
                annotated_img, s5_context, verified_elements
            )
            save_json(
                s6_details, f"{base_name}_{timestamp}_stage6_extracted_details.json"
            )

            # Stage 7
            s7_routing = run_stage7_hierarchy_and_routing(
                annotated_img, verified_elements, s5_context
            )
            save_json(s7_routing, f"{base_name}_{timestamp}_stage7_routing.json")

            # Stage 8
            s8_fe_graph = run_stage8_graph_conversion(verified_elements, s7_routing)
            save_json(
                s8_fe_graph, f"{base_name}_{timestamp}_stage8_networkx_graph.json"
            )

            # Stage 9
            s9_summary = run_stage9_summary(s8_fe_graph, s5_context)
            save_json(s9_summary, f"{base_name}_{timestamp}_stage9_summary.json")
            print(
                f"\n--- FE SUMMARY ---\n{s9_summary.get('markdown_summary')}\n------------------\n"
            )

            # Stage 10
            s10_validation = run_stage10_critic(raw_img_array, s9_summary, s8_fe_graph)
            save_json(
                s10_validation,
                f"{base_name}_{timestamp}_stage10_critic_validation.json",
            )

            print(
                f"\n-> SUCCESS: Pipeline complete for {base_name}. All JSON artifacts saved."
            )

        except Exception as e:
            print(f"\n[!] PIPELINE FAILED at {base_name}: {str(e)}")


if __name__ == "__main__":
    main()
