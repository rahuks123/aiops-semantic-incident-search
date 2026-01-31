import gradio as gr
import boto3
import json
from rag import retrieve

CONFIDENCE_THRESHOLD = 0.6
BEDROCK_REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def call_bedrock(prompt):
    response = bedrock.invoke_model(
        modelId="amazon.nova-lite-v1:0",
        body=json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
        }),
        contentType="application/json",
        accept="application/json"
    )



    result = json.loads(response["body"].read())

    # Defensive extraction
    # if "content" not in result or not result["content"]:
    #     return "No response from model."


    return result['output']['message']['content'][0]['text']

def build_fallback_prompt(query):
    return f"""
You are a senior DevOps engineer.

No similar historical incidents were found in the knowledge base.

Provide GENERAL troubleshooting guidance and best practices
for investigating the issue below.
Do NOT claim this is the root cause.
Do NOT reference past incidents.

Issue:
{query}

Response:
- Possible causes
- Initial checks
- Safe diagnostic steps
"""

def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Source: {c['source']} | Type: {c['type']} | Chunk: {c['chunk']}]\n{c['text']}"
        for c in chunks
    )

    return f"""
You are an AIOps incident RCA assistant.

Use ONLY the context below to perform root cause analysis.
Reference the sources explicitly.
Do not invent causes.

Context:
{chunks}

User query:
{query}

Answer:
"""


def rag_ui(query, k):
    retrieved = retrieve(query, k)

    max_score = max(r["score"] for r in retrieved)
    print("Max retrieval score:", max_score)

    if max_score >= CONFIDENCE_THRESHOLD:
        # Grounded RAG mode
        prompt = build_prompt(query, retrieved)
        answer = call_bedrock(prompt)

        mode_label = "Grounded Answer (based on historical incidents/runbooks)"

        citations = []
        for r in retrieved:
            citations.append(
                f"Source: {r['source']} | Type: {r['type']} | "
                f"Chunk: {r['chunk']} | Score: {r['score']:.3f}"
            )

        sources_text = "\n".join(citations)

    else:
        # Fallback guidance mode
        prompt = build_fallback_prompt(query)
        answer = call_bedrock(prompt)

        mode_label = (
            "No strong historical match found.\n"
            "Showing general troubleshooting guidance (not RCA)."
        )

        sources_text = "No relevant historical incidents or runbooks found."

    return answer, sources_text, mode_label


with gr.Blocks(title="AIOps Incident RCA Assistant") as demo:
    gr.Markdown("# 🔧 AIOps Incident RCA Assistant")

    query = gr.Textbox(label="Incident Query")
    k = gr.Slider(1, 5, value=3, step=1, label="Top K Retrieved Chunks")

    mode = gr.Textbox(label="Answer Mode", interactive=False)
    answer = gr.Textbox(label="LLM Response", lines=8)
    sources = gr.Textbox(label="Sources / Context", lines=10)

    btn = gr.Button("Analyze Incident")
    btn.click(
        rag_ui,
        inputs=[query, k],
        outputs=[answer, sources, mode]
    )

demo.launch()