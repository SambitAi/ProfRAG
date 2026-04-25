from __future__ import annotations

from pathlib import Path

from core.llm import get_openai_client
from core.metadata import load_metadata, mark_step_success
from core.paths import build_images_dir, build_tables_dir
from core.storage import read_json, write_json


def _build_index_context(retrieval_payload: dict, document_folder: Path) -> list[str]:
    """Read images/tables index.json and inject metadata for assets in retrieved chunks."""
    images_index: dict = read_json(build_images_dir(document_folder) / "index.json", default={})
    tables_index: dict = read_json(build_tables_dir(document_folder).resolve() / "index.json", default={})

    image_by_name = {Path(v["abs_path"]).name: v for v in images_index.values() if v.get("abs_path")}
    table_by_name = {Path(v["csv_path"]).name: v for v in tables_index.values() if v.get("csv_path")}

    blocks: list[str] = []
    seen: set[str] = set()

    for meta in retrieval_payload.get("metadatas", []):
        chunk_path = meta.get("chunk_path", "")
        if not chunk_path:
            continue
        chunk = read_json(chunk_path, default={})

        for img_path in chunk.get("image_paths", []):
            fname = Path(img_path).name
            if fname in seen or fname not in image_by_name:
                continue
            seen.add(fname)
            info = image_by_name[fname]
            parts = []
            if info.get("title"):
                parts.append(f"Title: {info['title']}")
            if info.get("caption"):
                parts.append(f"Caption: {info['caption']}")
            ocr = info.get("ocr_text")
            if ocr and ocr != info.get("caption"):
                parts.append(f"OCR text: {ocr}")
            if info.get("prev_text"):
                parts.append(f"Context before: {info['prev_text']}")
            if info.get("next_text"):
                parts.append(f"Context after: {info['next_text']}")
            if info.get("page"):
                parts.append(f"Page: {info['page']}")
            if parts:
                blocks.append(f"[Image: {fname}]\n" + "\n".join(parts))

        for tbl_path in chunk.get("table_paths", []):
            fname = Path(tbl_path).name
            if fname in seen or fname not in table_by_name:
                continue
            seen.add(fname)
            info = table_by_name[fname]
            parts = []
            if info.get("title"):
                parts.append(f"Title: {info['title']}")
            if info.get("headers"):
                parts.append(f"Columns: {info['headers']}")
            if info.get("prev_text"):
                parts.append(f"Context before: {info['prev_text']}")
            if info.get("next_text"):
                parts.append(f"Context after: {info['next_text']}")
            if info.get("page"):
                parts.append(f"Page: {info['page']}")
            if parts:
                blocks.append(f"[Table: {fname}]\n" + "\n".join(parts))

    return blocks


def build_prompt(
    document_name: str,
    question: str,
    retrieval_payload: dict,
    media_instruction: str = "",
    index_context: list[str] | None = None,
) -> str:
    context_blocks: list[str] = []
    documents = retrieval_payload.get("documents", [])
    metadatas = retrieval_payload.get("metadatas", [])

    for index, document in enumerate(documents, start=1):
        metadata = metadatas[index - 1] if index - 1 < len(metadatas) else {}
        section_path = metadata.get("section_path", "")
        if section_path:
            source_label = f"{document_name} > {section_path}"
        else:
            source_label = f"{document_name}, Chunk: {metadata.get('chunk_number', 'unknown')}"
        context_blocks.append(f"[Source {index}] {source_label}\n{document}")

    # Directly retrieved media items
    for item in retrieval_payload.get("media_items", []):
        item_type = item.get("item_type", "")
        if item_type == "table":
            context_blocks.append(f"[Table from {document_name}]\n{item['document']}")
        elif item_type == "image":
            caption = item.get("caption", "")
            if caption:
                context_blocks.append(f"[Image from {document_name}]: {caption}")

    # All document images
    for img in retrieval_payload.get("images_meta", []):
        label = img.get("title") or img.get("caption") or img.get("prev_text") or ""
        if label:
            context_blocks.append(f"[Document Image from {document_name}]: {label}")

    # Parent-section expanded media
    section_media = retrieval_payload.get("section_media", {})
    for table_path in section_media.get("table_paths", []):
        try:
            csv_text = Path(table_path).read_text(encoding="utf-8")[:800]
            context_blocks.append(f"[Section Table]\n{csv_text}")
        except OSError:
            pass

    # Index metadata for assets in retrieved chunks
    if index_context:
        context_blocks.extend(index_context)

    context = "\n\n".join(context_blocks)
    media_note = f"\n\n{media_instruction.strip()}" if media_instruction else ""
    return (
        "You are a helpful assistant answering questions about uploaded PDF documents.\n"
        "Use only the provided context to answer. If the answer is not in the context, say you do not know.\n"
        "The context includes [Image: filename] and [Table: filename] blocks — these contain metadata "
        "(title, OCR text, captions, surrounding text) extracted from the document's image and table indexes. "
        "Use them to answer questions about visual or tabular content.\n"
        f"When possible, mention the relevant filename and section.{media_note}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def run(
    retrieval_input_path: str | Path,
    output_answer_path: str | Path,
    document_folder: str | Path,
    chat_model: str,
    media_instruction: str = "",
) -> str:
    client = get_openai_client()
    document_folder = Path(document_folder)
    retrieval_payload = read_json(retrieval_input_path, default={})
    metadata = load_metadata(document_folder)

    index_context = _build_index_context(retrieval_payload, document_folder)

    prompt = build_prompt(
        document_name=metadata.get("document_name", "Unknown Document"),
        question=retrieval_payload.get("question", ""),
        retrieval_payload=retrieval_payload,
        media_instruction=media_instruction,
        index_context=index_context,
    )

    response = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Answer the user's question using the retrieved document context only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content or "I could not generate an answer."
    answer_payload = {
        "question": retrieval_payload["question"],
        "answer": answer,
        "sources": retrieval_payload.get("metadatas", []),
        "media_items": retrieval_payload.get("media_items", []),
        "section_image_paths": retrieval_payload.get("section_media", {}).get("image_paths", []),
    }
    write_json(output_answer_path, answer_payload)
    mark_step_success(document_folder, "chat_response", {"output_answer_path": str(output_answer_path)})
    return str(output_answer_path)
