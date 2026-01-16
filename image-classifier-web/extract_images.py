#!/usr/bin/env python3
"""
PDF Image Extraction Tool

Extracts embedded images from Dutch towing/parking case report PDFs,
specifically targeting the "Bijlage: foto's" appendix section.

Usage:
    python extract_images.py --pdf "path/to/file.pdf" --out_dir "./data"
"""

import argparse
import json
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)


class ImageMetadata(TypedDict):
    """Metadata for an extracted image."""
    file: str
    page: int
    index_on_page: int
    method: str
    width: int
    height: int


def get_unique_filename(path: Path) -> Path:
    """
    Return a unique filename by appending _v2, _v3, etc. if file exists.

    Args:
        path: The desired file path

    Returns:
        A unique path that doesn't exist
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    version = 2
    while True:
        new_path = parent / f"{stem}_v{version}{suffix}"
        if not new_path.exists():
            return new_path
        version += 1


def extract_embedded_images(
    doc: fitz.Document,
    page_num: int,
    pdf_stem: str,
    out_dir: Path,
    min_size: int = 200
) -> tuple[list[ImageMetadata], int]:
    """
    Extract embedded images from a PDF page.

    Args:
        doc: PyMuPDF document object
        page_num: Page number (0-indexed)
        pdf_stem: PDF filename stem for naming output files
        out_dir: Output directory path
        min_size: Minimum dimension to keep (filters small icons)

    Returns:
        Tuple of (list of image metadata, count of filtered small images)
    """
    page = doc[page_num]
    images_metadata: list[ImageMetadata] = []
    filtered_count = 0

    # Get all images on this page
    image_list = page.get_images(full=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]

        try:
            # Extract image data
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width = base_image["width"]
            height = base_image["height"]

            # Filter small images (likely icons)
            if width < min_size or height < min_size:
                filtered_count += 1
                continue

            # Convert to JPEG for consistency
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Generate filename: {pdf_stem}_p{page:02d}_img{index:02d}.jpg
            filename = f"{pdf_stem}_p{page_num + 1:02d}_img{img_index + 1:02d}.jpg"
            filepath = get_unique_filename(out_dir / filename)

            # Save image
            img.save(filepath, "JPEG", quality=95)

            images_metadata.append({
                "file": filepath.name,
                "page": page_num + 1,
                "index_on_page": img_index + 1,
                "method": "embedded",
                "width": width,
                "height": height
            })

        except Exception as e:
            print(f"  Warning: Failed to extract image {img_index + 1} on page {page_num + 1}: {e}")
            continue

    return images_metadata, filtered_count


def render_page_fallback(
    doc: fitz.Document,
    page_num: int,
    pdf_stem: str,
    out_dir: Path,
    dpi: int = 250
) -> ImageMetadata | None:
    """
    Render a full page as an image (fallback when embedded extraction fails).

    Args:
        doc: PyMuPDF document object
        page_num: Page number (0-indexed)
        pdf_stem: PDF filename stem for naming output files
        out_dir: Output directory path
        dpi: DPI for rendering (default 250)

    Returns:
        Image metadata dict or None if rendering fails
    """
    try:
        page = doc[page_num]

        # Calculate scale factor (72 DPI is PDF default)
        scale = dpi / 72
        mat = fitz.Matrix(scale, scale)

        # Render page to pixmap
        pixmap = page.get_pixmap(matrix=mat)

        # Generate filename: {pdf_stem}_p{page:02d}_page.jpg
        filename = f"{pdf_stem}_p{page_num + 1:02d}_page.jpg"
        filepath = get_unique_filename(out_dir / filename)

        # Convert to PIL Image and save as JPEG
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        img.save(filepath, "JPEG", quality=95)

        return {
            "file": filepath.name,
            "page": page_num + 1,
            "index_on_page": 0,
            "method": "rendered_page_fallback",
            "width": pixmap.width,
            "height": pixmap.height
        }

    except Exception as e:
        print(f"  Warning: Failed to render page {page_num + 1}: {e}")
        return None


def write_manifest(
    out_dir: Path,
    pdf_path: Path,
    images: list[ImageMetadata],
    notes: list[str]
) -> None:
    """
    Write manifest.json with extraction metadata.

    Args:
        out_dir: Output directory path
        pdf_path: Source PDF path
        images: List of image metadata dicts
        notes: List of warning/info notes
    """
    manifest = {
        "source_pdf": str(pdf_path.resolve()),
        "extraction_time_utc": datetime.now(timezone.utc).isoformat(),
        "images": images,
        "notes": notes
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    """CLI entrypoint for PDF image extraction."""
    parser = argparse.ArgumentParser(
        description="Extract images from PDF files (Dutch towing reports)"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to input PDF file"
    )
    parser.add_argument(
        "--out_dir",
        default="./data",
        help="Output directory (default: ./data)"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=200,
        help="Minimum image dimension to extract (default: 200)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="DPI for page rendering fallback (default: 250)"
    )
    parser.add_argument(
        "--render_fallback",
        action="store_true",
        help="Also render pages as images (useful if embedded extraction fails)"
    )

    args = parser.parse_args()

    # Validate PDF exists
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get PDF stem for naming
    pdf_stem = pdf_path.stem
    # Sanitize stem (remove special chars that might cause issues)
    pdf_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in pdf_stem)

    print(f"Processing: {pdf_path.name}")
    print(f"Output directory: {out_dir.resolve()}")
    print("-" * 50)

    # Open PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error: Could not open PDF: {e}")
        sys.exit(1)

    # Track results
    all_images: list[ImageMetadata] = []
    notes: list[str] = []
    total_filtered = 0
    rendered_pages = 0

    # Process each page
    for page_num in range(len(doc)):
        print(f"Page {page_num + 1}/{len(doc)}...")

        # Primary: Extract embedded images
        images, filtered = extract_embedded_images(
            doc, page_num, pdf_stem, out_dir, args.min_size
        )
        all_images.extend(images)
        total_filtered += filtered

        if images:
            print(f"  Extracted {len(images)} embedded image(s)")

        # Fallback: Render page if requested or if no embedded images found
        if args.render_fallback or (not images and page_num >= 2):
            # Only render pages 3+ by default (where photos typically are)
            if args.render_fallback or page_num >= 2:
                rendered = render_page_fallback(
                    doc, page_num, pdf_stem, out_dir, args.dpi
                )
                if rendered:
                    all_images.append(rendered)
                    rendered_pages += 1
                    print(f"  Rendered full page as fallback")

    # Store page count before closing
    total_pages = len(doc)
    doc.close()

    # Add notes
    if total_filtered > 0:
        notes.append(f"Filtered {total_filtered} small image(s) (< {args.min_size}px)")

    # Write manifest
    write_manifest(out_dir, pdf_path, all_images, notes)

    # Print summary
    embedded_count = sum(1 for img in all_images if img["method"] == "embedded")

    print("-" * 50)
    print("=== Extraction Summary ===")
    print(f"PDF: {pdf_path.name}")
    print(f"Pages processed: {total_pages}")
    print(f"Embedded images extracted: {embedded_count}")
    print(f"Small icons filtered: {total_filtered}")
    print(f"Rendered pages: {rendered_pages}")
    print(f"Total images saved: {len(all_images)}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Manifest: {out_dir.resolve() / 'manifest.json'}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
