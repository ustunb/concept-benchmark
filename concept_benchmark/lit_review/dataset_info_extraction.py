"""
Concept Bottleneck Model Dataset Extraction Tool
Processes academic papers to extract dataset information using Claude API

Created with Claude cause why not
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import anthropic
from tqdm import tqdm
import PyPDF2
import hashlib
import glob
import os
import re


class CBMDatasetExtractor:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.datasets_db = {}
        self.processed_papers = set()

    def load_existing_database(self, filepath: str = "datasets_db.json"):
        """Load existing dataset database if it exists"""
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                self.datasets_db = json.load(f)
                print(f"Loaded {len(self.datasets_db)} existing datasets")

    def save_database(self, filepath: str = "datasets_db.json"):
        """Save current database state"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.datasets_db, f, indent=2, ensure_ascii=False)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def create_extraction_prompt(self, paper_text: str) -> str:
        """Create the full prompt with known datasets context"""

        # Summarize known datasets for context
        known_datasets_summary = {}
        for dataset_name, info in self.datasets_db.items():
            known_datasets_summary[dataset_name] = {
                "task_type": info.get("task_type", "unknown"),
                "y": info.get("y", "unknown"),
                "num_papers": len(info.get("papers", [])),
                "sizes": info.get("sizes", {})
            }

        prompt_template = """# Dataset Extraction Prompt for Concept Bottleneck Models

## Task
Extract dataset information from this academic paper about Concept Bottleneck Models. You will be provided with:
1. The paper text
2. A list of already-known datasets (if any)

## Output Format
Return a JSON object with the following structure:

```json
{
  "datasets": [
    {
      "dataset_name": "string",
      "is_new": boolean,
      "reference": "string (original citation from paper)",
      "y": "string (what is being predicted)",
      "c": "string or array (what are the concepts)", 
      "task_type": "string (classification/multiclass/regression/etc)",
      "sizes": {
        "n": "number or string (number of samples)",
        "m": "number or string (number of features/concepts)",
        "other": "string (any other size info)"
      },
      "properties": {
        "num_concepts": "number or string",
        "concept_source": "string (how concepts were chosen/defined)",
        "missingness": "string (info about missing data)",
        "other_properties": "array of strings"
      },
      "citekey": "string (if identifiable)",
      "usage_context": "string (how this dataset was used in this paper)",
      "confidence": "high/medium/low (confidence in extraction)"
    }
  ],
  "paper_info": {
    "title": "string",
    "authors": "string", 
    "venue": "string",
    "year": "number"
  }
}
```

## Known Datasets Context
""" + json.dumps(known_datasets_summary, indent=2) + """

## Instructions

### For Each Dataset Found:
1. **Dataset Identification**: Look for dataset names, typically in methodology sections, experiments, or results
2. **New vs Known**: 
   - If dataset appears in known datasets above, set `is_new: false`
   - Only extract NEW information not already captured
   - Focus on different usage contexts or additional properties
3. **Information Extraction**:
   - **y (target)**: What is being predicted? (e.g., "bird species", "scene categories", "medical diagnosis")
   - **c (concepts)**: What concepts are used? (e.g., "visual attributes like color, shape", "parts-based features", "clinical symptoms")
   - **task_type**: classification, multi-class, multi-label, regression, etc.
   - **sizes**: Extract n (samples), m (features/concepts), other relevant dimensions
   - **properties**: Number of concepts, how concepts were selected, missing data info, etc.

### Extraction Guidelines:
- **Be precise**: Extract exact quotes when possible
- **Handle ambiguity**: If information is unclear, note uncertainty in confidence field
- **Context matters**: Same dataset might be used differently across papers
- **Look for tables**: Dataset statistics often appear in tables or experimental setup sections
- **Citations**: Extract the exact citation as it appears in the paper
- **Naming variations**: Be aware of different naming conventions for the same dataset, e.g., "CUB-200", "Caltech-UCSD Birds", "CUB" is the same dataset!

### Common Dataset Names to Watch For:
CUB-200-2011, AwA2 (Animals with Attributes), CelebA, CIFAR-10/100, ImageNet, Places365, Pascal VOC, MS-COCO, Caltech-UCSD Birds, etc.

### Special Cases:
- **Synthetic datasets**: Note if dataset is synthetic/generated for the paper
- **Modified datasets**: If standard dataset is modified, note modifications
- **Multiple tasks**: If dataset used for multiple prediction tasks, create separate entries
- **Concept annotations**: Pay special attention to how concepts are annotated or chosen

## Paper Text:
""" + paper_text

        return prompt_template

    def extract_from_paper(self, paper_text: str, paper_id: str = None) -> Dict[str, Any]:
        """Extract dataset information from a single paper"""

        # Skip if already processed
        paper_hash = hashlib.md5(paper_text.encode('utf-8', errors='ignore').decode('utf-8').encode()).hexdigest()
        if paper_hash in self.processed_papers:
            print(f"Skipping already processed paper: {paper_id}")
            return {}

        prompt = self.create_extraction_prompt(paper_text)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=20000,
                messages=[{"role": "user", "content": prompt}]
            )

            unprocessed_text = response.content[0].text
            processed_text = re.sub(r'^.*?```json\s*', '', unprocessed_text, flags=re.DOTALL)
            # Parse JSON response
            result = json.loads(processed_text[:-3])
            self.processed_papers.add(paper_hash)
            return result

        except json.JSONDecodeError as e:
            print(f"JSON parsing error for paper {paper_id}: {e}")
            print(f"Raw response: {processed_text[:-3]}")
            return {}
        except Exception as e:
            print(f"API error for paper {paper_id}: {e}")
            return {}

    def merge_dataset_info(self, dataset_info: Dict[str, Any], paper_info: Dict[str, Any]):
        """Merge new dataset information with existing database"""
        dataset_name = dataset_info["dataset_name"]

        if dataset_name not in self.datasets_db:
            # New dataset
            self.datasets_db[dataset_name] = {
                "dataset_name": dataset_name,
                "references": [dataset_info.get("reference", "")],
                "y": dataset_info.get("y", ""),
                "c": dataset_info.get("c", ""),
                "task_type": dataset_info.get("task_type", ""),
                "sizes": dataset_info.get("sizes", {}),
                "properties": dataset_info.get("properties", {}),
                "citekeys": [dataset_info.get("citekey", "")],
                "papers": [paper_info],
                "usage_contexts": [dataset_info.get("usage_context", "")]
            }
        else:
            # Existing dataset - merge information
            existing = self.datasets_db[dataset_name]

            # Add paper reference
            existing["papers"].append(paper_info)
            existing["usage_contexts"].append(dataset_info.get("usage_context", ""))

            # Update fields if new information provided
            if dataset_info.get("reference") and dataset_info["reference"] not in existing["references"]:
                existing["references"].append(dataset_info["reference"])

            if dataset_info.get("citekey") and dataset_info["citekey"] not in existing["citekeys"]:
                existing["citekeys"].append(dataset_info["citekey"])

            # Update properties with new information
            if dataset_info.get("properties"):
                for key, value in dataset_info["properties"].items():
                    if key not in existing["properties"] or not existing["properties"][key]:
                        existing["properties"][key] = value

            # Update sizes with new information
            if dataset_info.get("sizes"):
                for key, value in dataset_info["sizes"].items():
                    if key not in existing["sizes"] or not existing["sizes"][key]:
                        existing["sizes"][key] = value

    def deduplicate_database(self, batch_size=200):
        """Use Claude API to identify and merge duplicate datasets"""

        dataset_names = list(self.datasets_db.keys())
        if len(dataset_names) < 2:
            print("Not enough datasets to deduplicate")
            return

        print(f"Starting deduplication of {len(dataset_names)} datasets...")

        # Process in batches to avoid token limits
        for i in range(0, len(dataset_names), batch_size):
            batch_names = dataset_names[i:i + batch_size]
            batch_data = {name: self.datasets_db[name] for name in batch_names}

            print(f"Processing batch {i // batch_size + 1}: {len(batch_names)} datasets")

            duplicates = self._find_duplicates_in_batch(batch_data)

            if duplicates:
                print(f"Found {len(duplicates)} duplicate groups in this batch")
                for duplicate_group in duplicates:
                    self._merge_duplicate_group(duplicate_group)
            else:
                print("No duplicates found in this batch")

        print(f"Deduplication complete. Final dataset count: {len(self.datasets_db)}")

    def _find_duplicates_in_batch(self, batch_data):
        """Use Claude API to identify duplicates in a batch of datasets"""

        # Create simplified representations for Claude analysis
        simplified_data = {}
        for name, data in batch_data.items():
            simplified_data[name] = {
                "dataset_name": data.get("dataset_name", name),
                "y": data.get("y", ""),
                "sizes": data.get("sizes", {}),
                "references": data.get("references", [])[:3],  # First 3 refs only
                "task_type": data.get("task_type", "")
            }

        prompt = f"""Analyze these datasets and identify duplicates. Look for:
1. Same dataset with different names (e.g., "CUB", "CUB-200-2011", "Caltech-UCSD Birds")
2. Same prediction target (y) and similar sizes
3. Similar references/citations
4. Variants of the same core dataset

Dataset Data:
{json.dumps(simplified_data, indent=2)}

Return ONLY a JSON list of duplicate groups. Each group should be a list of dataset names that are duplicates.
If no duplicates found, return an empty list [].

Example format:
[
  ["CUB-200-2011", "CUB", "Caltech-UCSD Birds-200-2011"],
  ["CIFAR-10", "CIFAR10", "CIFAR-10 (with CLIP concepts)"]
]

JSON OUTPUT:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            raw_text = response.content[0].text.strip()

            # Clean response
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:].strip()
            if raw_text.startswith("```"):
                raw_text = raw_text[3:].strip()
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

            duplicates = json.loads(raw_text)
            return duplicates

        except Exception as e:
            print(f"Error finding duplicates: {e}")
            return []

    def _merge_duplicate_group(self, duplicate_group):
        """Merge a group of duplicate datasets into one"""

        if len(duplicate_group) < 2:
            return

        # Use the first dataset as the base (or find the most complete one)
        base_name = duplicate_group[0]

        # Find the most complete dataset to use as base
        max_papers = 0
        for name in duplicate_group:
            if name in self.datasets_db:
                paper_count = len(self.datasets_db[name].get("papers", []))
                if paper_count > max_papers:
                    max_papers = paper_count
                    base_name = name

        if base_name not in self.datasets_db:
            print(f"Warning: Base dataset {base_name} not found")
            return

        base_dataset = self.datasets_db[base_name]
        print(f"Merging duplicates into: {base_name}")

        # Merge data from other datasets
        for dup_name in duplicate_group:
            if dup_name == base_name or dup_name not in self.datasets_db:
                continue

            print(f"  - Merging {dup_name}")
            dup_dataset = self.datasets_db[dup_name]

            # Merge references
            existing_refs = set(base_dataset.get("references", []))
            for ref in dup_dataset.get("references", []):
                if ref not in existing_refs:
                    base_dataset.setdefault("references", []).append(ref)
                    existing_refs.add(ref)

            # Merge citekeys
            existing_cites = set(base_dataset.get("citekeys", []))
            for cite in dup_dataset.get("citekeys", []):
                if cite not in existing_cites:
                    base_dataset.setdefault("citekeys", []).append(cite)
                    existing_cites.add(cite)

            # Merge papers
            existing_paper_paths = set()
            for paper in base_dataset.get("papers", []):
                existing_paper_paths.add(paper.get("file_path", ""))

            for paper in dup_dataset.get("papers", []):
                if paper.get("file_path", "") not in existing_paper_paths:
                    base_dataset.setdefault("papers", []).append(paper)
                    existing_paper_paths.add(paper.get("file_path", ""))

            # Merge usage contexts
            base_dataset.setdefault("usage_contexts", []).extend(
                dup_dataset.get("usage_contexts", [])
            )

            # Update fields if base has missing/less detailed info
            for field in ["y", "c", "task_type"]:
                if not base_dataset.get(field) and dup_dataset.get(field):
                    base_dataset[field] = dup_dataset[field]
                elif len(str(dup_dataset.get(field, ""))) > len(str(base_dataset.get(field, ""))):
                    base_dataset[field] = dup_dataset[field]

            # Merge sizes (keep most complete)
            if dup_dataset.get("sizes"):
                base_sizes = base_dataset.setdefault("sizes", {})
                for key, value in dup_dataset["sizes"].items():
                    if not base_sizes.get(key) and value:
                        base_sizes[key] = value

            # Merge properties
            if dup_dataset.get("properties"):
                base_props = base_dataset.setdefault("properties", {})
                for key, value in dup_dataset["properties"].items():
                    if not base_props.get(key) and value:
                        base_props[key] = value

            # Remove the duplicate
            del self.datasets_db[dup_name]

        print(
            f"  Final dataset has {len(base_dataset.get('papers', []))} papers and {len(base_dataset.get('usage_contexts', []))} usage contexts")

    def run_full_deduplication(self):
        """Run complete deduplication process with user confirmation"""

        original_count = len(self.datasets_db)
        print(f"Starting deduplication of {original_count} datasets...")

        # Backup original data
        backup_file = "datasets_db_backup.json"
        with open(backup_file, 'w') as f:
            json.dump(self.datasets_db, f, indent=2, ensure_ascii=False)
        print(f"Backup saved to {backup_file}")

        # Run deduplication
        self.deduplicate_database()

        final_count = len(self.datasets_db)
        print(f"\nDeduplication Results:")
        print(f"  Original: {original_count} datasets")
        print(f"  Final: {final_count} datasets")
        print(f"  Merged: {original_count - final_count} duplicates")

        # Save results
        self.save_database("datasets_db.json")
        self.export_to_csv("cbm_datasets.csv")

        return final_count

    def process_papers(self, paper_paths: List[str], save_frequency: int = 5):
        """Process multiple papers and update database"""

        for i, paper_path in enumerate(tqdm(paper_paths)):
            print(f"\nProcessing: {Path(paper_path).name}")

            # Extract text
            if paper_path.endswith('.pdf'):
                paper_text = self.extract_text_from_pdf(paper_path)
            else:
                with open(paper_path, 'r', encoding='utf-8') as f:
                    paper_text = f.read()

            if not paper_text.strip():
                print(f"No text extracted from {paper_path}")
                continue

            # Extract information
            result = self.extract_from_paper(paper_text, paper_id=Path(paper_path).name)

            if not result:
                continue

            # Process extracted datasets
            paper_info = result.get("paper_info", {})
            paper_info["file_path"] = paper_path

            for dataset_info in result.get("datasets", []):
                self.merge_dataset_info(dataset_info, paper_info)
                print(f"  - Found dataset: {dataset_info['dataset_name']}")

            # Save periodically
            if (i + 1) % save_frequency == 0:
                self.save_database()
                print(f"Saved progress after {i + 1} papers")

        # Final save
        self.save_database()
        print(f"\nCompleted processing {len(paper_paths)} papers")
        print(f"Total datasets in database: {len(self.datasets_db)}")

    def export_to_csv(self, filepath: str = "cbm_datasets.csv"):
        """Export database to CSV format"""
        rows = []
        for dataset_name, info in self.datasets_db.items():
            row = {
                "dataset_name": dataset_name,
                "y": info.get("y", ""),
                "c": str(info.get("c", "")),
                "task_type": info.get("task_type", ""),
                "n_samples": info.get("sizes", {}).get("n", ""),
                "n_concepts": info.get("sizes", {}).get("m", ""),
                "num_concepts": info.get("properties", {}).get("num_concepts", ""),
                "concept_source": info.get("properties", {}).get("concept_source", ""),
                "num_papers": len(info.get("papers", [])),
                "references": " | ".join(info.get("references", [])),
                "citekeys": " | ".join(filter(None, info.get("citekeys", [])))
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Exported to {filepath}")


# Usage example
if __name__ == "__main__":
    # Initialize extractor
    extractor = CBMDatasetExtractor(api_key="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")

    # Load any existing database
    extractor.load_existing_database()

    # Process papers
    papers_dir = "/Users/jskirzynski/Desktop/papers"

    # Load all files
    paper_files = glob.glob(os.path.join(papers_dir, "*.pdf"))
    #extractor.process_papers(paper_files)

    # Show current stats
    print(f"Current database has {len(extractor.datasets_db)} datasets")

    # Run deduplication
    final_count = extractor.run_full_deduplication()

    print(f"Deduplication complete! Reduced to {final_count} unique datasets")

    # Export results
    extractor.export_to_csv()