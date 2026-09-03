from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)

from docling.datamodel.base_models import (
    InputFormat,
)


class DoclingParser:

    def __init__(self):

        pipeline_options = PdfPipelineOptions()

        # Required for our multimodal image pipeline
        pipeline_options.generate_picture_images = True

        # We don't need full-page images
        pipeline_options.generate_page_images = False

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )


    def parse(
        self,
        source: Path,
    ):

        result = self.converter.convert(
            source
        )

        return result.document


    def save_json(
        self,
        document,
        output_path: Path,
    ):

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        document.save_as_json(
            output_path,
            indent=2,
        )