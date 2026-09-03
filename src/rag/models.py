from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Element:

    # ========================================================
    # Document identity
    # ========================================================

    document_id: str
    document_name: str
    document_hash: str


    # ========================================================
    # Element identity
    # ========================================================

    element_id: str
    element_type: str
    source_ref: Optional[str] = None

    # Optional semantic subtype
    #
    # Examples:
    #   figure
    #   chart
    #   diagram
    #   equation
    #   photo
    #
    element_subtype: Optional[str] = None


    # ========================================================
    # Content
    # ========================================================

    content: Optional[str] = None

    caption: Optional[str] = None


    # ========================================================
    # Location
    # ========================================================

    page_number: Optional[int] = None


    # ========================================================
    # Relationships
    # ========================================================

    parent_element_id: Optional[str] = None

    related_element_ids: list[str] = field(
        default_factory=list
    )


    # ========================================================
    # Asset
    # ========================================================

    asset_path: Optional[str] = None


    # ========================================================
    # Chunk information
    # ========================================================

    chunk_id: Optional[str] = None

    chunk_index: Optional[int] = None


    # ========================================================
    # Parser
    # ========================================================

    parser: str = "docling"


    def to_dict(self) -> dict:

        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None
        }