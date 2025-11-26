# FILE: app/processors/image_processor.py
"""
Image processor using Vision LLM for description generation.
Converts images to searchable text descriptions.
"""

from pathlib import Path
from typing import List, Optional
import logging
import base64

from app.processors.base import BaseProcessor, Document
from app.config import OPENAI_API_KEY, VISION_MODEL, VISION_DESCRIPTION_ENABLED

logger = logging.getLogger(__name__)


class ImageProcessor(BaseProcessor):
    """
    Process images using Vision LLM.
    
    Features:
    - Generates detailed text descriptions of images
    - Extracts visible text (OCR-like)
    - Identifies charts, diagrams, and their content
    - Falls back to basic metadata if API unavailable
    """
    
    supported_extensions = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = VISION_MODEL,
        enabled: bool = VISION_DESCRIPTION_ENABLED
    ):
        """
        Initialize image processor.
        
        Parameters
        ----------
        api_key : Optional[str]
            OpenAI API key for vision model.
        model : str
            Vision model to use (default: gpt-4o).
        enabled : bool
            Whether vision processing is enabled.
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.enabled = enabled
    
    def can_process(self, file_path: Path) -> bool:
        """Check if file is an image."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def process(self, file_path: Path) -> List[Document]:
        """
        Process image and generate text description.
        
        Parameters
        ----------
        file_path : Path
            Path to image file.
            
        Returns
        -------
        List[Document]
            Single document with image description.
        """
        file_path = Path(file_path)
        
        # Get image metadata
        metadata = self._get_image_metadata(file_path)
        
        # Generate description if enabled
        if self.enabled and self.api_key:
            description = self._generate_description(file_path)
        else:
            description = self._basic_description(file_path, metadata)
        
        if description:
            doc = self._create_document(
                content=description,
                source_file=file_path.name,
                file_type="image",
                image_format=file_path.suffix.lower().replace(".", ""),
                **metadata,
            )
            logger.info(f"Processed image {file_path.name}")
            return [doc]
        
        return []
    
    def _generate_description(self, file_path: Path) -> str:
        """
        Generate description using Vision LLM.
        
        Parameters
        ----------
        file_path : Path
            Path to image file.
            
        Returns
        -------
        str
            Generated description.
        """
        from openai import OpenAI
        
        try:
            # Read and encode image
            with open(file_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine MIME type
            suffix = file_path.suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }
            mime_type = mime_types.get(suffix, "image/png")
            
            # Call Vision API
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image comprehensively for a RAG system.
                                
Provide:
1. **Overall Description**: What the image shows
2. **Text Content**: Any visible text, labels, or numbers (transcribe exactly)
3. **Data/Charts**: If it's a chart/graph, describe the data, axes, trends
4. **Key Details**: Important elements, colors, relationships
5. **Context**: What domain/topic this image relates to

Be detailed and specific. This description will be used for semantic search."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Vision API error for {file_path}: {e}")
            return self._basic_description(file_path, self._get_image_metadata(file_path))
    
    def _basic_description(self, file_path: Path, metadata: dict) -> str:
        """
        Generate basic description from metadata.
        
        Parameters
        ----------
        file_path : Path
            Path to image file.
        metadata : dict
            Image metadata.
            
        Returns
        -------
        str
            Basic description.
        """
        desc_parts = [
            f"Image file: {file_path.name}",
            f"Format: {file_path.suffix.upper().replace('.', '')}",
        ]
        
        if "width" in metadata and "height" in metadata:
            desc_parts.append(f"Dimensions: {metadata['width']}x{metadata['height']} pixels")
        
        if "size_kb" in metadata:
            desc_parts.append(f"Size: {metadata['size_kb']:.1f} KB")
        
        return "\n".join(desc_parts)
    
    def _get_image_metadata(self, file_path: Path) -> dict:
        """
        Extract image metadata.
        
        Parameters
        ----------
        file_path : Path
            Path to image file.
            
        Returns
        -------
        dict
            Image metadata.
        """
        metadata = {
            "size_kb": file_path.stat().st_size / 1024
        }
        
        try:
            from PIL import Image
            
            with Image.open(file_path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["mode"] = img.mode
                
                # Extract EXIF data if available
                if hasattr(img, "_getexif") and img._getexif():
                    exif = img._getexif()
                    # Store relevant EXIF fields
                    
        except ImportError:
            logger.warning("PIL not installed, skipping image metadata")
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {e}")
        
        return metadata
