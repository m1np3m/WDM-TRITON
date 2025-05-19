from services.llm import LLMService
import json
import os

class LLMParser:
    def __init__(self, image_paths: list[str]):
        self.llm_service = LLMService()
        self.image_paths = image_paths

        self.providers = [{"name": "gemini-image", "model": "gemini-2.5-flash-preview-04-17", "temperature": 0.9, "retry": 3}]

        self.SYSTEM_PARSING_PROMPT = """
        ### ROLE
            - You are a document parsing tool.

        ### INSTRUCTIONS
            - You will receive image documents that can contain many different objects such as: text, images, tables, charts, ...
            - The parsing order from top to bottom needs to be re-marked.
            - You must extract all data that appears in the document without summarizing.
            - You need to pay special attention to tables that contain many different types of objects such as images, text. Try to extract that type of table for me.

        ### REQUIREMENTS
            - For table objects:
                + You must extract all table information in that document, distinguishing between different tables.
                + Even if the amount of data in the table is large, you must export it all.

            - For image data:
                + You only need to give a general description for that image.

        ### OUTPUT
            - The parsing format of the table is csv, the rest is Markdown.
            - Your OUTPUT must follow the JSON format, in which:
                + key: type of object + ordinal number from top to bottom.
                + value: represents the parsed data.
	        
            - You must parse the content in the document, absolutely do not make up other content.

            For example:
            {
                "table_1": "...",
                "image_2": "...",
                "text_3": "..."
            }
        """

    def parse(self, output_path: str) -> str:
        for image_path in self.image_paths:
            for attempt in range(3):
                parsing_results = self.llm_service.complete(
                                system_prompt=self.SYSTEM_PARSING_PROMPT,
                                user_prompt="Please parse the content of this document image in detail",
                                image_paths=[image_path],
                                json_output=True,
                                providers=self.providers,
                                )
                
                json_result = json.loads(json.dumps(parsing_results))

                if not json_result:
                    print(f"Failed to parse image {image_path} after {attempt + 1} attempts")
                    continue
                break

            # write json_result to a file
            os.makedirs(output_path, exist_ok=True)
            image_name = os.path.basename(image_path)

            with open(f"{os.path.join(output_path, image_name)}.json", "w") as f:
                json.dump(json_result, f, indent=4)