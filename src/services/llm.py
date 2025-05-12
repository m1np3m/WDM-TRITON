from google.genai.types import HarmCategory, HarmBlockThreshold, SafetySetting, GenerateContentConfig
from google import genai
from openai import OpenAI
import json
from loguru import logger
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

class LLMService:
    
    def show_log(self, message, level: str = "info"):
        if level == "debug" and os.getenv('DEBUG'):
            logger.debug(str(message))
        elif level == "error":
            logger.error(str(message))
        else:
            logger.info(str(message))
            
    def complete(self, providers: list, system_prompt: str, user_prompt: str, json_output: bool = False, limit_length: bool = False, image_paths: list[str] = None):
        if not providers:
            raise Exception("Providers is empty")
        try:
            is_success, response, error = False, None, None
            for provider in providers:
                # if provider["name"] == "openai":
                #     is_success, response, error = self.__call_openai(
                #         system_prompt=system_prompt,
                #         user_prompt=user_prompt,
                #         model=provider["model"],
                #         retry=provider["retry"],
                #         json_output=json_output,
                #         temperature=provider.get("temperature", 0.0),
                #         limit_length=limit_length,
                #         image_url=image_url
                #     )
                #     if is_success:
                #         return response

                if provider["name"] == "gemini-image":
                    is_success, response, error = self.__call_gemini_image(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=provider["model"],
                        retry=provider["retry"],
                        json_output=json_output,
                        temperature=provider.get("temperature", 0.0),
                        limit_length=limit_length,
                        image_paths=image_paths
                    )
                    if is_success:
                        return response

            #     if provider["name"] == "gemini":
            #         is_success, response, error = self.__call_gemini(
            #             system_prompt=system_prompt,
            #             user_prompt=user_prompt,
            #             model=provider["model"],
            #             retry=provider["retry"],
            #             json_output=json_output,
            #             temperature=provider.get("temperature", 0.0),
            #             limit_length=limit_length,
            #             image_url=image_url
            #         )
            #         if is_success:
            #             return response
            # if not is_success:
            #     raise Exception(error)

            return response
        except Exception as ex:
            self.show_log(message=f"Fail to call complete with ex: {ex}", level="error")
            return None

    def __call_gemini(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        model: str, 
        retry: int = 3, 
        json_output: bool = True, 
        temperature: float = 0.0, 
        limit_length: bool = False,
        image_url: str = None  # New optional parameter
    ):
        """
        Calls the Gemini API with the given prompts and optional image.

        Args:
            system_prompt (str): The system-level prompt.
            user_prompt (str): The user-level prompt.
            model (str): The OpenAI model to use.
            retry (int, optional): Number of retry attempts. Defaults to 3.
            json_output (bool, optional): Whether to parse the response as JSON. Defaults to True.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            limit_length (bool, optional): Whether to enforce a length limit. Defaults to False.
            image_url (Optional[str], optional): Url to the image. Defaults to None.

        Returns:
            Tuple[bool, Optional[dict or str], Optional[str]]: Success flag, result, and error message.
        """
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

            self.show_log(message=f"__call_gemini", level="info")
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": 8192,
            }
            
            if json_output:
                generation_config["response_mime_type"] = "application/json"
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            model_llm = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_prompt,
            )
            
            message = [{'role': 'user', 'parts': [user_prompt]}]
            
            if image_url:
                message.append({'role': 'user', 'parts': [f"Image: {image_url}"]})
                self.show_log(message=f"Image data added to the request: {image_url}", level="debug")

            responses = model_llm.generate_content(message, request_options={'timeout': 600})
            full_response_text = ''
            try:
                for response in responses:
                    if response.text:
                        full_response_text += response.text
            except AttributeError:
                for response in responses:
                    for part in response.parts:
                        full_response_text += part.text
            
            if json_output:
                try:
                    result = json.loads(full_response_text)
                except json.JSONDecodeError as jde:
                    raise Exception(f"JSON decode error: {jde}")
            else:
                result = full_response_text
            
            if not result:
                raise Exception("Empty response")

            return True, result, None

        except Exception as ex:

            if retry <= 0:
                self.show_log(message=f"Fail to call __call_gemini with ex: {ex}, retry: {retry}", level="error")
                return False, None, str(ex)
            
            self.show_log(message=f"__call_gemini -> Retry {retry}", level="error")

            retry -= 1

            return self.__call_gemini(
                system_prompt=system_prompt, 
                user_prompt=user_prompt, 
                model=model, 
                retry=retry,
                json_output=json_output, 
                temperature=temperature, 
                limit_length=limit_length,
                image_url=image_url  # Pass the image_url in the recursive call
            )
            
    def __call_gemini_image(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        model: str, 
        retry: int = 3, 
        json_output: bool = True, 
        temperature: float = 0.0, 
        limit_length: bool = False,
        image_paths: list[str] = None
    ):
        """
        Calls the Gemini API with the given prompts and optional image.

        Args:
            system_prompt (str): The system-level prompt.
            user_prompt (str): The user-level prompt.
            model (str): The OpenAI model to use.
            retry (int, optional): Number of retry attempts. Defaults to 3.
            json_output (bool, optional): Whether to parse the response as JSON. Defaults to True.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            limit_length (bool, optional): Whether to enforce a length limit. Defaults to False.
            image_paths (list[str], optional): List of image paths. Defaults to None.

        Returns:
            Tuple[bool, Optional[dict or str], Optional[str]]: Success flag, result, and error message.
        """
        try:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

            self.show_log(message=f"__call_gemini_image", level="info")
            

            generation_config = GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=8192,
                top_p=0.95,
                top_k=40,
                response_mime_type="application/json" if json_output else "text/plain",
                safety_settings=[
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=HarmBlockThreshold.BLOCK_NONE
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=HarmBlockThreshold.BLOCK_NONE
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=HarmBlockThreshold.BLOCK_NONE
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=HarmBlockThreshold.BLOCK_NONE
                    )
                ]
            )
                     
            if image_paths:
                import uuid
                import requests
                import time
                
                def upload_to_gemini(image_path):
                    # Upload file lên Gemini
                    try:
                        file = client.files.upload(file=image_path)
                        self.show_log(message=f"Uploaded file as: {file.uri}", level="debug")
                        return file
                    finally:
                        # Đợi một chút để đảm bảo file đã được upload hoàn tất
                        time.sleep(1)
                
                # Upload tất cả các file
                files = []
                for path in image_paths:
                    files.append(upload_to_gemini(path))
                
                # Bắt đầu chat session với các file đã upload
                chat_session = client.chats.create(
                    model=model,
                    config=generation_config
                )

                for file in files:
                    chat_session.send_message(file)

                # Gửi tin nhắn của người dùng
                response = chat_session.send_message(user_prompt)
                full_response_text = response.text
            else:
                # Trường hợp không có ảnh
                response = client.models.generate_content(model=model, contents=[user_prompt], config=generation_config)
                full_response_text = response.text
                
            if json_output:
                try:
                    result = json.loads(full_response_text)
                except json.JSONDecodeError as jde:
                    raise Exception(f"JSON decode error: {jde}")
            else:
                result = full_response_text
            
            if not result:
                raise Exception("Empty response")

            return True, result, None

        except Exception as ex:

            if retry <= 0:
                self.show_log(message=f"Fail to call __call_gemini_image with ex: {ex}, retry: {retry}", level="error")
                return False, None, str(ex)
            
            self.show_log(message=f"__call_gemini_image -> Retry {retry}", level="error")

            retry -= 1

            return self.__call_gemini_image(
                system_prompt=system_prompt, 
                user_prompt=user_prompt, 
                model=model, 
                retry=retry,
                json_output=json_output, 
                temperature=temperature, 
                limit_length=limit_length,
                image_paths=image_paths
            )
