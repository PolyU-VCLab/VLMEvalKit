import base64
import json
import math
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from vlmeval.smp import encode_image_to_base64, get_logger
from .base import BaseAPI
from .openai_sdk import _apply_response_charset

logger = get_logger(__name__)

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def _parse_stream_chunk(chunk: dict):
    choices = chunk.get('choices', [])
    if not choices:
        return ''
    delta = choices[0].get('delta', {})
    if not isinstance(delta, dict):
        return ''
    return delta.get('content') or ''


def _collect_streaming_content(response, verbose=False):
    answer_parts = []
    chunk_count = 0
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        line = line.strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            if verbose:
                logger.warning(f'Invalid streaming chunk: {data[:120]}')
            continue
        part = _parse_stream_chunk(chunk)
        if part:
            answer_parts.append(part)
            chunk_count += 1
    answer = ''.join(answer_parts).strip()
    if verbose:
        logger.info(
            f'Streaming finished: chunks={chunk_count}, chars={len(answer)}'
        )
    return answer


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 300,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = -1,
                 total_img_size: int = -1,
                 max_file_size: int = 1e9,
                 img_detail: str = 'low',
                 image_format: str = 'JPEG',
                 use_azure: bool = False,
                 stream: bool = False,
                 response_charset: str = 'utf-8',
                 audio_format: str = None,
                 return_response_json: bool = False,
                 preserve_generation_config: bool = False,
                 token_limit_field: str = None,
                 enable_thinking: bool = None,
                 thinking_parameter: str = "chat_template",
                 **kwargs):
        """
        Args:
            response_charset: Charset used to decode HTTP responses that do
                not declare a charset in the Content-Type header. Defaults to
                ``utf-8``. Set to ``None`` to keep requests' inferred encoding.
        """

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        if thinking_parameter not in ("chat_template", "enable_thinking"):
            raise ValueError("thinking_parameter must be chat_template or enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean or None")
        self.enable_thinking = enable_thinking
        self.thinking_parameter = thinking_parameter
        self.use_azure = use_azure

        if 'step' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'internvl2-pro' in model:
            env_key = os.environ.get('InternVL2_PRO_KEY', '')
            if key is None:
                key = env_key
        elif 'abab' in model:
            env_key = os.environ.get('MiniMax_API_KEY', '')
            if key is None:
                key = env_key
        elif 'moonshot' in model:
            env_key = os.environ.get('MOONSHOT_API_KEY', '')
            if key is None:
                key = env_key
        elif 'grok' in model:
            env_key = os.environ.get('XAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'gemini' in model and 'preview' in model:
            # Will only handle preview models
            env_key = os.environ.get('GOOGLE_API_KEY', '')
            if key is None:
                key = env_key
            # api_base = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif 'ernie' in model:
            env_key = os.environ.get('BAIDU_API_KEY', '')
            if key is None:
                key = env_key
            api_base = 'https://qianfan.baidubce.com/v2/chat/completions'
            self.baidu_appid = os.environ.get('BAIDU_APP_ID', None)
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert total_img_size > 0 or total_img_size == -1
        self.total_img_size = total_img_size
        self.max_file_size = max_file_size
        assert img_detail in ['high', 'low', None]
        self.img_detail = img_detail
        if image_format not in ('JPEG', 'PNG'):
            raise ValueError('image_format must be JPEG or PNG')
        self.image_format = image_format
        self.timeout = timeout
        self.stream = stream
        self.response_charset = response_charset
        if audio_format not in (None, 'input_audio'):
            raise ValueError('audio_format must be None or input_audio')
        self.audio_format = audio_format
        self.return_response_json = return_response_json
        self.preserve_generation_config = preserve_generation_config
        if token_limit_field not in (None, 'max_tokens', 'max_completion_tokens'):
            raise ValueError('Unsupported token_limit_field')
        if token_limit_field is not None and not preserve_generation_config:
            raise ValueError('token_limit_field requires preserve_generation_config=True')
        self.token_limit_field = token_limit_field
        self.allowed_types = list(type(self).allowed_types)
        if audio_format == 'input_audio':
            self.allowed_types.append('audio')
        if stream and return_response_json:
            raise ValueError('return_response_json requires non-streaming responses')
        self.is_max_completion_tokens = ('o1' in model) or ('o3' in model) or ('o4' in model) or ('gpt-5' in model)
        self.is_o_model = ('o1' in model) or ('o3' in model) or ('o4' in model)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            else:
                logger.error('Unknown API Base. ')
                raise NotImplementedError
            if os.environ.get('BOYUE', None):
                self.api_base = os.environ.get('BOYUE_API_BASE')
                self.key = os.environ.get('BOYUE_API_KEY')

        logger.info(f'Using API Base: {self.api_base}; API Key: [REDACTED]')

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        image_num = len([x['type'] == 'image' for x in inputs])
        if has_images or any(x['type'] == 'audio' for x in inputs):
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text' and msg['value']:  # Skip empty string
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    target_size = math.inf
                    if self.img_size > 0:
                        target_size = self.img_size
                    if self.total_img_size > 0:
                        target_size = min(target_size, max(1, int(self.total_img_size / (image_num**0.5))))
                    target_size = -1 if math.isinf(target_size) else target_size
                    b64 = encode_image_to_base64(img,
                                                 target_size=target_size,
                                                 max_file_size=self.max_file_size,
                                                 fmt=self.image_format)
                    img_struct = dict(url=f'data:image/{self.image_format.lower()};base64,{b64}')
                    if self.img_detail is not None:
                        img_struct['detail'] = self.img_detail
                    content_list.append(dict(type='image_url', image_url=img_struct))
                elif msg['type'] == 'audio':
                    if self.audio_format != 'input_audio':
                        raise ValueError('Enable input_audio before sending audio')
                    path = Path(msg['value'])
                    extension = path.suffix.lower().lstrip('.')
                    if extension not in ('mp3', 'wav'):
                        raise ValueError('input_audio requires a local MP3 or WAV file')
                    content_list.append({
                        'type': 'input_audio',
                        'input_audio': {
                            'data': base64.b64encode(path.read_bytes()).decode('ascii'),
                            'format': extension,
                        },
                    })
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs, system_prompt=None):
        input_msgs = []
        if system_prompt is not None:
            input_msgs.append(dict(role='system', content=system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, system_prompt=None, **kwargs) -> str:
        # OmniCompass uses strict Chat Completions endpoints; preserve other tasks' payloads.
        if (kwargs.get('dataset') or '').startswith('OmniCompass'):
            kwargs.pop('dataset')
        input_msgs = self.prepare_inputs(inputs, system_prompt)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        elif 'internvl2-pro' in self.model:
            headers = {'Content-Type': 'application/json', 'Authorization': self.key}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        if hasattr(self, 'baidu_appid'):
            headers['appid'] = self.baidu_appid

        payload = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            **kwargs)
        stream = payload.pop('stream', self.stream)
        if self.enable_thinking is not None:
            if self.thinking_parameter == "enable_thinking":
                payload["enable_thinking"] = self.enable_thinking
            else:
                template_kwargs = dict(payload.get("chat_template_kwargs", {}))
                template_kwargs["enable_thinking"] = self.enable_thinking
                payload["chat_template_kwargs"] = template_kwargs

        if self.token_limit_field is not None:
            payload[self.token_limit_field] = max_tokens
        elif self.is_max_completion_tokens:
            payload['max_completion_tokens'] = max_tokens
            payload.pop('temperature')
        else:
            payload['max_tokens'] = max_tokens

        if 'gemini' in self.model and not self.preserve_generation_config:
            payload.pop('max_tokens')
            payload.pop('n')
            payload['reasoning_effort'] = 'high'

        if 'openai.com' not in self.api_base and payload.get('n') == 1:
            payload.pop('n', None)
        if stream:
            payload['stream'] = True

        proxies = {}
        if os.getenv('http_proxy'):
            proxies['http'] = os.getenv('http_proxy')
        if os.getenv('https_proxy'):
            proxies['https'] = os.getenv('https_proxy')
        proxies = proxies or None

        response = requests.post(
            self.api_base,
            headers=headers,
            data=json.dumps(payload),
            proxies=proxies,
            timeout=self.timeout * 1.1,
            stream=stream,
        )
        _apply_response_charset(response, self.response_charset)
        if self.return_response_json:
            body = response.json()
            encoded = json.dumps({'api_response': {'status_code': response.status_code, 'body': body}})
            code = 0 if 200 <= response.status_code < 300 else response.status_code
            # Return permanent HTTP failures once; the envelope retains their failure status.
            if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                code = 0
            return code, encoded, response
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            if stream and ret_code == 0:
                answer = _collect_streaming_content(response, verbose=self.verbose)
            else:
                resp_struct = json.loads(response.text)
                answer = resp_struct['choices'][0]['message']['content'].strip()
        except Exception as err:
            logger.error(f'{type(err)}: {err}')
            if self.verbose:
                logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception as err:
            if 'gpt' in self.model.lower():
                if self.verbose:
                    logger.warning(f'{type(err)}: {err}')
                enc = tiktoken.encoding_for_model('gpt-4')
            else:
                return 0
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += self.get_image_token_len(item['value'], detail=self.img_detail)
        return tot


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)
