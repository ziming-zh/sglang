# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The entry point of inference server.
SRT = SGLang Runtime.
"""

import asyncio
import atexit
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from http import HTTPStatus
from typing import AsyncIterator, Dict, List, Optional, Union

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import aiohttp
import orjson
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from uvicorn.config import LOGGING_CONFIG

from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    OpenSessionReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.metrics.func_timer import enable_func_timer, time_func_latency
from sglang.srt.openai_api.adapter import (
    load_chat_template_for_openai_api,
    v1_batches,
    v1_cancel_batch,
    v1_chat_completions,
    v1_completions,
    v1_delete_file,
    v1_embeddings,
    v1_files_create,
    v1_retrieve_batch,
    v1_retrieve_file,
    v1_retrieve_file_content,
)
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    CompleteTokenQueryService,
    add_api_key_middleware,
    add_prometheus_middleware,
    assert_pkg_version,
    configure_logger,
    delete_directory,
    is_port_available,
    kill_process_tree,
    maybe_set_triton_cache_manager,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.utils import get_exception_traceback
from sglang.version import __version__

logger = logging.getLogger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Fast API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer_manager: TokenizerManager = None
scheduler_info: Dict = None
##### Native API endpoints #####


@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """Check the health of the inference server by generating one token."""

    if tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            input_ids=[0], sampling_params={"max_new_tokens": 1, "temperature": 0.7}
        )
    else:
        gri = EmbeddingReqInput(
            input_ids=[0], sampling_params={"max_new_tokens": 1, "temperature": 0.7}
        )

    try:
        async for _ in tokenizer_manager.generate_request(gri, request):
            break
        return Response(status_code=200)
    except Exception as e:
        logger.exception(e)
        return Response(status_code=503)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {
        "model_path": tokenizer_manager.model_path,
        "tokenizer_path": tokenizer_manager.server_args.tokenizer_path,
        "is_generation": tokenizer_manager.is_generation,
    }
    return result


@app.get("/get_server_info")
async def get_server_info():
    return {
        **dataclasses.asdict(tokenizer_manager.server_args),  # server args
        **scheduler_info,
        "version": __version__,
    }


@app.post("/flush_cache")
async def flush_cache():
    """Flush the radix cache."""
    tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


@app.get("/start_profile")
@app.post("/start_profile")
async def start_profile_async():
    """Start profiling."""
    tokenizer_manager.start_profile()
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.get("/stop_profile")
@app.post("/stop_profile")
async def stop_profile_async():
    """Stop profiling."""
    tokenizer_manager.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


@app.post("/update_weights_from_disk")
@time_func_latency
async def update_weights_from_disk(obj: UpdateWeightFromDiskReqInput, request: Request):
    """Update the weights from disk inplace without re-launching the server."""
    success, message = await tokenizer_manager.update_weights_from_disk(obj, request)
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.post("/init_weights_update_group")
async def init_weights_update_group(
    obj: InitWeightsUpdateGroupReqInput, request: Request
):
    """Initialize the parameter update group."""
    success, message = await tokenizer_manager.init_weights_update_group(obj, request)
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(
    obj: UpdateWeightsFromDistributedReqInput, request: Request
):
    """Update model parameter from distributed online."""
    success, message = await tokenizer_manager.update_weights_from_distributed(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.api_route("/get_weights_by_name", methods=["GET", "POST"])
async def get_weights_by_name(obj: GetWeightsByNameReqInput, request: Request):
    """Get model parameter by name."""
    try:
        ret = await tokenizer_manager.get_weights_by_name(obj, request)
        if ret is None:
            return ORJSONResponse(
                {"error": {"message": "Get parameter by name failed"}},
                status_code=HTTPStatus.BAD_REQUEST,
            )
        else:
            return ORJSONResponse(ret, status_code=200)
    except Exception as e:
        return ORJSONResponse(
            {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
        )


@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: OpenSessionReqInput, request: Request):
    """Open a session, and return its unique session id."""
    try:
        session_id = await tokenizer_manager.open_session(obj, request)
        return session_id
    except Exception as e:
        return ORJSONResponse(
            {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
        )


@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: CloseSessionReqInput, request: Request):
    """Close the session"""
    try:
        await tokenizer_manager.close_session(obj, request)
        return Response(status_code=200)
    except Exception as e:
        return ORJSONResponse(
            {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
        )


# fastapi implicitly converts json in the request to obj (dataclass)
@app.api_route("/generate", methods=["POST", "PUT"])
@time_func_latency
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in tokenizer_manager.generate_request(obj, request):
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                yield b"data: " + orjson.dumps(
                    out, option=orjson.OPT_NON_STR_KEYS
                ) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await tokenizer_manager.generate_request(obj, request).__anext__()
            return ret
        except ValueError as e:
            return ORJSONResponse(
                {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )


@app.api_route("/encode", methods=["POST", "PUT"])
@time_func_latency
async def encode_request(obj: EmbeddingReqInput, request: Request):
    """Handle an embedding request."""
    try:
        ret = await tokenizer_manager.generate_request(obj, request).__anext__()
        return ret
    except ValueError as e:
        return ORJSONResponse(
            {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
        )


@app.api_route("/classify", methods=["POST", "PUT"])
@time_func_latency
async def classify_request(obj: EmbeddingReqInput, request: Request):
    """Handle a reward model request. Now the arguments and return values are the same as embedding models."""
    try:
        ret = await tokenizer_manager.generate_request(obj, request).__anext__()
        return ret
    except ValueError as e:
        return ORJSONResponse(
            {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
        )


##### OpenAI-compatible API endpoints #####


@app.post("/v1/completions")
@time_func_latency
async def openai_v1_completions(raw_request: Request):
    return await v1_completions(tokenizer_manager, raw_request)


@app.post("/v1/chat/completions")
@time_func_latency
async def openai_v1_chat_completions(raw_request: Request):
    return await v1_chat_completions(tokenizer_manager, raw_request)


@app.post("/v1/embeddings", response_class=ORJSONResponse)
@time_func_latency
async def openai_v1_embeddings(raw_request: Request):
    response = await v1_embeddings(tokenizer_manager, raw_request)
    return response


@app.get("/v1/models", response_class=ORJSONResponse)
def available_models():
    """Show available models."""
    served_model_names = [tokenizer_manager.served_model_name]
    model_cards = []
    for served_model_name in served_model_names:
        model_cards.append(ModelCard(id=served_model_name, root=served_model_name))
    return ModelList(data=model_cards)


@app.post("/v1/files")
async def openai_v1_files(file: UploadFile = File(...), purpose: str = Form("batch")):
    return await v1_files_create(
        file, purpose, tokenizer_manager.server_args.file_storage_pth
    )


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/delete
    return await v1_delete_file(file_id)


@app.post("/v1/batches")
async def openai_v1_batches(raw_request: Request):
    return await v1_batches(tokenizer_manager, raw_request)


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batches(batch_id: str):
    # https://platform.openai.com/docs/api-reference/batch/cancel
    return await v1_cancel_batch(tokenizer_manager, batch_id)


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str):
    return await v1_retrieve_batch(batch_id)


@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve
    return await v1_retrieve_file(file_id)


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve-contents
    return await v1_retrieve_file_content(file_id)


def launch_engine(
    server_args: ServerArgs,
):
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """

    global tokenizer_manager
    global scheduler_info
    global complete_token_manager

    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    port_args = PortArgs.init_new(server_args)
    logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        scheduler_procs = []
        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        manager = mp.Manager()
        complete_token_manager = CompleteTokenQueryService(tp_rank_range.stop, manager)
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = server_args.base_gpu_id + tp_rank % tp_size_per_node
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, port_args, gpu_id, tp_rank, None, writer, complete_token_manager),
            )
            proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        if server_args.node_rank >= 1:
            # For other nodes, they do not need to run tokenizer or detokenizer,
            # so they can just wait here.
            while True:
                pass
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Launch tokenizer process
    tokenizer_manager = TokenizerManager(server_args, port_args)
    if server_args.chat_template:
        load_chat_template_for_openai_api(tokenizer_manager, server_args.chat_template)

    # Wait for model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        data = scheduler_pipe_readers[i].recv()

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have same max_total_num_tokens
    scheduler_info = scheduler_infos[0]


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
):
    """
    Launch SRT (SGLang Runtime) Server

    The SRT server consists of an HTTP server and the SRT engine.

    1. HTTP server: A FastAPI server that routes requests to the engine.
    2. SRT engine:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server and TokenizerManager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """
    launch_engine(server_args=server_args)

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Send a warmup request
    t = threading.Thread(
        target=_wait_and_warmup, args=(server_args, pipe_finish_writer)
    )
    t.start()

    try:
        # Update logging configs
        LOGGING_CONFIG["formatters"]["default"][
            "fmt"
        ] = "[%(asctime)s] %(levelprefix)s %(message)s"
        LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        LOGGING_CONFIG["formatters"]["access"][
            "fmt"
        ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
        LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"

        # Listen for HTTP requests
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        t.join()


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer",
            "0.1.6",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )

    # Register the signal handler.
    # The child processes will send SIGQUIT to this process when any error happens
    # This process then clean up the whole process tree
    def sigquit_handler(signum, frame):
        kill_process_tree(os.getpid())

    signal.signal(signal.SIGQUIT, sigquit_handler)

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _wait_and_warmup(server_args, pipe_finish_writer):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/get_model_info", timeout=5, headers=headers)
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return

    model_info = res.json()

    # Send a warmup request
    request_name = "/generate" if model_info["is_generation"] else "/encode"
    max_new_tokens = 8 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [10, 11, 12]
    else:
        json_data["text"] = "The capital city of France is"

    try:
        for _ in range(server_args.dp_size):
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200, f"{res}"
    except Exception:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return

    # Debug print
    # logger.info(f"{res.json()=}")

    logger.info("The server is fired up and ready to roll!")
    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")

    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model_path)


STREAM_END_SYMBOL = b"data: [DONE]"
STREAM_CHUNK_START_SYMBOL = b"data:"


class Engine:
    """
    SRT Engine without an HTTP server layer.

    This class provides a direct inference engine without the need for an HTTP server. It is designed for use cases where
    launching the HTTP server adds unnecessary complexity or overhead,
    """

    def __init__(self, log_level: str = "error", *args, **kwargs):
        """See the arguments in server_args.py::ServerArgs"""

        # before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        atexit.register(self.shutdown)

        server_args = ServerArgs(*args, log_level=log_level, **kwargs)
        launch_engine(server_args=server_args)

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        stream: bool = False,
    ):
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            lora_path=lora_path,
            stream=stream,
        )

        # get the current event loop
        loop = asyncio.get_event_loop()
        ret = loop.run_until_complete(generate_request(obj, None))

        if stream is True:

            def generator_wrapper():
                offset = 0
                loop = asyncio.get_event_loop()
                generator = ret.body_iterator
                while True:
                    chunk = loop.run_until_complete(generator.__anext__())

                    if chunk.startswith(STREAM_END_SYMBOL):
                        break
                    else:
                        data = json.loads(chunk[len(STREAM_CHUNK_START_SYMBOL) :])
                        data["text"] = data["text"][offset:]
                        offset += len(data["text"])
                        yield data

            # we cannot yield in the scope of generate() because python does not allow yield + return in the same function
            # however, it allows to wrap the generator as a subfunction and return
            return generator_wrapper()
        else:
            return ret

    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Dict] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        stream: bool = False,
    ):
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            lora_path=lora_path,
            stream=stream,
        )

        ret = await generate_request(obj, None)

        if stream is True:
            generator = ret.body_iterator

            async def generator_wrapper():

                offset = 0

                while True:
                    chunk = await generator.__anext__()

                    if chunk.startswith(STREAM_END_SYMBOL):
                        break
                    else:
                        data = json.loads(chunk[len(STREAM_CHUNK_START_SYMBOL) :])
                        data["text"] = data["text"][offset:]
                        offset += len(data["text"])
                        yield data

            return generator_wrapper()
        else:
            return ret

    def shutdown(self):
        kill_process_tree(os.getpid(), include_parent=False)

    def get_tokenizer(self):
        global tokenizer_manager

        if tokenizer_manager is None:
            raise ReferenceError("Tokenizer Manager is not initialized.")
        else:
            return tokenizer_manager.tokenizer

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
    ):
        obj = EmbeddingReqInput(text=prompt)

        # get the current event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(encode_request(obj, None))

    def start_profile(self):
        tokenizer_manager.start_profile()

    def stop_profile(self):
        tokenizer_manager.stop_profile()

    def get_server_info(self):
        return {
            **dataclasses.asdict(tokenizer_manager.server_args),  # server args
            **scheduler_info,
            "version": __version__,
        }

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ):
        """Initialize parameter update group."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

        async def _init_group():
            return await tokenizer_manager.init_weights_update_group(obj, None)

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_init_group())

    def update_weights_from_distributed(self, name, dtype, shape):
        """Update weights from distributed source."""
        obj = UpdateWeightsFromDistributedReqInput(
            name=name,
            dtype=dtype,
            shape=shape,
        )

        async def _update_weights():
            return await tokenizer_manager.update_weights_from_distributed(obj, None)

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_update_weights())

    def get_weights_by_name(self, name, truncate_size=100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)

        async def _get_weights():
            return await tokenizer_manager.get_weights_by_name(obj, None)

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_get_weights())


class Runtime:
    """
    A wrapper for the HTTP server.
    This is used for launching the server in a python program without
    using the commond line interface.

    It is mainly used for the frontend language.
    You should use the Engine class if you want to do normal offline processing.
    """

    def __init__(
        self,
        log_level: str = "error",
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        atexit.register(self.shutdown)

        # Pre-allocate ports
        for port in range(10000, 40000):
            if is_port_available(port):
                break
            port += 1
        self.server_args.port = port

        self.url = self.server_args.url()
        self.generate_url = self.url + "/generate"

        # NOTE: We store pid instead of proc to fix some issues during __delete__
        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)

        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, pipe_writer),
        )
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "ready":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )

        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        if self.pid is not None:
            kill_process_tree(self.pid)
            self.pid = None

    def cache_prefix(self, prefix: str):
        self.endpoint.cache_prefix(prefix)

    def get_tokenizer(self):
        return get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
        )

    async def async_generate(
        self,
        prompt: str,
        sampling_params: Optional[Dict] = None,
    ):
        if self.server_args.skip_tokenizer_init:
            json_data = {
                "input_ids": prompt,
                "sampling_params": sampling_params,
                "stream": True,
            }
        else:
            json_data = {
                "text": prompt,
                "sampling_params": sampling_params,
                "stream": True,
            }
        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        if "text" in data:
                            cur = data["text"][pos:]
                            if cur:
                                yield cur
                            pos += len(cur)
                        else:
                            yield data

    add_request = async_generate

    def generate(
        self,
        prompt: Union[str, List[str]],
        sampling_params: Optional[Dict] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
    ):
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "lora_path": lora_path,
        }
        assert not isinstance(lora_path, list) or len(lora_path) == len(prompt)
        response = requests.post(
            self.url + "/generate",
            json=json_data,
        )
        return json.dumps(response.json())

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
    ):
        json_data = {"text": prompt}
        response = requests.post(self.url + "/encode", json=json_data)
        return json.dumps(response.json())

    async def get_server_info(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/get_server_info") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    raise RuntimeError(
                        f"Failed to get server info. {error_data['error']['message']}"
                    )

    def __del__(self):
        self.shutdown()
