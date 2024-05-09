# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import transformers
from cloud_infer_cb import QAICInferenceSession

io_files = []
np.random.seed(42)
vocab_size = 50258


def write_io_files(
    inputs: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    write_io_dir: str,
    write_io_subdir: str,
    write_io_name: str,
    include_dims: bool = False,
    reset: bool = False,
):
    global io_files

    if reset:
        io_files = []

    io = []
    os.makedirs(f"{write_io_dir}/{write_io_subdir}", exist_ok=True)

    for iname, iarray in inputs.items():
        iarray.tofile(f"{write_io_dir}/{write_io_subdir}/{iname}.raw")
        ispec = {
            "path": f"{write_io_subdir}/{iname}.raw",
            "io-direction": "in",
            "elem-size": iarray.itemsize,
            "map-to": iname,
        }
        if include_dims:
            ispec["dims"] = iarray.shape
        io.append(ispec)

    for oname, oarray in outputs.items():
        oarray.tofile(f"{write_io_dir}/{write_io_subdir}/{oname}.raw")
        ospec = {
            "path": f"{write_io_subdir}/{oname}.raw",
            "io-direction": "out",
            "elem-size": oarray.itemsize,
            "map-to": oname,
        }
        if include_dims or oname.endswith("_RetainedState"):
            ospec["dims"] = oarray.shape
        io.append(ospec)

    io_files.append(io)
    with open(f"{write_io_dir}/{write_io_name}.json", "w") as fp:
        json.dump({"IO-files": io_files}, fp, indent=True)


def set_logits_bsize(session, batch_size):
    """
    sets the size of the output logits expected
    """
    logits_out_placeholder = np.zeros((batch_size, 1, vocab_size), dtype=np.float32)
    session.set_buffers({"logits": logits_out_placeholder})


def populate_inputs(source, dest, index):
    """
    populates the dest input dict at the specified index with the source input dict's items
    """
    for k, v in dest.items():
        # print("populating input at key ", k)
        if k == "batch_index":
            continue
        dest[k][index] = source[k]


def run_prefill(index, prefill_queue, session, tokenizer, prompt_len, ctx_len, decode_batch_size, slot_idx):
    """
    runs prefill on the prompt at the specified index in the prefill queue

    returns the generated token id to start decoding from,
    the length of the prompt, the position id and cache index to
    start decoding this prompt from

    accepts decode batch size to populate the attention mask accordingly
    accepts the slot_idx to indicate which slot we are trying to replace
    with the current prompt/request
    """
    assert slot_idx < decode_batch_size
    decode_start_input = dict()
    # retrieve the prompt from the prefill queue
    prompt = prefill_queue[index]
    input_len = tokenizer(prompt, return_tensors="np", padding=True).input_ids.shape[1]
    num_chunks = -(input_len // -prompt_len)  # ceil divide without float
    input_len = num_chunks * prompt_len  # Convert input_len to a multiple of prompt_len
    assert input_len <= ctx_len, "input_len should be less than ctx_len"
    # pad the prompt tokens to match the input_len
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=input_len)
    # TODO need to store the attention mask and position ids for each batch element so that we can access them
    # at decode time
    inputs["attention_mask"] = np.concatenate(
        [inputs["attention_mask"].astype(bool) for j in range(decode_batch_size)], 0
    )
    inputs["position_ids"] = (np.cumsum(inputs["attention_mask"][0:1], 1) - 1) * inputs["attention_mask"][0:1]
    inputs["attention_mask"] = np.concatenate(
        [
            inputs["attention_mask"].astype(bool),
            np.zeros((decode_batch_size, ctx_len - input_len), dtype=bool),
        ],
        1,
    )
    cache_index = np.array([[0]], np.int64)
    batch_index = np.array([[slot_idx]], np.int64)
    inputs["cache_index"] = cache_index
    inputs["batch_index"] = batch_index
    # print(f"batch_index: {batch_index} prompt: {prompt}")

    # FIXME assumes prefill batch size is always 1
    set_logits_bsize(session, 1)
    # print(f"** PREFILL INPUTS {index} PROMPT {prefill_queue[index]} **")
    # print(inputs)
    # breakpoint()
    # Run prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, cache_index[0, 0] : cache_index[0, 0] + prompt_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, cache_index[0, 0] : cache_index[0, 0] + prompt_len]
        chunk_inputs["attention_mask"] = inputs["attention_mask"].copy()
        chunk_inputs["attention_mask"][:, cache_index[0, 0] + prompt_len :] = False
        outputs = session.run(chunk_inputs)
        cache_index += prompt_len

    # Get first token
    logits = outputs["logits"]
    if len(logits.shape) == 2:
        logits = np.expand_dims(logits, 1)
    decode_start_input_id = logits.argmax(2)
    decode_start_pos_id = inputs["attention_mask"][0:1].sum(1, keepdims=True)

    # update the attention mask so that it can be used as input to start decode
    inputs["attention_mask"][:, cache_index] = True
    # populate the decode start info to return, to be able resume with the AR decoding this sequence
    # decoding will start using these inputs for the current request
    decode_start_input["input_ids"] = decode_start_input_id
    decode_start_input["position_ids"] = decode_start_pos_id
    decode_start_input["attention_mask"] = inputs.pop("attention_mask")[0:1]
    decode_start_input["cache_index"] = cache_index
    decode_start_input["batch_index"] = batch_index
    decode_start_input["input_len"] = input_len
    # print(f"prefill output id:{decode_start_input_id[0]} token: {tokenizer.convert_ids_to_tokens(decode_start_input_id[0])}")
    # print("returning decode start info as:", decode_start_input)

    return decode_start_input


def main(
    model_name: str,
    qpc: str,
    prompt: List[str],
    input_len: Optional[int] = None,
    generation_len: Optional[int] = None,
    stream: bool = True,
    device_id: List[int] = [0],
    enable_debug_logs: bool = False,
    write_io_dir: Optional[str] = None,
    automation: bool = False,
) -> Dict[str, float]:
    # Load QPC and tokenizer
    session = QAICInferenceSession(qpc, device_id, enable_debug_logs=enable_debug_logs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Skip inputs/outputs
    session.skip_buffers([x for x in session.input_names if x.startswith("past_")])
    session.skip_buffers([x for x in session.output_names if x.endswith("_RetainedState")])
    # session.skip_buffers([x for x in session.output_names if x.startswith("past_")])

    # Read prompt and ctx len from session
    # TODO need to read from the session bindings corresponding to prefill and decode according to stage
    batch_size, ctx_len = session.bindings[session.binding_index_map["attention_mask"]].dims
    print(f"session batch_size: {batch_size} , ctx_len: {ctx_len}")
    json_prompts = dict()
    with open("./sub_dataset.json", "r") as f:
        json_prompts = json.load(f)
    for i in range(len(json_prompts)):
        print(len(json_prompts[i]["input"]))
        json_prompts[i]["input"] = json_prompts[i]["input"][:100]
        print(len(json_prompts[i]["input"]))
    prompt_insert_str = [
        json_prompts[i]["instruction"] + "\n" + json_prompts[i]["input"] for i in range(len(json_prompts))
    ]
    prompt = [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt_insert_str[i]}\n\n### Response:"
        for i in range(len(prompt_insert_str))
    ]
    prompt_queue_size = 20
    # prefill_batch_size = 1
    decode_batch_size = 4
    prefill_queue = []
    prompt_len = max(
        [x[session.binding_index_map["input_ids"]][1][1] for x in session.allowed_shapes]
        + [session.bindings[session.binding_index_map["input_ids"]].dims[1]]
    )

    if generation_len is None:
        generation_len = ctx_len
    if len(prompt) < prompt_queue_size:
        print(f"Repeating prompt {prompt_queue_size} times")
        prompt = prompt * -(prompt_queue_size // -len(prompt))  # Repeat prompt to required size
    prompt = prompt[:prompt_queue_size]  # Truncate prompts to required size

    # FIXME for now, generating random numbers to decide how many tokens to generate for each prompt request
    req_max_length = list(np.random.randint(low=20, high=100, size=prompt_queue_size))
    # add all prompts to the prefill queue
    prefill_queue = list(prompt)
    print("Request queue initially: ", prefill_queue)
    print("Max tokens/request:", req_max_length)

    cache_index = np.zeros((batch_size, 1), np.int64)
    batch_index = np.reshape(np.array(np.arange(batch_size), np.int64), (batch_size, 1))

    # initialize empty list to store generated tokens for each prompt
    generated_ids = [[] for i in range(prompt_queue_size)]
    # store the length of each prompt requested
    input_lengths = [0 for i in range(prompt_queue_size)]
    # store the number of prompts processed out of the prompt_queue
    num_prompts_processed = 0
    # initialize dynamic container which will hold all the global request ids (position in prompt request queue)
    # of the prompts currently being processed
    current_batch_req_ids = []

    # Prepare inputs for first iteration
    start = perf_counter()
    decode_inputs = dict()
    decode_inputs["position_ids"] = np.zeros((decode_batch_size, 1), np.int64)
    decode_inputs["input_ids"] = np.full((decode_batch_size, 1), tokenizer.pad_token_id)
    decode_inputs["cache_index"] = cache_index
    decode_inputs["batch_index"] = batch_index
    decode_inputs["attention_mask"] = np.zeros((decode_batch_size, ctx_len), dtype=bool)
    # iteratively run prefill with bs=1 for decode_batch_size number of requests to fill the decode bs=decode_batch_size -long container
    ##TODO handle when prefill queue size is less than the decode batch size
    for bi in range(decode_batch_size):
        # FIXME assumes that prefill queue will always be popped from the front
        decode_start_input = run_prefill(
            index=0,
            prefill_queue=prefill_queue,
            session=session,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            decode_batch_size=decode_batch_size,
            slot_idx=bi,
        )
        # TODO move below code to a function, that takes the prefill-dict and populates it into a particular index
        # this way, we will directly get the updated full batch input dict to run decode
        # print("Populating inputs to kick off decode for slot",bi)
        populate_inputs(source=decode_start_input, dest=decode_inputs, index=bi)
        # FIXME assumes that prefill queue will always be popped from the front
        current_batch_req_ids.append(bi)
        input_lengths[current_batch_req_ids[bi]] = decode_start_input["input_len"]
        num_prompts_processed += 1
        # update generated id list for this request, right after running prefill
        generated_ids[current_batch_req_ids[bi]].append(decode_start_input["input_ids"][0, 0])
        # pop the front of the prefill queue
        # assumes that prefill queue will always be popped from the front
        prefill_queue = prefill_queue[1:]
    # FIXME currently we CANNOT Skip attention_mask from next iteration to use retained attention_mask
    # session.skip_buffers(["attention_mask"])
    # all_inputs.pop("attention_mask")
    print("** INITIAL PREFILL DONE **")
    # update logits placeholder for multi-batch decode
    set_logits_bsize(session, decode_batch_size)

    loop_start = perf_counter()
    next_token_id = decode_inputs["input_ids"]
    # finished_sequences = next_token_id == tokenizer.eos_token_id
    niter = 0
    current_decode_ongoing = True
    while (len(prefill_queue) > 0) or current_decode_ongoing:
        # print(f"Running Decode Iter {niter} inputs:")
        # print(decode_inputs)
        # decode_inputs2=dict(decode_inputs)
        # decode_inputs2.pop("attention_mask")
        outputs = session.run(decode_inputs)
        # print(outputs)
        # get the next token id for each batch element
        logits = outputs["logits"]
        if len(logits.shape) == 2:
            logits = np.expand_dims(logits, 1)
        next_token_id = logits.argmax(2)
        # generated_next_tokens = [tokenizer.batch_decode(next_token_id[i], skip_special_tokens=True) for i in range(decode_batch_size)]
        # print("Tokens Generated: ", next_token_id,generated_next_tokens)
        # breakpoint()

        # update the container of generated ids with the next_token_id
        current_decode_ongoing = False
        for idx, pid in enumerate(current_batch_req_ids):
            if (len(generated_ids[pid]) < req_max_length[pid]) and not (
                generated_ids[pid][-1] == tokenizer.eos_token_id
            ):
                generated_ids[pid].append(next_token_id[idx, 0])
                current_decode_ongoing = True

        for idx in range(decode_batch_size):
            # print("Checking generated token at decode batch index:",idx)
            # check if any of the sequences have reached eos/max length

            if (
                next_token_id[idx, 0] == tokenizer.eos_token_id
                or len(generated_ids[current_batch_req_ids[idx]]) >= req_max_length[current_batch_req_ids[idx]]
            ):
                if len(prefill_queue) == 0:
                    # print("prefill queue is empty. no slot replacement. accepting over-compute")
                    continue
                print(f"** running prefill on new request at prompt queue index = {num_prompts_processed} **")
                # FIXME assumes that prefill queue will always be popped from the front
                # stop this sequence and replace it with a new one from the prefill queue
                # run prefill on the new request
                # FIXME assumes that prefill queue will always be popped from the front
                new_decode_start_input = run_prefill(
                    index=0,
                    prefill_queue=prefill_queue,
                    session=session,
                    tokenizer=tokenizer,
                    prompt_len=prompt_len,
                    ctx_len=ctx_len,
                    decode_batch_size=decode_batch_size,
                    slot_idx=idx,
                )
                # populate the inputs at the idxth position
                populate_inputs(source=new_decode_start_input, dest=decode_inputs, index=idx)
                # print(decode_inputs)
                # breakpoint()
                # update the current_batch_req_ids[idx] to this request's id
                # FIXME assumes that prefill queue will always be popped from the front
                current_batch_req_ids[idx] = num_prompts_processed
                num_prompts_processed += 1
                # update the input_length of the current request
                input_lengths[current_batch_req_ids[idx]] = new_decode_start_input["input_len"]
                # update generated id list for this new request with the prefill generated token
                generated_ids[current_batch_req_ids[idx]].append(new_decode_start_input["input_ids"][0, 0])

                # depopulate the prefill queue
                # FIXME assumes that prefill queue will always be popped from the front
                prefill_queue = prefill_queue[1:]
                # reset the logits placeholder back to decode mode
                set_logits_bsize(session, decode_batch_size)
            else:
                # increment the cache index and position ids for this idx to continue decoding
                # update the attention mask
                req_id = current_batch_req_ids[idx]
                # print("Updating inputs from last decode for ", req_id)
                # print("input lengths: ", input_lengths[req_id])
                # print("generated id length: ", len(generated_ids[req_id]))
                decode_inputs["attention_mask"][idx][
                    input_lengths[req_id] : input_lengths[req_id] + len(generated_ids[req_id])
                ] = 1
                decode_inputs["input_ids"][idx] = next_token_id[idx]
                decode_inputs["position_ids"][idx] = decode_inputs["position_ids"][idx] + 1
                decode_inputs["cache_index"][idx] = decode_inputs["cache_index"][idx] + 1
        # print(f"Decoding batch ids: ", current_batch_req_ids)
        # print(f"Decode Iteration: {niter} next token per batch element: {next_token_id}")
        niter += 1

    end = perf_counter()
    print("Generated ID lengths: ", [len(o) for o in generated_ids])
    print("Max Tokens per Request assumed: ", req_max_length)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # generated_tokens = [
    #     tokenizer.batch_decode(generated_ids[i], skip_special_tokens=True) for i in range(len(generated_ids))
    # ]
    for i in range(1 if stream else 0, num_prompts_processed):
        print()
        print(i, prompt[i], generated_texts[i])

    total_num_decoded_tokens = sum([(len(generated_ids[i]) - 1) for i in range(prompt_queue_size)])
    prefill_perf = 1 / (loop_start - start)
    # decode_perf = (cache_index.item() - input_len - 1) / (end - loop_start)
    decode_perf = (total_num_decoded_tokens) / (end - loop_start)
    # total_perf = (cache_index.item() - input_len) / (end - start)
    total_perf = (total_num_decoded_tokens + prompt_queue_size) / (end - start)

    print()

    if automation:
        print()
        print("input=", prompt)
        print("output=", generated_texts)
        print("Prefill token/sec is=", round(prefill_perf * batch_size, 2))
        print("Decode token/sec is=", round(decode_perf * batch_size, 2))
        print("Total token/sec is=", round(total_perf * batch_size, 2))
        return

    print("TTFT:", round(loop_start - start, 2), "s")
    print("E2ET:", round(end - start, 2), "s")
    print("Prefill:", round(prefill_perf, 2), "tok/s")
    print("Decode:", round(decode_perf, 2), "tok/s")
    print("E2E:", round(total_perf, 2), "tok/s")
    # if batch_size > 1:
    #     print("Prefill (batch):", round(prefill_perf * batch_size, 2), "tok/s")
    #     print("Decode (batch):", round(decode_perf * batch_size, 2), "tok/s")
    #     print("E2E (batch):", round(total_perf * batch_size, 2), "tok/s")


if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument("--model-name", required=True, help="Model name to run")
    argp.add_argument("--qpc", required=True, help="Compiled binary QPC")
    argp.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        default="My name is",
        help="Input prompt(s) to generate for (pipe-separated)",
    )
    argp.add_argument("--input-len", type=int, help="Input length")
    argp.add_argument("--generation-len", type=int, help="generation length")
    argp.add_argument("--no-stream", action="store_false", dest="stream", help="Don't stream output text")
    argp.add_argument(
        "--device_id",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        help="QAIC device ids (comma-separated)",
    )
    argp.add_argument("--enable-debug-logs", action="store_true", help="Enable debug logs in LRT")
    argp.add_argument("--write-io-dir", help="Directory to write inputs/outputs into")
    argp.add_argument("--automation", action="store_true", help="Print outputs in required format for automation")

    args = argp.parse_args()
    main(**vars(args))
