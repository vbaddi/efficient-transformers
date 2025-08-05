import torch
from transformers import AutoConfig, GptOssForCausalLM, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

torch.manual_seed(42)
model_id = "/home/vbaddi/transformers/src/transformers/models/gpt_oss/new_weights"
config = AutoConfig.from_pretrained(model_id)
model = GptOssForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, use_cache=True, attn_implementation="eager"
)
model.eval()

tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_id)
config = model.config
batch_size = len(Constants.INPUT_STR)

api_runner = ApiRunner(batch_size, tokenizer, config, Constants.INPUT_STR, Constants.PROMPT_LEN, Constants.CTX_LEN)

# pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model)
qeff_model = QEFFAutoModelForCausalLM(model, continuous_batching=False)
# pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
onnx_model_path = qeff_model.export()
# ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=False)

qpc_path = qeff_model.compile(
    prefill_seq_len=32,
    ctx_len=128,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=4,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
)
print(f"qpc path is {qpc_path}")
streamer = TextStreamer(tokenizer)
exec_info = qeff_model.generate(
    tokenizer,
    streamer=streamer,
    prompts="Who is your creator? and What all you are allowed to do?",
    device_ids=[0, 1, 2, 3],
)
