from modules import script_callbacks, extra_networks, prompt_parser, sd_models
from modules.sd_hijack import model_hijack
from functools import partial, reduce


def get_token_counter(text, steps):
    # copy from modules.ui.py
    try:
        text, _ = extra_networks.parse_prompt(text)

        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    try:
        from modules_forge import forge_version
        forge = True
    except:
        forge = False

    flat_prompts = reduce(lambda list1, list2: list1 + list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]

    if forge:
        # 🛡️ 防御性判断，防止模型未加载时报错
        sd_model = getattr(sd_models.model_data, "sd_model", None)
        if sd_model is None or not hasattr(sd_model, "cond_stage_model") or sd_model.cond_stage_model is None:
            print("⚠️ [Prompt All-in-One] 模型尚未加载完成，跳过 token 计数。")
            return {"token_count": 0, "max_length": 0}

        cond_stage_model = sd_model.cond_stage_model
        token_count, max_length = max(
            [model_hijack.get_prompt_lengths(prompt, cond_stage_model) for prompt in prompts],
            key=lambda args: args[0]
        )
    else:
        token_count, max_length = max(
            [model_hijack.get_prompt_lengths(prompt) for prompt in prompts],
            key=lambda args: args[0]
        )

    return {"token_count": token_count, "max_length": max_length}
