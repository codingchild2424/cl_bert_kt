from models.cl_monacobert import CL_MonaCoBERT

# get models
def get_models(num_q, num_r, num_pid, num_diff, device, config):
    # choose the models
    if config.model_name == "cl_monacobert":
        model = CL_MonaCoBERT(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            num_diff=num_diff,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            config=config,
            dropout_p=config.dropout_p
        ).to(device)
    else:
        print("Wrong model_name was used...")

    return model