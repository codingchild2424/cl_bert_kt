from trainers.cl_monacobert_trainer import CL_MonaCoBERT_Trainer
def get_trainers(model, optimizer, device, num_q, crit, config, q_diff_dicts=None, pid_diff_dicts=None):

    # choose trainer
    if config.model_name == "cl_monacobert":
        trainer = CL_MonaCoBERT_Trainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            config=config,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    else:
        print("wrong trainer was choosed..")

    return trainer
