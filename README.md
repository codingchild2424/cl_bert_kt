# DCL4KT+LLM: Difficulty-focused Contrasitve Learning for Knowledge Tracing with Large Language Model

This repository is for the research named Difficulty-focused Contrasitve Learning for Knowledge Tracing with LLM.  


# Performance (changed)

- Batch size: Batch size was 512. You can use grad accumulation option, if you don't have enough GPU resources.
- Early stop: Early stop was 10.
- Training, validation, test ratio: Training ratio was 80%, test ratio was 20%, valid ratio was 10% of training ratio.
- Learning rate and optimizer: The learning rate was 0.001. Adam was used.


|Dataset | Metrics | DKT | DKVMN | AKT | CL4KT | MCB | DCL4KT | DCL4KT-A
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|assist09 | AUC | 0.7285 | 0.7271 | 0.7449 | 0.7600 | 0.8059 | 0.8111 | 0.8153
| | RMSE | 0.4328 | 0.4348 | 0.4413 | 0.4337 | 0.4063  | 0.4068 | 0.4034
|algebra05 | AUC | 0.8088 | 0.8146 | 0.7673 | 0.7871 | 0.8201 | 0.8288 | 0.8295
| | RMSE | 0.3703 | 0.3687 | 0.3918 | 0.3824 | 0.3584  | 0.3657 | 0.3644
|algebra06 | AUC | 0.7939 | 0.7961 | 0.7505 | 0.7789 |  0.8064 | 0.8258 | 0.8278
| | RMSE | 0.3666 | 0.3661 | 0.3986 | 0.3863 | 0.3672  | 0.3657 | 0.3504
|EdNet | AUC | 0.6609 | 0.6602 | 0.6687 | 0.6651 | 0.7336 | 0.7392 | 0.7403
| | RMSE | 0.4598 | 0.4597 | 0.4783 | 0.4750 | 0.4516  | 0.4505 | 0.4500
|Homerun20 | AUC | 0.7619 | 0.7543 | 0.5903 | 0.6014 | 0.7659 | 0.7766 | 0.7808
| | RMSE | 0.4092 | 0.4212 | 0.4745 | 0.4631 | 0.4880  | 0.4042 | 0.4014


# Setups

1. We used python-3.9 in docker environments.
2. If you don't use docker environments, then you can use **requirements.txt**.

   ```
   pip install -r requirements.txt
   ```

3. You can download the preprocessed dataset from our Google Drive.
   [https://drive.google.com/drive/folders/1B5vHabKUwzGEQAyDaj_0cBT6UqTMA13L?usp=sharing](https://drive.google.com/drive/folders/1B5vHabKUwzGEQAyDaj_0cBT6UqTMA13L?usp=sharing)

4. However, the dataset with text of question (Homerun20) was not opened to public, because the source of dataset is from EdTech company. So if you want to use sim_diff_llm_loader then you have to use your personal dataset with text.

5. If you want to preprocess yourself, you can use **preprocess_data.py** in cl4kt [https://github.com/UpstageAI/cl4kt].

   ```
   python preprocess_data.py --data_name assist09 --min_user_inter_num 5
   ```

# How to run this code?

If you want to run the DCL4KT+LLM, you can use train.sh. All of the options are contained in the define_argparser.py.
