# DCL4KT+LLM: Difficulty-focused Contrasitve Learning for Knowledge Tracing with Large Language Model

This repository is for the research named Difficulty-focused Contrasitve Learning for Knowledge Tracing with LLM.  


# Performance (changed)

- Batch size: Batch size was 512. You can use grad accumulation option, if you don't have enough GPU resources.
- Early stop: Early stop was 10.
- Training, validation, test ratio: Training ratio was 80%, test ratio was 20%, valid ratio was 10% of training ratio.
- Learning rate and optimizer: The learning rate was 0.001. Adam was used.


|Dataset | Metrics | DKT | DKVMN | SAKT | AKT | CL4KT | **MCB** | DCL4KT+LLM
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----
|assist09 | AUC | 0.7285 | 0.7271 | 0.7179 | 0.7449 | 0.7600 | **0.8059** | 0000
| | RMSE | 0.4328 | 0.4348 | 0.4381 | 0.4413 | 0.4337 | _0.4063_  | 0000
|assist12 | AUC | 0.7006 | 0.7011 | 0.6998 | 0.7505 | 0.7314 | **0.8130**  | 0000
| | RMSE | 0.4338 | 0.4355 | 0.4360 | 0.4250 | 0.4284 |  **0.3935**  | 0000
|assist17 | AUC | **0.7220** | 0.7095 | 0.6792 | 0.6803 | 0.6738 | _0.7141_  | 0000
| | RMSE | **0.4469** | _0.4516_ | 0.4591 | 0.4722 | 0.4713 | 0.4630  | 0000
|algebra05 | AUC | 0.8088 | 0.8146 | 0.8162 | 0.7673 | 0.7871 | **0.8201**  | 0000
| | RMSE | 0.3703 | 0.3687 | _0.3685_ | 0.3918 | 0.3824 | **0.3584**  | 0000
|algebra06 | AUC | 0.7939 | 0.7961 | 0.7927 | 0.7505 | 0.7789 |  **0.8064**  | 0000
| | RMSE | _0.3666_ | **0.3661** | 0.3675 | 0.3986 | 0.3863 | 0.3672  | 0000
|EdNet | AUC | 0.6609 | 0.6602 | 0.6506 | 0.6687 | 0.6651 | **0.7336**  | 0000
| | RMSE | 0.4598 | 0.4597 | 0.4629 | 0.4783 | 0.4750 | **0.4516**  | 0000


# Setups

1. We used docker environments, **ufoym/deefo**.  
   [https://hub.docker.com/r/ufoym/deepo/](https://hub.docker.com/r/ufoym/deepo/)
2. If you don't use docker environments, then you can use **requirements.txt**.

   ```
   pip install -r requirements.txt
   ```
3. You can download the preprocessed dataset from our Google Drive.
   [https://drive.google.com/drive/folders/1B5vHabKUwzGEQAyDaj_0cBT6UqTMA13L?usp=sharing](https://drive.google.com/drive/folders/1B5vHabKUwzGEQAyDaj_0cBT6UqTMA13L?usp=sharing)

4. However, the dataset with text of question was not opened to public, because the source of dataset is from EdTech company. So if you want to use sim_diff_llm_loader then you have to use your personal dataset with text.

5. If you want to preprocess yourself, you can use **preprocess_data.py**.

   ```
   python preprocess_data.py --data_name assist09 --min_user_inter_num 5
   ```

# How to run this code?

If you want to run the DCL4KT+LLM, you can user train.sh. All of the options are contained in the define_argparser.py.


# Errata

If you have any question or find error in the code, you can send me a mail.

Contact: Unggi Lee ([codingchild@korea.ac.kr](mailto:codingchild@korea.ac.kr)).
