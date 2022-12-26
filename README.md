# IDG
 This work studies and solves imbalanced domain generalization (IDG) via generative inference network (GINet).
 
## Abstract
 Domain generalization (DG) aims to learn transferable knowledge from multiple source domains and generalize it to the unseen target domain. To achieve such expectation, the intuitive solution is to seek domain-invariant representations via generative adversarial mechanism or minimization of cross-domain discrepancy. However, the widespread imbalanced data scale problem across source domains and category in real-world applications becomes the key bottleneck of improving generalization ability of model due to its negative effect on learning the robust classification model. Motivated by this observation, we first formulate a practical and challenging imbalance domain generalization (IDG) scenario, and then propose a straightforward but effective novel method generative inference network (GINet), which augments reliable samples for minority domain/category to promote discriminative ability of the learned model. Concretely, GINet utilizes the available cross-domain images from the identical category and estimates their common latent variable, which derives to discover domain-invariant knowledge for unseen target domain. According to these latent variables, our GINet further generates more novel samples with optimal transport constraint and deploys them to enhance the desired model with more robustness and generalization ability. Considerable empirical analysis and ablation studies on three popular benchmarks under normal DG and IDG setups suggests the advantage of our method over other DG methods on elevating model generalization.
 
## Datasets
 In this work, we mainly conducted experiments on three popular benchmarks: [PACS](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ), [VLCS](https://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file) and [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw). When doing experiments, you should download the corresponding datasets and place them under "datasets" folder. Moreover, for imbalanced domain generalization setting, the statistics of datasets per task are clearly illustrated in "appendix.pdf" and utilized to organize multiple source domain and the unseen target domain.
 
## Dependencies

```python
 python==3.6.2  
 pytorch==1.0.0  
 torchvision==0.2.2  
 numpy==1.16.2  
 scikit-learn==0.20.2  
 scipy==1.1.0  
 opencv-python==4.0.0 
```
 
## Training & Evalution
```python
python main.py
```
Notes: "main.py" is under "main" folder and includes training and test phases. For the specific training parameter, you can modify them in "config.py" under the same folder and perform the corresponding experiments. The logs of training procedure and the results of test stage will be stored under "results" folder.

## Citation

If you think this work is interesting, please cite:
```
Haifeng Xia, Taotao Jing and Zhengming Ding. Generative Inference Network for Imbalanced Domain Generalization. (Under Review).
```
## Contact

If you have any questions on this work, feel free to contact

- hxia@tulane.edu
