
# TemMed-Bench: Evaluating Temporal Medical Image Reasoning in Vision-Language Models

[**🌐 Homepage**](https://t) | [**🤗 Dataset**](https://t) | [**📖 Paper**](https://t)



## News


* 🔥 [2024-10-10] MRAG-Bench is released.


## Intro

<img src="./misc/Teaser_Figure.png" width="1000" />

TemMed-Bench features three primary highlights. 
- (1) **Temporal reasoning focus**: each sample in TemMed-Bench includes historical condition information, challenging models to analyze changes in patient conditions over time. 
- (2) **Multi-image input**: each sample in TemMed-Bench contains multiple images from different visits as input, emphasizing the need for models to process and reason over multiple images. 
- (3) **Diverse task suite**: TemMed-Bench comprises three tasks including VQA, report generation, and image-pair selection. These tasks are all built upon a test set consisting of 1,000 samples. Additionally, TemMed-Bench includes a knowledge corpus with over 17,000 instances.




## Results


<img src="./misc/Task_Figure.png" width="800" />


- We conducted extensive experiments on TemMed-Bench to evaluate six proprietary and six open-source LVLMs. The results show that most LVLMs lack the ability to analyze changes in patients’ conditions across temporal medical images. 

  - In the VQA task, GPT-4o-mini and Claude 3.5 Sonnet achieved accuracies of 79.15% and 69.90%, respectively, while most LVLMs scored below 60\%. For the more challenging tasks of report generation and image-pair selection, all LVLMs underperformed, with the highest average BLEU, ROUGE-L, and METEOR score at 20.67 for report generation and a top accuracy of 39.33% for image-pair selection in a three-option setting. These results reveal a fundamental gap in current LVLM training, i.e., lack of focus on temporal image reasoning.

- Given the limited performance of current LVLMs in tracking condition changes under the zero-shot setting, we adopt the Retrieval-Augmented Generation (RAG) framework for evaluation. In addition to augmenting the input with retrieved textual information, we further explore augmenting the input with both retrieved visual and textual modalities in the medical domain. 

  - Experimental results demonstrate that augmenting input with both visual and textual information substantially boosts performance for most models compared to text-only augmentation. Notably, HealthGPT exhibits an accuracy improvement of over 10% in the VQA task when augmented with multi-modal retrieved information.



## Load Dataset

- Coming Soon

## Evaluation 

- Coming Soon


## Contact

* Junyi Zhang: JunyiZhang2002@g.ucla.edu


## Citation

- Coming Soon
