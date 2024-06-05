Download the data from https://drive.google.com/file/d/1crRAVPNpCf6dI0XQVSAyS1oIawafC-_D/view?usp=drive_link to data folder.

@inproceedings{zhou-etal-2023-clcl,
    title = "{CLCL}: Non-compositional Expression Detection with Contrastive Learning and Curriculum Learning",
    author = "Zhou, Jianing  and
      Zeng, Ziheng  and
      Bhat, Suma",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.43",
    doi = "10.18653/v1/2023.acl-long.43",
    pages = "730--743",
    abstract = "Non-compositional expressions present a substantial challenge for natural language processing (NLP) systems, necessitating more intricate processing compared to general language tasks, even with large pre-trained language models. Their non-compositional nature and limited availability of data resources further compound the difficulties in accurately learning their representations. This paper addresses both of these challenges. By leveraging contrastive learning techniques to build improved representations it tackles the non-compositionality challenge. Additionally, we propose a dynamic curriculum learning framework specifically designed to take advantage of the scarce available data for modeling non-compositionality. Our framework employs an easy-to-hard learning strategy, progressively optimizing the model{'}s performance by effectively utilizing available training data. Moreover, we integrate contrastive learning into the curriculum learning approach to maximize its benefits. Experimental results demonstrate the gradual improvement in the model{'}s performance on idiom usage recognition and metaphor detection tasks. Our evaluation encompasses six datasets, consistently affirming the effectiveness of the proposed framework. Our models available at \url{https://github.com/zhjjn/CLCL.git}.",
}
