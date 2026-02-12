## About

This is a tool to search in up to trillion-scale corpora, finding not just exact matches but also similar matches from the corpus data. In our experiments on an AWS instance, the median latency was **less than 90 milliseconds**, and the p95 latency was **less than 300 milliseconds for more than 6TB of text data**.

~~~
| Rank | Score | #Match | String
---------------------------------------------------
|    1 | 100.0 |    838 | olympics gold medalist
|    2 |  89.1 |    456 | olympics gold medallist
|    3 |  86.4 | 95,816 | olympic gold medalist
|    4 |  84.8 |    500 | olympics silver medalist
|    5 |  82.0 | 22,506 | olympic gold medallist
|    6 |  80.9 |     75 | olympics silver medallist
|    7 |  79.6 |  8,945 | olympic silver medalist
|    8 |  77.1 |  2,409 | olympic silver medallist
|    9 |  75.1 |    128 | olympics , gold medalist
|   10 |  73.1 |      2 | olympics . gold medalist
|   11 |  71.0 |     63 | olympics gold medalists
|   12 |  69.3 | 10,292 | olympic gold medalists
|   13 |  69.2 |      7 | olympics gold champion
|   14 |  69.1 |      1 | olympics gold olympic
|   15 |  68.9 |      7 | olympics silver medalists
|   16 |  67.7 |    160 | olympic gold champion
|   17 |  67.6 |     11 | olympic gold olympic
|   18 |  67.5 |    467 | olympic silver medalists
|   19 |  66.9 |     15 | olympics , gold medallist
|   20 |  65.6 |    196 | paralympics gold medalist
~~~

## Quick Start

### 1. Compilation

The first step is to compile the program using the following commands:

~~~bash
$ uv sync
$ uv run maturin develop --release --manifest-path rust/Cargo.toml
~~~

### 2. Build Index: `softmatcha-index`

The next step is to build indices with the following command. The final filesize is typically ~10x the size of the raw text for small corpora, but less than 3x for larger corpora.

~~~bash
$ softmatcha-index --index [index directory] [text file]

Example:
$ softmatcha-index --index corpus corpus.txt
~~~

For faster indexing, we recommend setting the indexing memory usage via `--mem_size`. For faster search, we also recommend setting the search memory usage via `--mem_size_ex`. 

Note that a large `mem_size_ex` increases loading time, so we suggest using a lower value (e.g., 100) for small corpora.

~~~bash
$ softmatcha-index --index corpus --mem_size=5000 --mem_size_ex=1000
~~~

(5,000MB memory for indexing, 1,000MB memory for execution (search))

### 3. Search: `softmatcha-search`

Finally, you can search for phrases using the following command:

~~~bash
$ softmatcha-search --index [index directory] [pattern]

Example:
$ softmatcha-search --index corpus "olympics gold medalist"
~~~

To adjust the number of outputs, similarity thresholds, or max runtime, use the following options:

~~~bash
Example:
$ softmatcha-search --index corpus --num_candidates=100 --min_similarity=0.2 --max_runtime=20 "olympics gold medalist"
~~~

### 4. Output Examples: `softmatcha-exact`

You can also search for exact match examples (KWIC) with the following commands:

~~~bash
$ softmatcha-exact --index [index directory] [pattern]

Example:
$ softmatcha-exact --index corpus "olympics gold medalist"
$ softmatcha-exact --index corpus --display=20 --padding=200 "olympics gold medalist" # Output up to 20 examples with +/- 200 bytes context
~~~

<br />

## Multilingual Support

To search in languages other than English, build an index by specifying the backend model:

~~~bash
$ softmatcha-index --index corpus --backend=fasttext --model=fasttext-ja-vectors corpus.txt
$ softmatcha-index --index corpus --backend=fasttext --model=fasttext-zh-vectors corpus.txt
$ softmatcha-index --index corpus --backend=fasttext --model=fasttext-fr-vectors corpus.txt
$ softmatcha-index --index corpus --backend=fasttext --model=fasttext-de-vectors corpus.txt
$ softmatcha-index --index corpus --backend=fasttext --model=fasttext-it-vectors corpus.txt
~~~

Then, perform the search as follows:

~~~bash
$ softmatcha-search --index corpus --backend=fasttext --model=fasttext-ja-vectors "金メダル"
$ softmatcha-search --index corpus --backend=fasttext --model=fasttext-zh-vectors "中国"
$ softmatcha-search --index corpus --backend=fasttext --model=fasttext-fr-vectors "France"
$ softmatcha-search --index corpus --backend=fasttext --model=fasttext-de-vectors "Deutschland"
$ softmatcha-search --index corpus --backend=fasttext --model=fasttext-it-vectors "Italia"
~~~

<br />

## Citation

~~~
@article{yoneda-preprint-2026-softmatcha2,
  title         = "{SoftMatcha 2: A Fast and Soft Pattern Matcher for
                   Trillion-scale Corpora}",
  author        = "Yoneda, Masataka and Matsushita, Yusuke and Kamoda, Go and
                   Suenaga, Kohei and Akiba, Takuya and Waga, Masaki and Yokoi,
                   Sho",
  journal       = "arXiv [cs.CL]",
  month         =  "11~" # feb,
  year          =  2026,
  url           = "http://dx.doi.org/10.48550/arXiv.2602.10908",
  archivePrefix = "arXiv",
  primaryClass  = "cs.CL",
  doi           = "10.48550/arXiv.2602.10908"
}
~~~

<br />

## License

This software is mainly developed by [Masataka Yoneda](https://sites.google.com/view/e869120-webpage/home) and published under Apache License 2.0.
