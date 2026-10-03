# Confirmation tool-retrieval failure analysis

This exploratory analysis describes **1500 held-out confirmation queries after the frozen routing predictions and evaluation were saved**. It does not train a router, change routing decisions, or alter relevance labels. Its case examples and hypotheses are descriptive diagnostics, not additional primary tests.

## Candidate coverage and ranking changes

| Source | Queries | Hybrid nDCG@10 | Always-CE nDCG@10 | CE delta | Wins / losses / unchanged | Mean candidate-label recall@20 | No positive in prefix | Gold left / entered top 10 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| apigen | 500 | 0.585090 | 0.614499 | +0.029409 | 130 / 106 / 264 | 0.8010 | 65 | 23 / 35 |
| toolace | 500 | 0.618552 | 0.637245 | +0.018693 | 118 / 107 / 275 | 0.7882 | 85 | 19 / 25 |
| toolbench | 500 | 0.358821 | 0.320424 | -0.038396 | 116 / 184 / 200 | 0.5671 | 132 | 94 / 47 |

Candidate recall is averaged per query using original exact-ID labels. A top-20 permutation cannot recover a positive tool outside that prefix. Positive-label counts do not establish which tools are mandatory for executing the request.

## Exact tokenizer replay and unjudged promotions

| Source | Truncated prefix pairs / total | Truncated positive pairs / positive prefix pairs | Query-truncated pairs | Unjudged cross-source CE top 1 | Name-counterpart hint at CE top 1 |
|---|---|---|---:|---:|---:|
| apigen | 921 / 10000 | 17 / 481 | 17 | 187 | 37 |
| toolace | 2804 / 10000 | 87 / 489 | 1391 | 123 | 14 |
| toolbench | 1752 / 10000 | 70 / 637 | 13 | 330 | 61 |

The pinned CE tokenizer is replayed with the original flattened tool serialization and longest-first truncation at 256 tokens. Token_type_ids and special-token masks give retained query/document counts. Name-counterpart hints require matching normalized names or a complete suffix match with the shorter name at least 12 characters; they are not executable-equivalence judgments. Unjudged tools are not automatically wrong, and the qrels remain unchanged.

## Frozen router's selected and bypassed outcomes

| Policy | Required CE calls | Selected CE wins / losses / neutral | Bypassed CE wins / losses / neutral | Mean policy minus always nDCG |
|---|---:|---|---|---:|
| fixed_disagreement | 1077 | 306 / 302 / 469 | 58 / 95 / 270 | +0.007316 |
| cheap_jaccard | 1391 | 335 / 364 / 692 | 29 / 33 / 47 | -0.000126 |
| utility_25 | 409 | 114 / 112 / 183 | 250 / 285 / 556 | +0.002845 |
| utility_50 | 829 | 235 / 217 / 377 | 129 / 180 / 362 | +0.006469 |
| utility_75 | 1149 | 304 / 310 / 535 | 60 / 87 / 204 | +0.003201 |

A bypassed CE win is gain forgone; a bypassed CE loss is harm avoided. These labels are assigned only after applying frozen predictions. They are not features available to a deployable router. Required call counts are checked against the evaluator; actual execution and repeated latency are separate artifacts.

## Descriptive ToolBench truncation breakdown

| Any positive pair truncated | Queries | CE wins / losses / unchanged | Mean CE minus hybrid nDCG |
|---|---:|---|---:|
| no | 442 | 98 / 152 / 192 | -0.035313 |
| yes | 58 | 18 / 32 / 8 | -0.061891 |

These post-outcome strata are confounded by query, schema, source, and annotation differences. They do not establish that truncation causes or prevents a reranking change. JSON also contains positive-count, query-length, gold-document-length, and candidate-coverage strata.

## Deterministic confirmation cases

For each source, the five largest losses and five largest wins are selected by signed nDCG delta and query-ID tie break. These are outcome-selected examples and cannot estimate population frequencies. No manual notes from development are reused.

### apigen_query_215 (-1.000000 CE delta)

Labeled positive(s) leave top 10. CE top 1 is unjudged and comes from another corpus source.

Query excerpt: Retrieve science news articles in Spanish for Spain and Mexico.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_342 (science) | 1 | 15 | 54 | false |

CE top 1: `toolACE_tool_12575` — CoinTelegraph News API (hybrid rank 7; labeled positive: False).

### apigen_query_85 (-0.684535 CE delta)

CE top 1 is unjudged and comes from another corpus source.

Query excerpt: Decode the VIN numbers 'WAUZZZ8V5BN066654', 'WBA3B9C58EN176718', '1HGCM82633A073743', and '1GKS2JKY1J5126157' to find out the car model, maker, and year, and the engine details.

Frozen utility75 decision: bypass.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_431 (vin_decoder) | 1 | 8 | 68 | false |

CE top 1: `toolACE_tool_7993` — AU Decode (hybrid rank 13; labeled positive: False).

### apigen_query_560 (-0.613147 CE delta)

CE top 1 is unjudged and comes from another corpus source.

Query excerpt: What is the current stock quote for Tesla in the US market and for Volkswagen in the German market?

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_1817 (stock_get_stock_quote_type_data) | 1 | 5 | 134 | false |

CE top 1: `toolACE_tool_7416` — Get Live Equity Quote (hybrid rank 3; labeled positive: False).

### apigen_query_883 (-0.613147 CE delta)

Ordering changes within the fixed scored prefix.

Query excerpt: Retrieve 15 records from the 'Birmingham' region.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_445 (fetch_by_region) | 1 | 5 | 47 | false |

CE top 1: `apigen_tool_2180` — shows_id_episodes (hybrid rank 15; labeled positive: False).

### apigen_query_892 (-0.613147 CE delta)

Labeled positive(s) leave top 10. Some positive labels are absent from the scored prefix. CE top 1 is unjudged and comes from another corpus source.

Query excerpt: List all surebets from bookmakers 'Unibet' and '10Bet', and get detailed stats for the first 20 fighters in the UFC Fight Night on June 17, 2023.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_2468 (ufc_fight_night_vettori_vs_cannonier_june_17_2023) | 1 | 11 | 113 | false |
| apigen_tool_680 (list) | absent | absent | 185 | not scored |

CE top 1: `toolACE_tool_13853` — Get All UFC Fighters (hybrid rank 18; labeled positive: False).

### apigen_query_295 (+1.000000 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: Get the fan ratings for two different events: a basketball game with ID 67890 and a tennis match with ID 54321.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_2610 (fan_rating) | 12 | 1 | 52 | false |

CE top 1: `apigen_tool_2610` — fan_rating (hybrid rank 12; labeled positive: True).

### apigen_query_424 (+1.000000 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: Fetch the latest news in English from the United States.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_195 (latest) | 11 | 1 | 46 | false |

CE top 1: `apigen_tool_195` — latest (hybrid rank 11; labeled positive: True).

### apigen_query_662 (+1.000000 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: Can you fetch the straddle options data for Tesla (TSLA) and Amazon (AMZN) stocks?

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_2016 (straddle) | 13 | 1 | 56 | false |

CE top 1: `apigen_tool_2016` — straddle (hybrid rank 13; labeled positive: True).

### apigen_query_67 (+1.000000 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: Simulate a database query on a 'products' table with conditions {'price': '<100', 'category': 'electronics'}.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_28 (simulate_query_database) | 18 | 1 | 44 | false |

CE top 1: `apigen_tool_28` — simulate_query_database (hybrid rank 18; labeled positive: True).

### apigen_query_690 (+0.735145 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: Describe the characteristics of easy weed strains and the OBD2 code P0100.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| apigen_tool_156 (difficulty) | 9 | 1 | 55 | false |
| apigen_tool_157 (obd2_code) | 11 | 3 | 54 | false |

CE top 1: `apigen_tool_156` — difficulty (hybrid rank 9; labeled positive: True).

### toolACE_query_24 (-1.000000 CE delta)

Labeled positive(s) leave top 10. CE top 1 is unjudged and comes from another corpus source.

Query excerpt: user:I need to remove an outdated inventory table from our main database. Can you handle that for me?


Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_12211 (dropTable) | 1 | 13 | 45 | false |

CE top 1: `rotbench_tool_6` — ask_to_user (hybrid rank 9; labeled positive: False).

### toolACE_query_345 (-1.000000 CE delta)

Labeled positive(s) leave top 10.

Query excerpt: user:I need to check the rounds for the tournaments with ids 125, 327, and 871 for the season with id 2022.


Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_10311 (Get Tournaments Rounds) | 1 | 12 | 56 | false |

CE top 1: `toolACE_tool_12218` — Get League Rounds (hybrid rank 2; labeled positive: False).

### toolACE_query_54 (-1.000000 CE delta)

Labeled positive(s) leave top 10.

Query excerpt: user:Could you please generate a unique identifier for a new project I am starting? I would prefer it without dashes.


Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_3752 (Generate UUID) | 1 | 13 | 71 | false |

CE top 1: `toolACE_tool_13159` — addTask (hybrid rank 10; labeled positive: False).

### toolACE_query_575 (-0.710935 CE delta)

CE top 1 is unjudged and comes from another corpus source.

Query excerpt: user:Hey there, darling virtual assistant! Could you work some of your digital magic and transform this page I’m stalking into a snazzy PDF for me? Here’s the URL: https://www.example.com/amazing-web-page


Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_10920 (Convert Web Page to PDF) | 1 | 10 | 136 | false |

CE top 1: `toolbench_tool_7025` — PragmavantApi_web_pdf (hybrid rank 11; labeled positive: False).

### toolACE_query_923 (-0.710935 CE delta)

Ordering changes within the fixed scored prefix.

Query excerpt: user:Hi there, I am curious if you have any capabilities related to document management, specifically PDF files?

assistant:Of course, in terms of managing PDF documents, I can retrieve sound annotations from a specific page of a PDF...

Frozen utility75 decision: bypass.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_6639 (GetPageSoundAnnotations) | 1 | 10 | 85 | false |

CE top 1: `toolACE_tool_13460` — GetComboBoxField (hybrid rank 5; labeled positive: False).

### toolACE_query_26 (+1.000000 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: user:I need to check the size of a recent dataset we compiled for the new project. Could you assist with that?

assistant:Could you please specify which dataset you are referring to?

user:It's the dataset named "October2021_ProjectX_Data."

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_2046 (getDataSize) | 14 | 1 | 35 | false |

CE top 1: `toolACE_tool_2046` — getDataSize (hybrid rank 14; labeled positive: True).

### toolACE_query_436 (+1.000000 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: user:Could you generate some relevant hashtags for my latest TikTok video? The topic is cooking.
assistant:Here are some relevant hashtags for your latest TikTok video on cooking:

- #cooking
- #food
- #recipe
- #homecooking
-...

Frozen utility75 decision: bypass.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_743 (TikTok Hashtag Generator) | 19 | 1 | 56 | true |

CE top 1: `toolACE_tool_743` — TikTok Hashtag Generator (hybrid rank 19; labeled positive: True).

### toolACE_query_476 (+1.000000 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: user:I need two timers: one for 5 minutes and another for 10 minutes.


Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_13058 (timer) | 12 | 1 | 29 | false |

CE top 1: `toolACE_tool_13058` — timer (hybrid rank 12; labeled positive: True).

### toolACE_query_547 (+1.000000 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: user:I want to analyze the performance of the electronics category, specifically for the first quarter of 2023. Can you fetch me the performance metrics?


Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_13705 (CategoryPerformanceTracker.fetchPerformanceMetrics) | 11 | 1 | 181 | false |

CE top 1: `toolACE_tool_13705` — CategoryPerformanceTracker.fetchPerformanceMetrics (hybrid rank 11; labeled positive: True).

### toolACE_query_984 (+1.000000 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: user:I want to know about the fighters with the IDs 102, 256 and 345. Can you get the information for me?


Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolACE_tool_7682 (Get Fighter) | 13 | 1 | 20 | false |

CE top 1: `toolACE_tool_7682` — Get Fighter (hybrid rank 13; labeled positive: True).

### toolbench_query_81 (-0.735932 CE delta)

Labeled positive(s) leave top 10. CE top 1 is unjudged and comes from another corpus source. CE top 1 has a name-counterpart hint; equivalence is unverified.

Query excerpt: I need to keep track of the token forwarding transactions for my company. Please retrieve the token forwarding transactions, including the offset, limit, total count, and the list of token forwarding transactions. Additionally, I want...

Frozen utility75 decision: bypass.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_238 (Token_Forwarding_Get_token_forwarding_transactions) | 1 | 4 | 140 | false |
| toolbench_tool_239 (Token_Forwarding_Get_usage_quota_for_the_current_month) | 2 | 15 | 53 | false |

CE top 1: `toolACE_tool_8266` — Get Token Forwarding Transactions (hybrid rank 3; labeled positive: False).

### toolbench_query_306 (-0.655653 CE delta)

Labeled positive(s) leave top 10. CE top 1 is unjudged and comes from another corpus source.

Query excerpt: I need to analyze the performance of a mutual fund. Could you fetch the latest price and historical prices of the fund with ISIN LU0690375182? It would be great if you could provide the prices from 2015-01-25 to 2020-12-31.

Frozen utility75 decision: bypass.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_2311 (Funds_v1GetFundLatestPrice) | 3 | 11 | 63 | false |
| toolbench_tool_2312 (Funds_v1GetFundHistoricalPrices) | 1 | 4 | 109 | false |

CE top 1: `apigen_tool_1992` — v1getfundlatestprice (hybrid rank 2; labeled positive: False).

### toolbench_query_24 (-0.636439 CE delta)

Labeled positive(s) leave top 10. CE top 1 is unjudged and comes from another corpus source.

Query excerpt: I'm planning a joke-themed party for my friends and I need some fresh material. Can you provide me with the joke of the day? Also, it would be great if you could give me a random joke to add some variety to the party. Finally, I'd like...

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_4554 (Dad_Jokes_v2_dad_jokes_joke_of_the_day) | 1 | 5 | 50 | false |
| toolbench_tool_4556 (Dad_Jokes_v2_dad_jokes_random) | 5 | 15 | 23 | false |
| toolbench_tool_4557 (Dad_Jokes_v2_dad_jokes_health) | 6 | 20 | 32 | false |

CE top 1: `toolACE_tool_5396` — Get Joke of the Day By Category (hybrid rank 16; labeled positive: False).

### toolbench_query_143 (-0.613147 CE delta)

Labeled positive(s) leave top 10. Some positive labels are absent from the scored prefix.

Query excerpt: Can you retrieve the stock price and details for the symbol GOOGL? Also, fetch the conversion rate from USD to JPY for an amount of 1254.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_2056 (YH_Finance_Complete_Stock_Price) | 128 | 128 | 50 | not scored |
| toolbench_tool_2057 (YH_Finance_Complete_Currency_Converter) | 1 | 11 | 144 | false |

CE top 1: `toolbench_tool_7537` — Yahoo_Finance_price (hybrid rank 9; labeled positive: False).

### toolbench_query_416 (-0.613147 CE delta)

Labeled positive(s) leave top 10. Some positive labels are absent from the scored prefix.

Query excerpt: I'm hosting a movie-themed party and I want to display movie suggestions. Can you suggest four related movies for the party? Also, fetch the list of favorite libraries for user 1 from the Python Libraries tst API.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_2541 (List_Movies_v3_Movie_Suggestions) | 60 | 60 | 195 | not scored |
| toolbench_tool_9261 (Python_Libraries_tst_View_User_List) | 1 | 20 | 57 | false |

CE top 1: `toolbench_tool_10572` — YTS_am_Torrent_Movie_Suggestions_XML (hybrid rank 5; labeled positive: False).

### toolbench_query_716 (+0.765361 CE delta)

Labeled positive(s) enter top 10. Some positive labels are absent from the scored prefix.

Query excerpt: I'm planning a surprise birthday party for my sister and I want to create personalized invitations. Can you provide me with some templates and their details from Nexweave? Additionally, I need icons related to birthdays and celebrations...

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_5717 (Nexweave_GetAllTemplates) | 15 | 1 | 48 | false |
| toolbench_tool_5718 (Nexweave_GetTemplateDetails) | 13 | 2 | 50 | false |
| toolbench_tool_9005 (Unofficial_Icons8_Search_Search) | absent | absent | 184 | not scored |

CE top 1: `toolbench_tool_5717` — Nexweave_GetAllTemplates (hybrid rank 15; labeled positive: True).

### toolbench_query_207 (+0.604296 CE delta)

Labeled positive(s) enter top 10.

Query excerpt: I work for a finance company and we need to analyze the earnings history and estimate for a particular stock. Can you provide us with the earnings history and estimate for the ticker symbol 'AAPL'?

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_2095 (Stock_Analysis_Earnings_History) | 14 | 1 | 119 | false |
| toolbench_tool_2096 (Stock_Analysis_Earnings_Estimate) | 8 | 9 | 141 | false |

CE top 1: `toolbench_tool_2095` — Stock_Analysis_Earnings_History (hybrid rank 14; labeled positive: True).

### toolbench_query_192 (+0.500000 CE delta)

Ordering changes within the fixed scored prefix.

Query excerpt: Help me find the issuer card information by entering the first 6 digits of a credit/debit card using the BIN/IIN Lookup API. I would like to know the details for a card with the BIN '470886'.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_8967 (BIN_IIN_Lookup_BIN_IIN_Lookup) | 3 | 1 | 61 | false |

CE top 1: `toolbench_tool_8967` — BIN_IIN_Lookup_BIN_IIN_Lookup (hybrid rank 3; labeled positive: True).

### toolbench_query_37 (+0.491149 CE delta)

Labeled positive(s) enter top 10. CE top 1 is unjudged and comes from another corpus source.

Query excerpt: I want to know the list of categories available in the advertising tool. Also, get the product details for the products in the 'Electronics' category.

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_3065 (asdfadsf_Get_Products_in_Category) | 14 | 3 | 46 | false |
| toolbench_tool_3068 (asdfadsf_Get_Categories) | 17 | 9 | 28 | false |

CE top 1: `toolACE_tool_5719` — shopping.compare_products (hybrid rank 6; labeled positive: False).

### toolbench_query_71 (+0.476053 CE delta)

Labeled positive(s) enter top 10. Some positive labels are absent from the scored prefix.

Query excerpt: My company is interested in partnering with a business on WhatsApp. Can you help us determine if the phone number +1 555-555-5555 is registered on WhatsApp? Additionally, we would like to fetch the profile picture of this number in low...

Frozen utility75 decision: rerank.

| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |
|---|---:|---:|---:|---|
| toolbench_tool_3822 (Whatsapp_Scraper_Fetch_profile_picture) | 19 | 1 | 176 | true |
| toolbench_tool_3824 (Whatsapp_Scraper_Fetch_business_info) | 9 | 8 | 135 | false |
| toolbench_tool_3825 (Whatsapp_Scraper_Is_registered_on_whatsapp) | 26 | 26 | 113 | not scored |

CE top 1: `toolbench_tool_3822` — Whatsapp_Scraper_Fetch_profile_picture (hybrid rank 19; labeled positive: True).

## Limits and interpretation

- This diagnostic does not alter the frozen primary hypothesis, noninferiority margin, router, or thresholds.
- Gold-tool component separation is not verified API-family separation or proof of clean pretrained models.
- The exact-ID benchmark does not evaluate execution success or adjudicate every plausible alternate tool.
- A controlled ablation or independently judged equivalence set is needed to test causal explanations.
- Hash checks link these outputs to the immutable predictions and scored cohort; chronology is verified against the evaluator's provenance declaration and source hashes, not an external timestamp authority.

## Provenance

```json
{
  "cache_sha256": "e8eaab43968c191fa73ad50ea0b80b52ca64e9799e43fea1aed58fa89c4ef7ad",
  "cache_manifest_sha256": "cfadbc419a7fd84412e0d5f8fd9980c50c9c7c71a2b2490041a09913585c78a3",
  "queries_sha256": "0aa1f8921c13e44997f7b4018198db513a6d67a0258a550c6fafc16c705cc701",
  "corpus_sha256": "ac192e7a6d2b1930f3f61544748ce81741c372a104674a3520f8214b2ac3c09f",
  "predictions_sha256": "d2dd099e3db31009bc5de3f92e5b89445516bbd2d2972307679dce6fc55e37b2",
  "prediction_provenance_sha256": "84c8d52236daccb850d9cb125bf83db10f19d6c1f30343e9a24207a5f0beacae",
  "evaluation_summary_sha256": "fda5bd3e67a6990c2e95076779c4f3dedf8faaa9ab7a7325eb6dfc5fee6f55d3",
  "router_sha256": "f24613a552d36737f4a17797b94f0c9348a67072a97cfc3cb471ae4bd25c159c",
  "protocol_sha256": "d33ceee4741679c5027b7b55cc96f5535f98aef0c2160292d8e0c948abd076bc",
  "data_manifest_sha256": "45d68454ecc950c56b2ecf9fd436cb4a6dafc780e4e64d3e154ef81fc304b2ea",
  "protocol_frozen_commit": "3e813e1af650104f33aaf77febce5a8c1a3f1213",
  "declared_predictions_created_before_quality_scoring": true,
  "created_after_frozen_predictions": true,
  "labels_altered": false,
  "analysis_modifies_rankings_or_router": false,
  "script_sha256": "430532f568238c915aa335e1ea31edad47dd3d25a751c2687d73d0e7d9ceb513",
  "reused_analysis_script_sha256": "d5bbd736aa59c226832f427b34c3d3b2cb0741ac59dd58afeccbb69d8bf11688",
  "text_serialization_sha256": "70bd54a54acf7da4c13d8a5c2e5b379f81b64f19f248681973951c79b99f0750"
}
```
